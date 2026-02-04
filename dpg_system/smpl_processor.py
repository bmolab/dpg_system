import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import pickle
import sys
import time
from dataclasses import dataclass
import logging
from dpg_system.one_euro_filter import OneEuroFilter

try:
    import torch
    import smplx
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False
    print("Warning: smplx or torch not found. Dynamic anthropometry will use approximate heuristics.")


class NumpySmartClampKF:
    """
    A Linear Kalman Filter with 'Smart Innovation Clamping' for smoothing
    multi-dimensional signals while rejecting sudden spikes.
    
    This filter limits the magnitude of innovation (correction) that can be applied
    per frame, preventing glitches from teleporting the filter state while still
    tracking smooth motion accurately.
    
    Used for dynamic torque smoothing in the rate limiter.
    """
    def __init__(self, dt, num_streams, dim):
        """
        Args:
            dt: Time step (seconds)
            num_streams: Number of independent streams (e.g., 22 joints)
            dim: Dimensionality per stream (e.g., 3 for xyz torque vectors)
        """
        self.dt = dt
        self.num_streams = num_streams
        self.dim = dim
        self.state_dim = 2 * dim  # Position + Velocity
        
        # State: [Position, Velocity] per stream
        # Shape: (num_streams, state_dim)
        self.x = np.zeros((num_streams, self.state_dim))
        
        # Error Covariance P: (num_streams, state_dim, state_dim)
        self.P = np.tile(np.eye(self.state_dim) * 0.1, (num_streams, 1, 1))
        
        # State Transition Matrix F: [[I, dt*I], [0, I]]
        self.F = np.eye(self.state_dim)
        self.F[:dim, dim:] = np.eye(dim) * dt
        
        # Measurement Matrix H: [I, 0] - we only measure position
        self.H = np.zeros((dim, self.state_dim))
        self.H[:dim, :dim] = np.eye(dim)
        
        # Process and Measurement Noise (defaults, updated by update_params)
        self.Q = np.eye(self.state_dim) * 0.01
        self.R = np.eye(dim) * 0.1
        
        # Innovation clamp radius
        self.clamp_radius = 15.0
        
        # Flags
        self._initialized = False
    
    def update_params(self, responsiveness, smoothness, clamp_radius, dt):
        """
        Update filter parameters.
        
        Args:
            responsiveness: Process noise (higher = faster tracking, 1-100)
            smoothness: Measurement noise (higher = smoother output, 0.1-10)
            clamp_radius: Max innovation magnitude per frame
            dt: Time step
        """
        self.dt = dt
        self.clamp_radius = clamp_radius
        
        # Update F with new dt
        self.F[:self.dim, self.dim:] = np.eye(self.dim) * dt
        
        # Build Q (Process Noise Covariance)
        # Position variance scales with dt^2, velocity variance scales with 1
        pos_var = (responsiveness * dt) ** 2
        vel_var = responsiveness ** 2
        q_diag = np.concatenate([np.full(self.dim, pos_var), np.full(self.dim, vel_var)])
        self.Q = np.diag(q_diag)
        
        # Build R (Measurement Noise Covariance)
        self.R = np.eye(self.dim) * smoothness
    
    def predict(self):
        """
        Prediction step: propagate state and covariance forward.
        """
        if not self._initialized:
            return
        
        # x = F @ x
        self.x = self.x @ self.F.T
        
        # P = F @ P @ F^T + Q (applied per stream)
        for i in range(self.num_streams):
            self.P[i] = self.F @ self.P[i] @ self.F.T + self.Q
    
    def update(self, z):
        """
        Update step with smart innovation clamping.
        
        Args:
            z: Measurement array of shape (num_streams, dim)
            
        Returns:
            Filtered position estimate of shape (num_streams, dim)
        """
        # Initialize on first measurement
        if not self._initialized:
            self.x[:, :self.dim] = z
            self._initialized = True
            return z.copy()
        
        # Innovation: y = z - H @ x = z - x[:, :dim]
        x_pos = self.x[:, :self.dim]
        y = z - x_pos
        
        # Smart Clamping: limit innovation magnitude per stream
        if self.clamp_radius > 0:
            innovation_mag = np.linalg.norm(y, axis=1, keepdims=True) + 1e-9
            scale = np.minimum(1.0, self.clamp_radius / innovation_mag)
            y = y * scale
        
        # Kalman Gain: K = P @ H^T @ (H @ P @ H^T + R)^-1
        # Since H = [I, 0], this simplifies significantly
        for i in range(self.num_streams):
            P_pos = self.P[i, :self.dim, :self.dim]  # Top-left block
            S = P_pos + self.R
            S_inv = np.linalg.inv(S + np.eye(self.dim) * 1e-9)
            
            # K = P @ H^T @ S_inv = P[:, :dim] @ S_inv
            K = self.P[i, :, :self.dim] @ S_inv
            
            # Update state: x = x + K @ y
            self.x[i] += K @ y[i]
            
            # Update covariance: P = (I - K @ H) @ P
            KH = np.zeros((self.state_dim, self.state_dim))
            KH[:, :self.dim] = K
            self.P[i] = (np.eye(self.state_dim) - KH) @ self.P[i]
        
        return self.x[:, :self.dim].copy()
    
    def reset(self):
        """Reset filter state."""
        self.x.fill(0)
        self.P = np.tile(np.eye(self.state_dim) * 0.1, (self.num_streams, 1, 1))
        self._initialized = False


@dataclass
class NoiseStats:
    """
    Tracks noise events detected by the rate limiter for file quality evaluation.
    
    Tracks both event counts AND severity (magnitude of violations) for accurate
    quality scores even when comparing files with different noise characteristics.
    """
    # Cumulative counts
    total_frames: int = 0
    spike_detections: int = 0        # PASS 1: Teleport/impossible movement detections
    rate_limit_clamps: int = 0       # PASS 3: Rate limit exceeded events
    jitter_damping_events: int = 0   # PASS 4: Jitter damping applied
    innovation_clamps: int = 0       # PASS 5: KF innovation exceeded threshold
    
    # Cumulative SEVERITY (magnitude of violations in N·m)
    spike_severity: float = 0.0          # Sum of excess torque above threshold
    rate_limit_severity: float = 0.0     # Sum of clamped magnitudes
    innovation_severity: float = 0.0     # Sum of clamped innovation magnitudes
    
    # Per-joint cumulative (for identifying problematic joints)
    joint_spike_counts: np.ndarray = None
    joint_spike_severity: np.ndarray = None
    joint_rate_limit_counts: np.ndarray = None
    joint_jitter_counts: np.ndarray = None
    
    def __post_init__(self):
        # Initialize per-joint arrays (24 joints)
        if self.joint_spike_counts is None:
            self.joint_spike_counts = np.zeros(24, dtype=np.int32)
        if self.joint_spike_severity is None:
            self.joint_spike_severity = np.zeros(24, dtype=np.float64)
        if self.joint_rate_limit_counts is None:
            self.joint_rate_limit_counts = np.zeros(24, dtype=np.int32)
        if self.joint_jitter_counts is None:
            self.joint_jitter_counts = np.zeros(24, dtype=np.int32)
    
    def reset(self):
        """Reset all statistics."""
        self.total_frames = 0
        self.spike_detections = 0
        self.rate_limit_clamps = 0
        self.jitter_damping_events = 0
        self.innovation_clamps = 0
        self.spike_severity = 0.0
        self.rate_limit_severity = 0.0
        self.innovation_severity = 0.0
        self.joint_spike_counts.fill(0)
        self.joint_spike_severity.fill(0.0)
        self.joint_rate_limit_counts.fill(0)
        self.joint_jitter_counts.fill(0)
    
    def get_noise_score(self):
        """
        Compute a normalized noise score (0-100).
        Higher = noisier file.
        
        Uses SEVERITY (magnitude) rather than just counts to better differentiate
        between files with mild vs severe noise.
        """
        if self.total_frames == 0:
            return 0.0
        
        # Severity-based scoring (N·m per frame)
        # Spike severity has highest weight - teleports are worst
        # Rate limit and innovation severity are secondary
        severity_per_frame = (
            self.spike_severity * 1.0 +           # Full weight for spikes
            self.rate_limit_severity * 0.2 +      # Lower weight for rate limits
            self.innovation_severity * 0.1 +      # Lowest for innovation
            self.jitter_damping_events * 0.1      # Jitter still count-based
        ) / self.total_frames
        
        # Normalize to 0-100 scale
        # 10 N·m/frame severity = score of 50
        # 20+ N·m/frame severity = score of 100
        score = min(100.0, severity_per_frame * 5.0)
        return score
    
    def get_report(self):
        """Return a summary dict of noise statistics."""
        return {
            'total_frames': self.total_frames,
            'noise_score': self.get_noise_score(),
            # Counts
            'spike_detections': self.spike_detections,
            'rate_limit_clamps': self.rate_limit_clamps,
            'jitter_damping_events': self.jitter_damping_events,
            'innovation_clamps': self.innovation_clamps,
            # Severity (total N·m)
            'spike_severity': self.spike_severity,
            'rate_limit_severity': self.rate_limit_severity,
            'innovation_severity': self.innovation_severity,
            # Per-frame metrics
            'spikes_per_frame': self.spike_detections / max(1, self.total_frames),
            'severity_per_frame': (self.spike_severity + self.rate_limit_severity) / max(1, self.total_frames),
            'noisiest_joints': self._get_noisiest_joints()
        }
    
    def _get_noisiest_joints(self, top_n=5):
        """Return indices of the noisiest joints (by severity)."""
        # Use severity, not counts, for ranking
        total_per_joint = self.joint_spike_severity + self.joint_rate_limit_counts * 0.5
        sorted_indices = np.argsort(total_per_joint)[::-1]
        return sorted_indices[:top_n].tolist()

@dataclass
class SMPLProcessingOptions:
    # --- Input / Coordinates ---
    input_type: str = 'axis_angle'
    input_up_axis: str = 'Y'
    quat_format: str = 'xyzw' # 'xyzw' (scipy) or 'wxyz'
    axis_permutation: str = None # "x,y,z"
    return_quats: bool = False
    
    # --- Physics / Dynamics ---
    dt: float = 1.0/60.0
    add_gravity: bool = False
    enable_passive_limits: bool = True # Enabled for Structural Support
    enable_apparent_gravity: bool = True
    
    # --- Filtering / Signal Processing ---
    enable_one_euro_filter: bool = True
    filter_min_cutoff: float = 1.0
    filter_beta: float = 0.0
    
    # --- Floor / Environment ---
    floor_enable: bool = False
    floor_height: float = 0.0
    floor_tolerance: float = 0.15
    heel_toe_bias: float = 0.0
    enable_impact_mitigation: bool = True
    
    # --- Torque Rate Limiting ---
    enable_rate_limiting: bool = True
    rate_limit_strength: float = 1.0  # Multiplier for per-joint rate limits
    enable_kf_smoothing: bool = True  # SmartClampKF filter for dynamic torque

class SMPLProcessor:
    def __init__(self, framerate, betas=None, gender='neutral', model_path=None, total_mass_kg=75.0):
        """
        Initialize the SMPLProcessor.
        
        Args:
            framerate (float): Motion capture framerate.
            betas (np.ndarray): Shape coefficients (usually size 10 or more).
            gender (str): 'male', 'female', or 'neutral'.
            model_path (str): Optional path to SMPL model file.
            total_mass_kg (float): Approximate total body mass for physics calculations.
        """
        self.framerate = framerate
        self.betas = betas if betas is not None else np.zeros(10)
        self.gender = gender
        self.model_path = model_path
        self.total_mass_kg = total_mass_kg
        self.current_zmp = np.zeros(3) # Initialize to safe default
        
        # Standard SMPL joint names (first 24)
        self.joint_names = [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
        ]
        
        # Passive Limits (Soft Limit in Radians, Stiffness)
        # Angles are "distance from neutral". Neutral = 0.
        # This is a simplification (isotropic limits).
        self.joint_limits = {
            'pelvis': {'limit': np.radians(180), 'k': 0.0}, # Root is free
            'neck': {'limit': np.radians(30), 'k': 20.0}, 
            'spine': {'limit': np.radians(30), 'k': 50.0}, 
            
            # Collar: Asymmetric Hinge limits for structural support
            # Axis 2 (Z-axis) carries the Gravity Load for Elevation (Up/Down).
            # Left (+Z Up): Min Limit at 0.05. Natural (-0.13) < 0.05 -> Supported.
            # Right (-Z Up): Max Limit at -0.05. Natural (-0.003) > -0.05 -> Supported.
            # Tightened limits (+/- 5 deg) to ensure support even near neutral.
            'left_collar': {'type': 'hinge', 'axis': 2, 'min': 0.05, 'max': 0.8, 'k': 2000.0}, 
            'right_collar': {'type': 'hinge', 'axis': 2, 'min': -0.8, 'max': -0.05, 'k': 2000.0},
            
            # Shoulders: Flexible
            'shoulder': {'limit': np.radians(90), 'k': 10.0}, 
            
            # Elbows: Hinge with Extension Stop
            # Left Flexion is Negative (approx), Right is Positive
            # Locked Axes: 0 (Twist/Pronation - assume passive?), 2 (Valgus/Varus - definitely passive)
            # Max=0.0 (Hard Stop), K=500.0 (Linear Stiffness).
            'left_elbow': {'type': 'hinge', 'axis': 1, 'min': -2.8, 'max': 0.0, 'k': 500.0, 'locked_axes': [0, 2]},
            'right_elbow': {'type': 'hinge', 'axis': 1, 'min': -0.0, 'max': 2.8, 'k': 500.0, 'locked_axes': [0, 2]},
            
            # Knees: Similar Hinge (assuming Axis 0 usually?)
            'knee': {'limit': np.radians(160), 'k': 50.0},
            
            'hip': {'limit': np.radians(90), 'k': 30.0},
            'default': {'limit': np.radians(180), 'k': 1.0}
        }
        
        # We are interested in the first 22 joints (indices 0-21)
        self.target_joint_count = 22
        
        self.limb_data = self._compute_limb_properties()
        self.skeleton_offsets = self._compute_skeleton_offsets()
        self.max_torques = self._compute_max_torque_profile()
        self.max_torque_array = self._compute_max_torque_array()

        self.perm_basis = None
        self.perm_basis_rot = None
        self.reset_physics_state()

    def _compute_limb_properties_from_model(self):
        """
        Attempts to load SMPL model via smplx and calculate exact limb lengths from betas.
        Returns:
            lengths (dict): Dictionary of limb lengths if successful, else None.
        """
        if not SMPLX_AVAILABLE:
            return None
            
        # User has smplh models in CWD/smplh/SMPLH_{GENDER}.pkl
        # smplx.create(model_path='.', model_type='smplh') expects exactly this structure.
        model_path = '.' 
        if hasattr(self, 'model_path') and self.model_path:
             model_path = self.model_path
             
        # Map gender. Default to MALE if neutral requested but likely only M/F exist for SMPLH
        gender_map = {'male': 'MALE', 'female': 'FEMALE'}
        g_tag = gender_map.get(self.gender, 'MALE') # Fallback to MALE for neutral
        
        try:
            # Create Model
            # use_pca=False is safer for just shape exploration? No, defaults are fine.
            model = smplx.create(model_path=model_path, 
                                 model_type='smplh',
                                 gender=g_tag, 
                                 num_betas=10, 
                                 ext='pkl')
                                 
            betas_tensor = torch.zeros(1, 10)
            if self.betas is not None:
                b = torch.tensor(self.betas, dtype=torch.float32)
                if b.numel() > 10: b = b[:10]
                betas_tensor[0, :b.numel()] = b
                
            output = model(betas=betas_tensor)
            
            # SMPLH has 52 joints (includes hands). First 24 are body joints.
            # We only care about the first 24 for our skeleton.
            joints = output.joints[0].detach().cpu().numpy() # (52, 3)
            # joints = joints[:24, :] # DO NOT SLICE YET! We need >24 check later.
            
            # Calculate Lengths
            # Indices:
            # 0: Pelvis
            # 1: L_Hip, 2: R_Hip
            # 3: Spine1
            # 4: L_Knee, 5: R_Knee
            # 6: Spine2
            # 7: L_Ankle, 8: R_Ankle
            # 9: Spine3
            # 10: L_Foot, 11: R_Foot
            # 12: Neck
            # 13: L_Collar, 14: R_Collar
            # 15: Head
            # 16: L_Shoulder, 17: R_Shoulder
            # 18: L_Elbow, 19: R_Elbow
            # 20: L_Wrist, 21: R_Wrist
            # 22: L_Hand, 23: R_Hand
            
            def dist(i, j):
                return float(np.linalg.norm(joints[i] - joints[j]))
                
            lengths = {}
            lengths['pelvis_width'] = dist(1, 2)
            lengths['upper_leg'] = (dist(1, 4) + dist(2, 5)) / 2.0
            lengths['lower_leg'] = (dist(4, 7) + dist(5, 8)) / 2.0
            
            # Compute Exact Offsets (24, 3) relative to parents
            # Parent indices (standard SMPL):
            # 0: -1 (Root)
            # 1: 0, 2: 0
            # 3: 0
            # 4: 1, 5: 2
            # 6: 3
            # 7: 4, 8: 5
            # 9: 6
            # 10: 7, 11: 8
            # 12: 9
            # 13: 9, 14: 9
            # 15: 12
            # 16: 13, 17: 14
            # 18: 16, 19: 17
            # 20: 18, 21: 19
            # 22: 20, 23: 21
            # Virtual End Effectors:
            # 24: L_Toe (Child of 10)
            # 25: R_Toe (Child of 11)
            # 26: L_Fingertip (Child of 22/L_Hand)
            # 27: R_Fingertip (Child of 37/R_Hand)
            # 27: R_Fingertip (Child of 23/R_Hand -- Replaced 37)
            # 28: L_Heel (Child of 7/L_Ankle)
            # 29: R_Heel (Child of 8/R_Ankle)
            parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 10, 11, 22, 23, 7, 8]
            self.parents = parents # Persist for RNE Physics
            
            # Note: We need 30 offsets now
            model_offsets = np.zeros((30, 3))
            
            # Use joints directly (World/Rest Pose).
            # Assume Root is at origin or doesn't matter for relative offsets.
            # But Root offset (0) is usually 0,0,0 or position relative to world origin? 
            # In our system, root translation is separate. 
            # Offsets[0] is usually 0. Or hip center rel to pelvis? 
            # Usually Offsets[0] is 0,0,0.
            
            for i in range(1, 24):
                p = parents[i]
                
                # Handle Hand Override for SMPL-H
                if joints.shape[0] > 24:
                    child_idx = i
                    if i == 22: child_idx = 22 # Left Hand Base
                    if i == 23: child_idx = 37 # Right Hand Base (remapped)
                    
                    # Store vector: Child - Parent
                    # Parent is standard index.
                    offset_vec = joints[child_idx] - joints[parents[i]]
                    model_offsets[i] = offset_vec
                else:
                    offset_vec = joints[i] - joints[parents[i]]
                    model_offsets[i] = offset_vec

            # --- Virtual Extensions (Indices 24-29) ---
            # Toes (24, 25): Child of 10, 11
            # User feedback: "Toe tip positions seem to be directly below foot position in rest pose."
            # This implies the previous (0, -1, 0) offset was visually wrong (vertical).
            # The Mesh Foot likely points FORWARD (Local Z) relative to the Ankle joint frame.
            # Updated to Z-forward offset.
            foot_vec = np.array([0.0, 0.0, 1.0]) 
            foot_len = 0.15 # Approx toe length extension from ankle
            model_offsets[24] = foot_vec * foot_len # L_Toe
            model_offsets[25] = foot_vec * foot_len # R_Toe
            
            # Fingertips (26, 27): Child of Hand (22, 37)
            # Extrapolate direction from Wrist->Hand.
            # L_Hand (22) Parent is 20 (Wrist).
            # R_Hand (37) Parent is 21 (Wrist).
            # Vector 20->22 is Wrist->Knuckle.
            # Fingertip should continue this vector.
            # If 20->22 is present in model_offsets[22].
            l_knuckle_vec = model_offsets[22]
            if np.linalg.norm(l_knuckle_vec) > 1e-4:
                l_dir = l_knuckle_vec / np.linalg.norm(l_knuckle_vec)
            else:
                l_dir = np.array([1.0, 0.0, 0.0])
                
            model_offsets[26] = l_dir * 0.08 # 8cm finger
            
            r_knuckle_vec = model_offsets[23]
            if np.linalg.norm(r_knuckle_vec) > 1e-4:
                r_dir = r_knuckle_vec / np.linalg.norm(r_knuckle_vec)
            else:
                r_dir = np.array([-1.0, 0.0, 0.0])
                
            model_offsets[27] = r_dir * 0.08 # 8cm finger
            
            # Heels (28, 29): Child of 7, 8 (Ankle)
            # Offset Down (~3cm) and Back (~7cm).
            # Ankle frame usually has Z forward? 
            # We assume Foot Vector [0,0,1] is Forward.
            # So Back is [0,0,-1].
            heel_vec = np.array([0.0, -0.03, -0.07])
            model_offsets[28] = heel_vec # L_Heel
            model_offsets[29] = heel_vec # R_Heel
                
            # Fix for SMPL-H (52 joints) vs SMPL (24 joints)
            # SMPL: 20->22 (L_Hand), 21->23 (R_Hand)
            # SMPL-H: 22 and 23 are Left Fingers. Right Fingers start around 37.
            if joints.shape[0] > 24:
                # SMPL-H Logic
                l_hand_len = dist(20, 22)
                r_hand_len = dist(21, 37)
                lengths['foot'] = (dist(7, 10) + dist(8, 11)) / 2.0 
            else:
                # Standard SMPL Logic
                l_hand_len = dist(20, 22)
                r_hand_len = dist(21, 23)
                lengths['foot'] = (dist(7, 10) + dist(8, 11)) / 2.0
                
            lengths['hand'] = (l_hand_len + r_hand_len) / 2.0
            
            lengths['spine_segment'] = (dist(0, 3) + dist(3, 6) + dist(6, 9)) / 3.0
            lengths['neck'] = dist(12, 15) # Neck -> Head
            lengths['head'] = 0.20 
            
            lengths['shoulder_width'] = dist(16, 17) 
            lengths['collar'] = (dist(13, 16) + dist(14, 17)) / 2.0
            
            lengths['upper_arm'] = (dist(16, 18) + dist(17, 19)) / 2.0
            lengths['lower_arm'] = (dist(18, 20) + dist(19, 21)) / 2.0
            lengths['hand'] = (dist(20, 22) + dist(21, 23)) / 2.0
            
            # Extend Lengths keys?
            # Or reliance on offsets is enough?
            # User wants "output as index 24 and 25 of both limb length and joint positions"
            # So 'limb_lengths' array must be 28.
            # The node calculates it from offsets.
            
            # Message only once? Or standard print
            print(f"Loaded SMPL-H model ({g_tag}) for accurate anthropometry.")
            return {'lengths': lengths, 'offsets': model_offsets}
            
        except Exception as e:
            # Only print warning once per session ideally, but for now simple print
            print(f"Failed to load SMPL-H model: {e}. Falling back to heuristics.")
            print(f"DEBUG: Ensure 'smplh' folder is in {os.path.abspath(model_path)} and contains SMPLH_{g_tag}.pkl")
            return None

    def _compute_limb_properties(self):
        """
        Approximates limb lengths and masses.
        """
        model_lengths = self._compute_limb_properties_from_model()

        
        # Anthropometric data (de Leva 1996) - approximate segment mass fractions
        # These sums usually add up to ~1.0. 
        # Note: SMPL joints don't map 1:1 to de Leva segments perfectly but we approximate.
        mass_fractions = {
            'pelvis': 0.142, # Pelvis + Lower trunk
            'spine': 0.15,   # Mid + Upper trunk
            'head': 0.069,   # Head + Neck
            'thigh': 0.14,   # Both thighs (0.1 each usually? No, singular is ~0.14? Check de Leva.)
                             # Winter: Thigh is 0.1 per leg. 
                             # Let's use simplified percent per SINGLE limb.
            'upper_leg': 0.10, 
            'lower_leg': 0.0465,
            'foot': 0.0145,
            'upper_arm': 0.028,
            'lower_arm': 0.016,
            'hand': 0.006
        }
        
        model_data = self._compute_limb_properties_from_model()
        
        offsets = None
        if model_data is not None:
             if 'lengths' in model_data:
                 lengths = model_data['lengths']
                 offsets = model_data.get('offsets')
             else:
                 # Legacy fallback if I missed something (unlikely)
                 lengths = model_data
        else:
            # Approximate lengths (in meters) for a standard ~1.7m human.
            # We can try to adjust this based on betas[0] (height correlation) lightly if we want.
            # Beta[0] is roughly height/scaling. +1 sigma is taller.
            # A very rough approximation: scale = 1.0 + (beta[0] * 0.06)
            scale_factor = 1.0
            if self.betas is not None and len(self.betas) > 0:
                scale_factor += self.betas[0] * 0.06
                
            if self.gender == 'female':
                scale_factor *= 0.92
            
            defaults = {
                'pelvis_width': 0.25,
                'upper_leg': 0.45,
                'lower_leg': 0.42,
                'foot': 0.20, # length
                'spine_segment': 0.12, # approx per spine joint
                'neck': 0.10,
                'head': 0.20,
                'shoulder_width': 0.40,
                'collar': 0.15,
                'upper_arm': 0.30,
                'lower_arm': 0.28,
                'hand': 0.18
            }
            lengths = {k: v * scale_factor for k, v in defaults.items()}
            
        # Masses
        # Base Mass (Neutral)
        m_base = self.total_mass_kg
        
        # Adjust for Shape (Betas)
        # Beta[0]: Stature (Height/Size). +1 sigma ~ +8% mass.
        # Beta[1]: Girth (BMI/Fat). +1 sigma ~ +20% mass (Volume scalling).
        scale_mass = 1.0
        if self.betas is not None and len(self.betas) > 1:
            scale_mass += self.betas[0] * 0.08
            scale_mass += self.betas[1] * 0.20
            
        m = m_base * max(0.4, scale_mass) # Safety clip (min 40% of base)
        
        print(f"Computed Total Mass: {m:.2f} kg (Base: {m_base}, Betas: {self.betas[:2] if self.betas is not None else 'None'})")

        masses = {
            'pelvis': m * mass_fractions['pelvis'],
            'spine': m * mass_fractions['spine'] / 3.0, # split across 3 spine joints
            'head': m * mass_fractions['head'],
            'upper_leg': m * mass_fractions['upper_leg'],
            'lower_leg': m * mass_fractions['lower_leg'],
            'foot': m * mass_fractions['foot'],
            'upper_arm': m * mass_fractions['upper_arm'],
            'lower_arm': m * mass_fractions['lower_arm'],
            'hand': m * mass_fractions['hand']
        }

        # Weights for segment mass distribution (approximate)
        self.mass_fractions = mass_fractions # Store for later use
        
        result = {'lengths': lengths, 'masses': masses}
        if offsets is not None:
             result['offsets'] = offsets
        return result

    def _compute_skeleton_offsets(self):
        """
        Compute static local offsets for the skeleton based on limb lengths.
        Returns:
            offsets (np.array): (24, 3) Local position vectors for each joint in parent frame.
        """
        # If we have exact model offsets, use them!
        if isinstance(self.limb_data, dict) and 'offsets' in self.limb_data:
            return self.limb_data['offsets']
        
        # Fallback to heuristic reconstruction
        offsets = np.zeros((28, 3))
        
        for i in range(28):
            if i < 24:
                node_name = self.joint_names[i]
            else:
                extra_names = ['l_toe', 'r_toe', 'l_fingertip', 'r_fingertip']
                node_name = extra_names[i-24]
            
            # Default logic
            length = 0.1
            if 'knee' in node_name: length = self.limb_data['lengths']['upper_leg']
            elif 'ankle' in node_name: length = self.limb_data['lengths']['lower_leg']
            elif 'foot' in node_name: length = 0.05
            elif 'elbow' in node_name: length = self.limb_data['lengths']['upper_arm']
            elif 'wrist' in node_name: length = self.limb_data['lengths']['lower_arm']
            elif 'hand' in node_name: length = self.limb_data['lengths']['hand']
            elif 'spine' in node_name: length = self.limb_data['lengths']['spine_segment']
            elif 'neck' in node_name or 'head' in node_name: length = self.limb_data['lengths']['neck']
            
            # Virtual Extensions
            elif 'toe' in node_name: length = 0.15
            elif 'fingertip' in node_name: length = 0.08

            # Define Base Local Position (unrotated)
            lx, ly, lz = 0.0, -1.0, 0.0 # Down
            
            if 'hip' in node_name:
                w = self.limb_data['lengths']['pelvis_width'] * 0.5
                dir_x = 1.0 if 'left' in node_name else -1.0
                lx = dir_x * w
                ly = -0.05
                lz = 0.0
                length = 1.0 
                
            elif 'spine' in node_name or 'neck' in node_name or 'head' in node_name:
                 lx, ly, lz = 0.0, 1.0, 0.0
                 
            elif 'shoulder' in node_name or 'collar' in node_name: 
                if 'collar' in node_name:
                    # Spine->Collar Offset (Base of Neck to Clavicle Start)
                    length = self.limb_data['lengths']['collar'] * 0.33
                else:
                    # Collar->Shoulder Offset (Clavicle Length)
                    length = self.limb_data['lengths']['collar']
                dir_x = 1.0 if 'left' in node_name else -1.0
                lx = dir_x
                ly = 0.0
                lz = 0.0
                
            elif 'elbow' in node_name or 'wrist' in node_name:
                dir_x = 1.0 if 'left' in node_name else -1.0
                lx, ly, lz = dir_x, 0.0, 0.0
                
            elif 'hand' in node_name:
                length = self.limb_data['lengths']['hand']
                dir_x = 1.0 if 'left' in node_name else -1.0
                lx, ly, lz = dir_x, 0.0, 0.0
                
            elif 'foot' in node_name:
                length = self.limb_data['lengths']['foot']
                # Corrected: Point Forward (+Z) and slightly Down (-Y)
                lx, ly, lz = 0.0, -0.2, 1.0
            
            base_vec = np.array([lx, ly, lz])
            if 'hip' in node_name:
                local_pos = base_vec
            else:
                norm = np.linalg.norm(base_vec)
                if norm > 1e-6:
                    base_vec = base_vec / norm
                local_pos = base_vec * length
            
            offsets[i] = local_pos
            
        return offsets

    def _update_physics_state(self, world_pos, pose_data_aa, options):
        """
        Updates physics state (Velocity, CoM, ZMP) for the current frame.
        Removes legacy contact probability logic.
        """
        F = world_pos.shape[0]
        J = world_pos.shape[1]
        dt = options.dt
        y_dim = getattr(self, 'internal_y_dim', 1)
        
        # --- 1. Velocity Smoothing State ---
        if not hasattr(self, 'prob_smoothed_vel') or self.prob_smoothed_vel is None or self.prob_smoothed_vel.shape[0] != J:
             self.prob_smoothed_vel = np.zeros((J, 3), dtype=np.float32)
             self.prob_prev_world_pos = None

        if not options.floor_enable:
             return
             
        floor_height = options.floor_height

        # --- Use Effective Position (Tips Override) ---
        effective_pos = world_pos.copy()
        if hasattr(self, 'temp_tips') and self.temp_tips:
             for j, t_pos in self.temp_tips.items():
                  if j < J:
                       effective_pos[:, j, :] = t_pos

        # --- Velocity Calculation ---
        if not hasattr(self, 'prob_prev_world_pos') or self.prob_prev_world_pos is None:
             lin_vel = np.zeros_like(effective_pos)
             self.prob_prev_world_pos = effective_pos[0].copy() if effective_pos.ndim == 3 else effective_pos.copy()
        elif F > 1:
             # Batch mode approximation
             lin_vel = np.zeros_like(effective_pos)
             lin_vel[1:] = (effective_pos[1:] - effective_pos[:-1]) / dt
             self.prob_prev_world_pos = effective_pos[-1].copy()
        else:
             lin_vel = (effective_pos - self.prob_prev_world_pos) / dt
             self.prob_prev_world_pos = effective_pos.copy()
             
        # Temporal Smoothing
        if F == 1:
             current_vel = lin_vel[0] # (J, 3)
             alpha_v = 0.8 
             self.prob_smoothed_vel = self.prob_smoothed_vel * (1.0 - alpha_v) + current_vel * alpha_v
        
        # --- 2. CoM & ZMP State ---
        
        # CoM Access
        if self.current_com is not None:
            com = self.current_com # (F, 3) 
            if com.ndim == 2: c = com[0] # Assume F=1/streaming for state mostly
            else: c = com
        else:
            c = self._compute_full_body_com(world_pos[0:1])[0]

        # CoM Dynamics (Acceleration)
        if not hasattr(self, 'prob_prev_com') or self.prob_prev_com is None:
            self.prob_prev_com = c.copy()
            self.prob_prev_com_vel = np.zeros_like(c)
            self.prob_prev_com_acc = np.zeros_like(c)
            self.com_acc_filter = OneEuroFilter(min_cutoff=0.5, beta=0.2, d_cutoff=1.0)
            
            com_vel = np.zeros_like(c)
            com_acc = np.zeros_like(c)
        else:
            # Kinematics
            if F > 1:
                 com_vel = (c - self.prob_prev_com) / dt # Approx
            else:
                 com_vel = (c - self.prob_prev_com) / dt
                 
            # Acceleration
            raw_acc = (com_vel - self.prob_prev_com_vel) / dt
            
            if not hasattr(self, 'com_acc_filter'):
                self.com_acc_filter = OneEuroFilter(min_cutoff=0.5, beta=0.2, d_cutoff=1.0)

            if dt > 0:
                self.com_acc_filter._freq = 1.0 / dt
            
            com_acc = self.com_acc_filter(raw_acc)
            
            # Update State
            self.prob_prev_com = c.copy()
            self.prob_prev_com_vel = com_vel.copy()
            self.prob_prev_com_acc = com_acc.copy()
            
        # ZMP Calculation
        plane_dims = [d for d in [0, 1, 2] if d != y_dim]
        
        # h = CoM Height relative to floor
        com_h = c[y_dim] - floor_height
        g = 9.81
        
        # Offset = - (h/g) * a_horz
        acc_horz = com_acc[plane_dims] 
        zmp_offset = - (com_h / g) * acc_horz
        
        # Static Projection
        p_zmp_static = c[plane_dims]
        p_zmp = p_zmp_static + zmp_offset
        
        # Store ZMP
        if not hasattr(self, 'current_zmp') or self.current_zmp is None:
             self.current_zmp = np.zeros(3) # Streaming assumption
        
        if self.current_zmp.ndim == 1:
            self.current_zmp[plane_dims] = p_zmp
            self.current_zmp[y_dim] = floor_height
        else:
            # Should handle batch if needed, but prob_prev state implies F=1
            self.current_zmp[-1, plane_dims] = p_zmp
            self.current_zmp[-1, y_dim] = floor_height
        
        # ZMP Smoothing
        ZMP_ALPHA = 0.1 
        
        if not hasattr(self, 'prev_zmp_smooth'):
            self.prev_zmp_smooth = self.current_zmp.copy()
        
        if self.prev_zmp_smooth.shape != self.current_zmp.shape:
             self.prev_zmp_smooth = self.current_zmp.copy()
        
        if self.current_zmp.ndim == 1:
            self.current_zmp = self.prev_zmp_smooth * (1.0 - ZMP_ALPHA) + self.current_zmp * ZMP_ALPHA
            self.prev_zmp_smooth = self.current_zmp.copy()
        else:
            # Simple batch smooth?
            pass

    def _compute_probabilistic_contacts_fusion(self, F, J, world_pos, vel_y, vel_h, floor_height, options):
        """
        New Continuous Fusion Logic:
        P_final = P_prox * P_vel * P_acc
        
        Removes hard gates. Weights factors fluidly.
        """
        # Init Result Array (F, J)
        # Note: We compute per-frame, but usually F=1 in real-time mode.
        # DEBUG
        # f is local loop var, here we don't have loop yet.
        # Wait, f is used later.
        # We can just print once.
        
        
        if not options.floor_enable:
             return np.zeros((F, J))
             
        contact_probs = np.zeros((F, J))
        
        # Init State if needed
        if not hasattr(self, 'prob_contact_probs_fusion') or self.prob_contact_probs_fusion is None:
             self.prob_contact_probs_fusion = np.zeros(J)
             
        # Plane dims
        if options.input_up_axis == 'Y':
             y_dim = 1
             plane_dims = [0, 2]
        elif options.input_up_axis == 'Z':
             y_dim = 2
             plane_dims = [0, 1]
        
        # ZMP state (Already computed in main loop)
        p_zmp = self.current_zmp # (J_subset or 3?) - It's (3,) in streaming.
        if p_zmp.ndim > 1: p_zmp = p_zmp[0]
        p_zmp_hz = p_zmp[plane_dims]
        
        # Process Frame 0 only (streaming assumption)
        f = 0 
        
        # Get CoM State
        if hasattr(self, 'prob_prev_com_acc'):
             com_acc = self.prob_prev_com_acc
        else:
             com_acc = np.zeros(3)
             
        # Dynamic Sigma H (Stricter than Legacy)
        # tol/3.0 -> contact extends to ~tol.
        # tol/4.0 -> contact fades faster (at 4.5cm/15cm, P drops to <0.5).
        sigma_h = options.floor_tolerance / 4.0
        if sigma_h < 0.02: sigma_h = 0.02
        
        # 2. Joint Loop
        for j in range(J):
            # Pos
            if hasattr(self, 'temp_tips') and j in self.temp_tips:
                pos = self.temp_tips[j][f]
            else:
                pos = world_pos[f, j]
            
            h = pos[y_dim] - floor_height
            
            # Apply Virtual Heel Offset
            if j in [7, 8, 10, 11, 20, 21]:
                h -= options.heel_toe_bias
            
            vy = vel_y[f, j] if vel_y.ndim > 1 else vel_y[j]
            vh = vel_h[f, j] if vel_h.ndim > 1 else vel_h[j]
            v_total = np.sqrt(vy**2 + vh**2)
            
            # --- 1. Proximity P_prox ---
            if h <= 0:
                p_prox = 1.0
            else:
                p_prox = np.exp(-0.5 * (h / sigma_h)**2)
            if h > 0.20: p_prox = 0.0
            
            # --- 2. Load Factor ---
            pos_hz = pos[plane_dims]
            dist_zmp = np.linalg.norm(pos_hz - p_zmp_hz)
            load_factor = np.exp(- (dist_zmp / 0.45)**2)
            
            # --- 3. Velocity P_vel ---
            # Split into Horizontal (Sliding allowed if loaded) and Vertical (Strict)
            
            # 3a. Horizontal (Load modulates pivot)
            # v_pivot_h shifts from 0.35 to 1.35
            v_pivot_h = 0.35 + 1.0 * load_factor
            exp_h = 10.0 * (vh - v_pivot_h)
            if exp_h > 20: exp_h = 20
            p_vel_h = 1.0 / (1.0 + np.exp(exp_h))
            
            # 3b. Vertical Velocity Penalty Removed
            # Reason: Downward velocity (plunging) is good (loading). 
            # Upward velocity (lifting) is handled by P_LIFT below.
            # Legacy P_VEL_Y penalized both, causing early contact loss during heel lift (dip).
            p_vel = p_vel_h
            
            # --- 4. Acceleration P_acc ---
            # Freefall penalty.
            acc_y = com_acc[y_dim]
            # Soft penalty starting at -5.0, max at -10.0
            if acc_y < -5.0 and v_total < 0.5 and load_factor < 0.2:
                 penalty = ((-5.0 - acc_y) / 5.0)
                 if penalty > 1.0: penalty = 1.0
                 if penalty < 0.0: penalty = 0.0
                 p_acc = 1.0 - penalty
            else:
                 p_acc = 1.0
            
            # --- Lift-Off Check (Asymmetric Trend & Softened) ---
            # Smoothing: Fast Decay (Landing), Slow Rise (Lift).
            # Fixes latency on contact onset.
            
            # Init trend state if needed (per joint)
            if not hasattr(self, 'prob_vel_trend') or self.prob_vel_trend is None or self.prob_vel_trend.shape[0] != J:
                 self.prob_vel_trend = np.zeros(J)
            
            vy = vel_y[f, j]
            
            if F == 1:
                 # Asymmetric Alpha
                 # If decelerating (vy < trend, typically landing/dropping), update fast (0.5).
                 # If accelerating (vy > trend, lifting), update slow (0.1) to ignore noise.
                 alpha_v = 0.5 if vy < self.prob_vel_trend[j] else 0.1
                 self.prob_vel_trend[j] = self.prob_vel_trend[j] * (1.0 - alpha_v) + vy * alpha_v
                 v_trend = self.prob_vel_trend[j]
            else:
                 v_trend = vy 
            
            # Check Trend
            # Threshold 0.02 m/s.
            if v_trend > 0.02: 
                 # Penalty: Softened steepness 50 -> 30
                 p_lift = np.exp(-30.0 * (v_trend - 0.02))
                 if p_lift > 1.0: p_lift = 1.0
            else:
                 p_lift = 1.0
            
            # --- Horizontal Slide Check (Asymmetric Trend & Softened) ---
            # Init trend state if needed
            if not hasattr(self, 'prob_vel_trend_h') or self.prob_vel_trend_h is None or self.prob_vel_trend_h.shape[0] != J:
                 self.prob_vel_trend_h = np.zeros(J)
            
            vh_inst = np.linalg.norm(vel_h[f, j])
            
            if F == 1:
                 # Asymmetric Alpha
                 # If slowing down (vh < trend), update fast (0.5).
                 # If speeding up (vh > trend), update slow (0.1).
                 alpha_h = 0.5 if vh_inst < self.prob_vel_trend_h[j] else 0.1
                 self.prob_vel_trend_h[j] = self.prob_vel_trend_h[j] * (1.0 - alpha_h) + vh_inst * alpha_h
                 vh_trend = self.prob_vel_trend_h[j]
            else:
                 vh_trend = vh_inst
            
            # Check Slide Trend
            # Threshold 0.3 m/s.
            if vh_trend > 0.3:
                 # Penalty: Softened steepness 20 -> 10
                 p_slide = np.exp(-10.0 * (vh_trend - 0.3))
                 if p_slide > 1.0: p_slide = 1.0
            else:
                 p_slide = 1.0
            
            # --- Fusion ---
            p_raw = p_prox * p_vel * p_acc * p_lift * p_slide
            

            

            

            
            # --- 5. Geometric "Toe-Off" Check (Updated for SMPL 24 convention) ---
            # Check Right Ankle (8) vs Right Foot (11)
            # User Feedback: "Penalize Ankle (8) if it is higher than Foot (11)"
            if j == 8:
                 if 11 in self.temp_tips: f_pos = self.temp_tips[11][f]
                 else: f_pos = world_pos[f, 11]
                 h_foot = f_pos[y_dim] - floor_height
                 
                 # Ankle is naturally above Foot (~4-8cm).
                 # If Ankle is significantly higher (e.g. > 15cm above foot), it implies heel lift/plantar flexion.
                 # Or if Foot is near ground and Ankle is high?
                 
                 # Let's say if Ankle is > Foot + 0.12 (12cm)
                 if h > (h_foot + 0.12):
                      p_raw *= 0.0

            # Check Left Ankle (7) vs Left Foot (10)
            if j == 7:
                 if 10 in self.temp_tips: f_pos = self.temp_tips[10][f]
                 else: f_pos = world_pos[f, 10]
                 h_foot = f_pos[y_dim] - floor_height
                 
                 if h > (h_foot + 0.12):
                      p_raw *= 0.0
            
            # --- Smoothing ---
            # Standard EMA 0.5
            self.prob_contact_probs_fusion[j] = self.prob_contact_probs_fusion[j] * 0.5 + p_raw * 0.5
            
            contact_probs[f, j] = self.prob_contact_probs_fusion[j]
            
        return contact_probs

    def _compute_max_torque_array(self):
        """
        Convert dictionary profile to per-joint max torque array.
        Returns:
            arr (np.array): (24, 3) Max torque vector for each joint.
        """
        arr = np.zeros((24, 3))
        
        # Biometric Estimates (Approximate)
        # Coordinate Systems (SMPL T-Pose):
        # Legs (Hip/Knee/Ankle): Bone along Y. X=Flex/Ext, Y=Twist, Z=Abd/Add.
        # Arms (Shldr/Elbow/Wrist): Bone along X. X=Twist, Y=Flex/Ext, Z=Abd/Add.
        # Spine: Bone along Y. X=Flex/Ext, Y=Twist, Z=LatBend.
        
        # Default fallback (Isotropic)
        default_t = 100.0
        
        for i in range(24):
            name = self.joint_names[i]
            
            # Start with isotropic or specific vector
            val = default_t
            
            if 'pelvis' in name: val = [500.0, 200.0, 500.0] # Core is strong in all axes
            elif 'hip' in name: val = [300.0, 50.0, 150.0] # Flex, Twist, Abd
            elif 'knee' in name: val = [250.0, 20.0, 20.0] # Hinge (Primary X)
            elif 'ankle' in name: val = [150.0, 20.0, 40.0] # Dorsi/Plantar strong
            elif 'foot' in name: val = [40.0, 10.0, 10.0]
            
            elif 'spine' in name: val = [400.0, 100.0, 300.0] # Flex, Twist, Bend
            elif 'neck' in name: val = [50.0, 20.0, 40.0]
            elif 'head' in name: val = [30.0, 10.0, 20.0]
            
            elif 'collar' in name: val = [100.0, 100.0, 500.0] # Structural Z support
            elif 'shoulder' in name: val = [60.0, 120.0, 100.0] # Twist, Flex, Abd
            elif 'elbow' in name: val = [20.0, 100.0, 20.0] # Hinge (Primary Y)
            elif 'wrist' in name: val = [10.0, 30.0, 20.0]
            elif 'hand' in name: val = [10.0, 10.0, 10.0]
            
            # Feature: User Overrides via set_max_torque (stored in dict)
            # If user set a value in self.max_torques, use it.
            # But keys in dict are generic 'knee', 'hip'. 
            # We must look up by specific joint name or generic key.
            # Efficiency: Reverse lookup?
            # Or iterate dict... slow.
            # Since self.max_torques was populated with defaults before,
            # we should clear the defaults from that dict and only store overrides?
            # Or just check if the dict has a value for this specific key?
            
            # Actually, the previous implementation checked generic keys in order.
            # Let's preserve that logic for Overrides, but use these defaults if missing.
            
            # Check for user overrides in self.max_torques
            # self.max_torques currently contains... whatever we init'd it with.
            # We should probably initialize self.max_torques to these new defaults in __init__ instead?
            # Or just use the defaults here if dict lookup fails.
            
            # Let's map these defaults to the dictionary in __init__ properly, 
            # so user editing the dict works as expected.
            # But this function *reads* the dict.
            
            # Let's just use the priority logic to pull from dict, assuming dict is populated.
            # Update: I will update _compute_max_torque_profile (the dict init) instead!
            # It's cleaner.
            
            arr[i] = val 
            
        return arr

    def _compute_max_torque_profile(self):
        """
        Returns a dictionary of biometric max torque vectors (N-m).
        """
        # Coordinate Systems (SMPL T-Pose):
        # Legs (Hip/Knee/Ankle): Bone Y. X=Flex/Ext, Y=Twist, Z=Abd/Add.
        # Arms (Shldr/Elbow/Wrist): Bone X. X=Twist, Y=Flex/Ext, Z=Abd/Add.
        # Spine: Bone Y. X=Flex/Ext, Y=Twist, Z=LatBend.

        max_t = {
            'pelvis': [500.0, 300.0, 500.0],
            'spine': [400.0, 100.0, 300.0],
            'hip': [300.0, 150.0, 200.0], 
            'knee': [250.0, 100.0, 100.0], # Hinge
            'ankle': [200.0, 60.0, 80.0],
            'foot': [80.0, 20.0, 20.0],
            'neck': [50.0, 20.0, 40.0],
            'head': [30.0, 10.0, 20.0],
            'collar': [100.0, 100.0, 500.0], # Z support
            'shoulder': [60.0, 120.0, 100.0],
            'elbow': [20.0, 100.0, 20.0], # Hinge
            'wrist': [10.0, 30.0, 20.0],
            'hand': [10.0, 10.0, 10.0]
        }
        
        scale = 1.0
        if self.gender == 'female':
            scale = 0.7
            
        return {k: np.array(v) * scale for k, v in max_t.items()}

    # def _compute_max_torque_array(self):
    #     """
    #     Convert dictionary profile to per-joint max torque array.
    #     Returns:
    #         arr (np.array): (24, 3) Max torque vector for each joint.
    #     """
    #     arr = np.zeros((24, 3))
    #     for i in range(24):
    #         name = self.joint_names[i]
    #         max_t = 100.0 # Default scalar
    #
    #         # Priority Logic (matches original process_frame)
    #         if 'pelvis' in name: max_t = self.max_torques.get('pelvis', 500.0)
    #         elif 'hip' in name: max_t = self.max_torques.get('hip', 300.0)
    #         elif 'knee' in name: max_t = self.max_torques.get('knee', 250.0)
    #         elif 'ankle' in name: max_t = self.max_torques.get('ankle', 40.0)
    #         elif 'foot' in name: max_t = self.max_torques.get('foot', 30.0)
    #         elif 'spine' in name: max_t = self.max_torques.get('spine', 400.0)
    #         elif 'neck' in name: max_t = self.max_torques.get('neck', 50.0)
    #         elif 'head' in name: max_t = self.max_torques.get('head', 15.0)
    #         elif 'collar' in name: max_t = self.max_torques.get('collar', 500.0)
    #         elif 'shoulder' in name: max_t = self.max_torques.get('shoulder', 120.0)
    #         elif 'elbow' in name: max_t = self.max_torques.get('elbow', 80.0)
    #         elif 'wrist' in name: max_t = self.max_torques.get('wrist', 20.0)
    #         elif 'hand' in name: max_t = self.max_torques.get('hand', 10.0)
    #
    #         # Handle Vector or Scalar
    #         if np.ndim(max_t) == 0:
    #              arr[i, :] = max_t
    #         else:
    #              # Check length
    #              v = np.array(max_t)
    #              if v.shape == (3,):
    #                   arr[i, :] = v
    #              else:
    #                   arr[i, :] = v[0] # Fallback?
    #
    #     return arr

    def set_max_torque(self, joint_name_filter, value):
        """
        Manually updates max torque for joints matching the filter.
        Args:
            joint_name_filter (str): Substring to match (e.g., 'neck').
            value (float or list[3]): New max torque value/vector.
        """
        count = 0
        # Check value format
        # If passed as string, list, or array from DPG
        processed_val = value
        
        # Try to clean input
        if hasattr(value, '__len__') and not isinstance(value, str):
             if len(value) == 3:
                  processed_val = np.array(value, dtype=float)
             elif len(value) == 1:
                  processed_val = float(value[0])
        else:
             processed_val = float(value)
             
        # 1. Update Profile Dict
        for k in self.max_torques:
            if joint_name_filter in k:
                self.max_torques[k] = processed_val
                count += 1
        
        # 2. Update Cached Array
        if hasattr(self, 'max_torque_array'):
            self.max_torque_array = self._compute_max_torque_array()
            
        return count

    def set_full_max_torque_profile(self, profile_array):
        """
        Set valid max torques for all joints at once.
        Args:
            profile_array: (24, 3) or (24,) array.
        """
        arr = np.array(profile_array)
        
        if arr.shape == (24, 3):
             self.max_torque_array = arr
        elif arr.shape == (24,) or arr.shape == (24, 1):
             # Broadcast to 3D
            if arr.shape == (24, 1): arr = arr[:, 0]
            self.max_torque_array = np.zeros((24, 3))
            self.max_torque_array[:, 0] = arr
            self.max_torque_array[:, 1] = arr
            self.max_torque_array[:, 2] = arr
        else:
             print(f"SMPLProcessor: Invalid max torque profile shape {arr.shape}")

    def _compute_torque_rate_limits(self):
        """
        Compute per-joint torque rate limits based on biomechanical capabilities.
        
        Returns:
            rate_limits (np.array): (24,) Max torque change per frame at 60 FPS (N·m/frame).
            
        Note:
            These values represent the maximum physically plausible rate of torque change.
            Larger joints (hips, spine) can sustain higher absolute torques but change more slowly.
            Smaller joints (wrists, fingers) have lower absolute limits but faster dynamics.
        """
        # Defaults indexed by joint group
        # Values are N·m per frame at 60 FPS reference rate
        defaults = {
            'pelvis': 20.0,
            'hip': 30.0,
            'spine': 15.0,
            'knee': 25.0,
            'ankle': 15.0,
            'foot': 10.0,
            'neck': 10.0,
            'head': 8.0,
            'collar': 12.0,
            'shoulder': 20.0,
            'elbow': 12.0,
            'wrist': 8.0,
            'hand': 5.0
        }
        
        rate_limits = np.zeros(self.target_joint_count)
        
        for i in range(self.target_joint_count):
            name = self.joint_names[i]
            
            # Find matching key by substring
            matched = False
            for key, val in defaults.items():
                if key in name:
                    rate_limits[i] = val
                    matched = True
                    break
            
            if not matched:
                rate_limits[i] = 15.0  # Fallback default
        
        return rate_limits

    def reset_physics_state(self):
        """Reset internal state for frame-by-frame physics calculation."""
        self.prev_pose_aa = None
        self.prev_pose_raw = None # For input spike detection
        self.prev_vel_aa_raw = None # For accel spike detection
        self.prev_acc_raw = None # For sign-flip detection
        self.last_frame_spiked = False # To prevent stuck rejection on fast moves
        self.prev_vel_aa = None
        self.prev_acc_aa = None
        
        # Dual Path (Effort) State
        self.prev_pose_q_effort = None
        self.prev_vel_aa_effort = None
        
        self.prev_world_pos = None # For Linear Velocity calculation
        self.prev_tips = None # For Tip Velocity calculation
        # Vectorized OneEuroFilter for 72 channels (24 joints * 3)
        self.pose_filter = None 
        self.prev_efforts = None # For spike rejection 
        
        # CoM Dynamics
        self.prev_com_pos = None
        self.prev_com_vel = None
        
        # Filter for CoM acceleration (to derive ZMP)
        # Reduced min_cutoff to 4.0 (from 10.0) to reduce jitter while maintaining responsiveness
        self.com_accel_filter = OneEuroFilter(framerate=self.framerate, min_cutoff=4.0, beta=0.05) 
        
        # Output Smoothing State for Contact Weights (Left, Right)
        # Stores the previous w_foot value (0.0 to 1.0)
        self.prev_w_foot = np.array([0.5, 0.5]) # Initialized to mid-foot 
        
        # Pitch History for Velocity Calculation
        self.prev_foot_pitch_arr = {10: 0.0, 11: 0.0} 
        
        # Probabilistic Contact State
        self.prob_prev_world_pos = None
        self.prob_contact_probs = None
        self.prob_smoothed_vel = None
        
        # Torque Rate Limiter State
        self.prev_dynamic_torque = None  # (24, 3) previous frame's dynamic torque vectors
        self.torque_rate_limits = self._compute_torque_rate_limits()  # (24,) per-joint limits 
        
        # Jitter Detection State
        # Track sign of torque components over a short window to detect oscillation
        JITTER_WINDOW = 8  # frames to track
        self.torque_sign_history = None  # (window, joints, 3) circular buffer of signs
        self.jitter_history_idx = 0  # current write position in circular buffer
        
        # SmartClampKF Filter for Dynamic Torque Smoothing
        self.dynamic_torque_kf = None  # Initialized on first use
        
        # Noise Statistics Tracking
        self.noise_stats = NoiseStats()

    def reset_noise_stats(self):
        """Reset noise statistics for a new file evaluation."""
        if hasattr(self, 'noise_stats') and self.noise_stats is not None:
            self.noise_stats.reset()
        else:
            self.noise_stats = NoiseStats()
    
    def get_noise_report(self):
        """Get cumulative noise statistics report."""
        if hasattr(self, 'noise_stats') and self.noise_stats is not None:
            return self.noise_stats.get_report()
        return None
    
    def get_noise_score(self):
        """Get current noise score (0-100)."""
        if hasattr(self, 'noise_stats') and self.noise_stats is not None:
            return self.noise_stats.get_noise_score()
        return 0.0


    def _get_hierarchy(self):
        """Defines parent-child relationships for SMPL 24 joints + Virtual."""
        if hasattr(self, 'parents'):
             return self.parents
             
        # Fallback (Should not happen if __init__ run)
        parents = [-1,  0,  0,  0,  1,  2, 
                    3,  4,  5,  6,  7,  8, 
                    9,  9,  9, 12, 13, 14, 
                   16, 17, 18, 19, 20, 21,
                   10, 11, 22, 23, 7, 8] # Included 28, 29 (Heels). 27->23.
        return parents

    def _compute_subtree_inertia(self, joint_idx, world_positions, world_orientations, limb_lengths, limb_masses):
        """
        Computes the effective moment of inertia of the subtree rooted at joint_idx,
        relative to the joint_idx position.
        
        Approximation:
        - Segments are thin rods.
        - I_local = 1/12 * m * L^2
        - Parallel axis theorem: I_effective = I_local + m * d^2
        """
        parents = self._get_hierarchy()
        children = [i for i, p in enumerate(parents) if p == joint_idx]
        
        # Traverse subtree
        subtree_indices = []
        stack = [joint_idx]
        while stack:
            curr = stack.pop()
            subtree_indices.append(curr)
            # Find children of curr
            curr_children = [i for i, p in enumerate(parents) if p == curr]
            stack.extend(curr_children)
            
        # We only care about the kinematic chain downstream. 
        # But we need to identify which 'limb segment' is attached to which joint.
        # Convention: The bone connecting Parent->Child is associated with Child's mass/length?
        # Or Parent? In SMPL, "Left Knee" joint rotates the "Left Lower Leg".
        # So Joint i controls Segment i (where Segment i is the bone extending from i to its children).
        
        # However, leaf nodes (hands, feet) have mass but no further joints in standard 24 set.
        # Actually usually:
        # Pelvis (0) -> connected to hips and spine. The "pelvis" mass is at 0.
        # Hip (1) -> connects to knee. "Thigh" mass is here.
        
        total_inertia = np.zeros(world_positions.shape[0])
        
        joint_pos = world_positions[..., joint_idx, :]
        
        for idx in subtree_indices:
            # Skip if joint index is outside our target 22 joints (though hands are 22/23)
            # We should include all 24 for mass purposes if possible.
            if idx >= 24: continue
            
            name = self.joint_names[idx]
            
            # Map joint name to limb data keys
            # simplified mapping
            m = 0.0
            l = 0.0
            
            if 'pelvis' in name: m, l = limb_masses['pelvis'], 0.1 # COM close to root
            elif 'hip' in name: m, l = limb_masses['upper_leg'], limb_lengths['upper_leg']
            elif 'knee' in name: m, l = limb_masses['lower_leg'], limb_lengths['lower_leg']
            elif 'ankle' in name: m, l = limb_masses['foot'], limb_lengths['foot']
            elif 'spine' in name: m, l = limb_masses['spine'], limb_lengths['spine_segment']
            elif 'neck' in name: m, l = limb_masses.get('head', 1.0)*0.2, limb_lengths['neck'] # partial head
            elif 'head' in name: m, l = limb_masses['head']*0.8, limb_lengths['head']
            elif 'collar' in name: m, l = limb_masses['upper_arm']*0.2, 0.1 # clavicle approx
            elif 'shoulder' in name: m, l = limb_masses['upper_arm']*0.8, limb_lengths['upper_arm']
            elif 'elbow' in name: m, l = limb_masses['lower_arm'], limb_lengths['lower_arm']
            elif 'wrist' in name: m, l = limb_masses['hand'], limb_lengths['hand']
            elif 'hand' in name: continue # leaf, handled by wrist mass usually, or just small tip
            elif 'foot' in name: m, l = limb_masses['foot'] * 0.4, 0.08 # Toes/Distal foot
            
            if m <= 0: continue

            # Find child node to define rod direction
            child_nodes = [c for c, p in enumerate(parents) if p == idx]
            
            com_pos = world_positions[..., idx, :] # Default fallback
            
            if len(child_nodes) > 0:
                # Take average of children as end point
                end_pos = np.mean([world_positions[..., c, :] for c in child_nodes], axis=0)
                com_pos = (world_positions[..., idx, :] + end_pos) * 0.5
            else:
                # Leaf node with mass (e.g. Toes/Foot tip)
                # We need to estimate a direction for the segment.
                # Heuristic: Use the parent-to-joint vector direction?
                # Or use a default forward vector (Z)?
                # Using parent-to-joint direction:
                p_idx = parents[idx]
                if p_idx != -1:
                    dir_vec = joint_pos - world_positions[..., p_idx, :]
                    norm = np.linalg.norm(dir_vec, axis=-1, keepdims=True)
                    norm_safe = norm.copy()
                    norm_safe[norm_safe < 1e-6] = 1.0 # Prevent division by zero
                    dir_vec_normalized = dir_vec / norm_safe
                    
                    # If norm was small, use default direction
                    is_small = (norm < 1e-6).flatten()
                    if np.any(is_small):
                         dir_vec_normalized[is_small] = np.array([0.0, 0.0, 1.0])
                    
                    dir_vec = dir_vec_normalized
                    
                    # Assume COM is at l/2 along this direction
                    com_pos = joint_pos + dir_vec * (l * 0.5)

            # Distance from the PIVOT (joint_idx) to this segment's COM
            r_vec = com_pos - joint_pos
            # r_sq = np.dot(r_vec, r_vec) # Scalar version
            r_sq = np.sum(r_vec**2, axis=-1) # Vectorized version (F,) or scalar
            
            # Parallel Axis Theorem
            # I_eff = I_local + m * r^2
            i_local = (1.0/12.0) * m * (l**2)
            
            total_inertia += (i_local + m * r_sq)
            
        return total_inertia
    
    def _compute_passive_torque(self, joint_name, active_torque_vec, pose_aa):
        """
        Calculates passive ligament torque if joint exceeds soft limits.
        Supports:
        - Symmetric 'Cone' limits (default): Limit on total angle magnitude.
        - Asymmetric 'Hinge' limits: Min/Max on specific axis.
        
        Args:
            joint_name (str): Name to look up limits.
            active_torque_vec (np.array): (F, 3) The active/net torque (used to determine opposition direction).
            pose_aa (np.array): (F, 3) The local pose axis-angle vector.
            
        Returns:
            passive_torque (np.array): (F, 3) Vector opposing the limit violation.
        """
        # Find limits
        limits = self.joint_limits.get('default')
        for key in self.joint_limits:
            if key in joint_name:
                limits = self.joint_limits[key]
                break
        
        limit_type = limits.get('type', 'cone')
        k = limits['k']
        
        if limit_type == 'hinge':
             # Hinge / Single Axis Limit
             axis_idx = limits.get('axis', 0)
             min_val = limits.get('min', -np.pi)
             max_val = limits.get('max', np.pi)
             
             # --- 1. Hinge Axis Logic ---
             angle = pose_aa[:, axis_idx] # (F,)
             torque_val = np.zeros_like(angle)
             
             
             # Check Min Violation (Angle < Min)
             mask_min = angle < min_val
             if np.any(mask_min):
                 deviation = min_val - angle[mask_min]
                 # Linear Stiffness for Hard Stops
                 torque_val[mask_min] = k * deviation
                 
             # Check Max Violation (Angle > Max)
             mask_max = angle > max_val
             if np.any(mask_max):
                 deviation = angle[mask_max] - max_val
                 # Linear Stiffness for Hard Stops
                 torque_val[mask_max] = -k * deviation
                 
             # --- 2. Locked Axes Logic (Structural Support) ---
             # If axes are explicitly locked, they absorb ALL Net Torque (Active = 0).
             # "Structure takes the load".
             locked_axes = limits.get('locked_axes', [])
             
             passive_torque = np.zeros_like(pose_aa)
             
             # Primary Axis (Spring Limit)
             passive_torque[:, axis_idx] = torque_val
             
             
             # Locked Axes (Force Match)
             if locked_axes:
                 for ax in locked_axes:
                     # Passive = Net.  => Active = Net - Passive = 0.
                     passive_torque[:, ax] = active_torque_vec[:, ax]
            
             
             # Fallback for non-locked, non-primary axes?
             # They assume 0 passive torque (Full Active).
             
             return passive_torque
             
        else:
            # Default: Symmetric Cone
            limit_angle = limits['limit']
            
            # Calculate deviation from neutral
            # Magnitude of rotation
            angle = np.linalg.norm(pose_aa, axis=1) # (F,)
            
            # Check excess
            excess = np.maximum(0, angle - limit_angle) # (F,)
            
            # If no excess, return 0
            if np.all(excess == 0):
                return np.zeros_like(pose_aa)
                
            # Passive Torque Magnitude (Exponential Spring)
            # T = k * excess^2 (Non-linear stiffening)
            t_mag = k * (excess ** 2)
            
            # Direction: Opposes the displacement (Restoring force)
            # Axis of rotation is pose_aa / angle
            # Restoring torque acts NEGATIVE to this axis.
            
            # Normalize axis safely
            axis = np.zeros_like(pose_aa)
            mask = angle > 1e-6
            axis[mask] = pose_aa[mask] / angle[mask][:, np.newaxis]
            
            # Passive Torque Vector
            # Opposes displacement -> -axis * magnitude
            passive_torque = -axis * t_mag[:, np.newaxis]
            
            return passive_torque

    def _compute_subtree_com(self, joint_idx, world_positions, limb_lengths, limb_masses):
        """
        Computes the Center of Mass (COM) and Total Mass of the subtree at joint_idx.
        Returns: (total_mass, com_vec_world)
        """
        parents = self._get_hierarchy()
        
        # Traverse subtree (BFS/DFS)
        subtree_indices = []
        stack = [joint_idx]
        while stack:
            curr = stack.pop()
            subtree_indices.append(curr)
            curr_children = [i for i, p in enumerate(parents) if p == curr]
            stack.extend(curr_children)
            
        total_mass = 0.0
        F = world_positions.shape[0]
        weighted_pos_sum = np.zeros((F, 3))
        
        for idx in subtree_indices:
            if idx >= 24: continue
            
            name = self.joint_names[idx]
            
            # Map joint name to limb data keys (simplified)
            m = 0.0
            l = 0.0
            # ... (Limb mapping omitted for brevity, should use same as above or factor out)
            if 'pelvis' in name: m, l = limb_masses['pelvis'], 0.1 
            elif 'hip' in name: m, l = limb_masses['upper_leg'], limb_lengths['upper_leg']
            elif 'knee' in name: m, l = limb_masses['lower_leg'], limb_lengths['lower_leg']
            elif 'ankle' in name: m, l = limb_masses['foot'], limb_lengths['foot']
            elif 'spine' in name: m, l = limb_masses['spine'], limb_lengths['spine_segment']
            elif 'neck' in name: m, l = limb_masses.get('head', 1.0)*0.2, limb_lengths['neck']
            elif 'head' in name: m, l = limb_masses['head']*0.8, limb_lengths['head']
            elif 'collar' in name: m, l = limb_masses['upper_arm']*0.2, 0.1
            elif 'shoulder' in name: m, l = limb_masses['upper_arm']*0.8, limb_lengths['upper_arm']
            elif 'elbow' in name: m, l = limb_masses['lower_arm'], limb_lengths['lower_arm']
            elif 'wrist' in name: m, l = limb_masses['hand'], limb_lengths['hand']
            elif 'hand' in name: continue 
            
            if m <= 0: continue
            
            # Segment COM estimation
            child_nodes = [c for c, p in enumerate(parents) if p == idx]
            com_pos = world_positions[..., idx, :]
            if len(child_nodes) > 0:
                end_pos = np.mean([world_positions[..., c, :] for c in child_nodes], axis=0)
                com_pos = (world_positions[..., idx, :] + end_pos) * 0.5
            
            total_mass += m
            weighted_pos_sum += m * com_pos
            
        if total_mass > 0:
            final_com = weighted_pos_sum / total_mass
        else:
            final_com = world_positions[..., joint_idx, :]
            
        return total_mass, final_com

    def _compute_full_body_com(self, world_positions):
        """Computes Total Body Center of Mass (F, 3)."""
        parents = self._get_hierarchy()
        limb_masses = self.limb_data['masses']
        limb_lengths = self.limb_data['lengths']
        
        total_mass = 0.0
        F = world_positions.shape[0]
        weighted_pos_sum = np.zeros((F, 3))
        
        for idx in range(24):
            name = self.joint_names[idx]
            
            # Map joint name to limb data keys
            m = 0.0
            if 'pelvis' in name: m = limb_masses['pelvis']
            elif 'hip' in name: m = limb_masses['upper_leg']
            elif 'knee' in name: m = limb_masses['lower_leg']
            elif 'ankle' in name: m = limb_masses['foot']
            elif 'spine' in name: m = limb_masses['spine']
            elif 'neck' in name: m = limb_masses.get('head', 1.0)*0.2
            elif 'head' in name: m = limb_masses['head']*0.8
            elif 'collar' in name: m = limb_masses['upper_arm']*0.2
            elif 'shoulder' in name: m = limb_masses['upper_arm']*0.8
            elif 'elbow' in name: m = limb_masses['lower_arm']
            elif 'wrist' in name: m = limb_masses['hand']
            # Hands/Others ignored or lumped
            
            if m <= 0: continue
            
            # Segment COM estimation
            child_nodes = [c for c, p in enumerate(parents) if p == idx]
            com_pos = world_positions[..., idx, :]
            if len(child_nodes) > 0:
                end_pos = np.mean([world_positions[..., c, :] for c in child_nodes], axis=0) # (F, 3)
                com_pos = (world_positions[..., idx, :] + end_pos) * 0.5
            
            total_mass += m
            weighted_pos_sum += m * com_pos
            
        if total_mass > 0:
            return weighted_pos_sum / total_mass
        else:
            return world_positions[..., 0, :] # Fallback to Pelvis
    def _compute_gravity_torques(self, world_pos, options, global_rots=None, tips=None):
        """
        Compute gravity torque vectors for all joints.
        
        Args:
            world_pos (np.array): (F, 24, 3) World positions of joints.
            options (SMPLProcessingOptions): Usage: add_gravity, input_up_axis.
            global_rots (list[R] or F,24,4): Optional rotations for accurate leaf CoM.
            tips (dict): Optional virtual tip positions.
            
        Returns:
            t_grav_vecs (np.array): (F, 24, 3) Gravity torque vectors.
        """
        F = world_pos.shape[0]
        t_grav_vecs = np.zeros((F, self.target_joint_count, 3))
        
        if not options.add_gravity and not options.enable_apparent_gravity:
            return t_grav_vecs
            
        J_full = world_pos.shape[1]
        
        # DEBUG MASS ACCUMULATION
        # verify mass logic
        pass
            
        # Determine Gravity Vector
        g_mag = 9.81
        
        # Gravity should align with the INTERNAL simulation frame (usually Y-Up).
        # We respect 'internal_y_dim' if set, otherwise assume Y (1).
        y_dim = getattr(self, 'internal_y_dim', 1)
        
        g_vec = np.zeros(3)
        g_vec[y_dim] = -g_mag
            
        parents = self._get_hierarchy()
        limb_masses = self.limb_data['masses']
        limb_lengths = self.limb_data['lengths']
        
        # 1. Compute Individual Segment Properties (Mass & CoM)
        # We store weighted position sum (M*r) and Total Mass per node
        # Initialize with SEGMENT values
        node_masses = np.zeros((F, J_full))
        node_weighted_com = np.zeros((F, J_full, 3))
        
        # Mapping Logic (Simplified from _compute_subtree_com)
        # We can pre-calculate the scalar mass for each joint index once?
        # Yes, mass is constant across frames.
        
        joint_segment_masses = np.zeros(24)
        # Identify constant segment masses
        for idx in range(24):
            name = self.joint_names[idx]
            m = 0.0
            if 'pelvis' in name: m = limb_masses['pelvis']
            elif 'hip' in name: m = limb_masses['upper_leg']
            elif 'knee' in name: m = limb_masses['lower_leg']
            elif 'ankle' in name: m = limb_masses['foot']
            elif 'spine' in name: m = limb_masses['spine']
            elif 'neck' in name: m = limb_masses.get('head', 1.0)*0.2
            elif 'head' in name: m = limb_masses['head']*0.8
            elif 'collar' in name: m = limb_masses['upper_arm']*0.2
            elif 'shoulder' in name: m = limb_masses['upper_arm']*0.8
            elif 'elbow' in name: m = limb_masses['lower_arm']
            elif 'wrist' in name: m = limb_masses['hand']
            joint_segment_masses[idx] = m
        
        # Iterate to compute Segment CoMs
        for idx in range(24):
            m = joint_segment_masses[idx]
            if m <= 0: continue
            
            # Segment CoM: Midpoint between Joint and Mean(Children)
            
            child_nodes = [c for c, p in enumerate(parents) if p == idx]
            
            joint_pos = world_pos[:, idx, :]
            if len(child_nodes) > 0:
                # Vectorized Mean of Children
                child_pos_sum = np.zeros((F, 3))
                for c in child_nodes:
                    child_pos_sum += world_pos[:, c, :]
                
                end_pos = child_pos_sum / len(child_nodes)
                seg_com = (joint_pos + end_pos) * 0.5
            else:
                # Leaf Logic (Head, Hands, Feet)
                # To create a non-zero lever arm for gravity, we must estimate where the
                # mass is relative to the joint pivot.
                
                seg_com = joint_pos.copy() # Default
                
                # Priority 1: Virtual Tip (Best for Hands/Feet)
                if tips is not None and idx in tips:
                    # tip is (F, 3)
                    tip_pos = tips[idx] # Might need broadcasting if F mismatch?
                    if tip_pos.shape[0] == F:
                         seg_com = (joint_pos + tip_pos) * 0.5
                    
                # Priority 2: Joint Rotation (Best for Head / Leaves without tips)
                elif global_rots is not None:
                      # global_rots is List[R] or (F, 24, 4)?
                      # _compute_forward_kinematics returns 'quats' (F,24,4) usually if we check signature?
                      # But docstring says List[R]. 
                      # Let's handle list of R objects.
                      
                      # Identify extension length
                      name = self.joint_names[idx]
                      ext_len = 0.0
                      if 'head' in name: ext_len = limb_lengths.get('head', 0.20)
                      elif 'hand' in name: ext_len = limb_lengths.get('hand', 0.18)
                      elif 'foot' in name: ext_len = limb_lengths.get('foot', 0.20)
                      
                      if ext_len > 0:
                          # Project along LOCAL Y-axis (Standard Bone Axis)
                          local_axis = np.array([0.0, 1.0, 0.0]) # Shape (3,)
                          
                          # Rotate to World
                          # If global_rots is list of R
                          if isinstance(global_rots, list):
                               r_obj = global_rots[idx] # (F,) rotation object
                               axis_world = r_obj.apply(local_axis) # (F, 3)
                               seg_com = joint_pos + axis_world * (ext_len * 0.5)
                          elif hasattr(global_rots, 'shape') and global_rots.ndim == 3 and global_rots.shape[-1] == 4:
                               # Quats (T, J, 4) -> (F, 4)
                               # Just in case caller passed raw quats
                               r_obj = R.from_quat(global_rots[:, idx, :])
                               axis_world = r_obj.apply(local_axis)
                               seg_com = joint_pos + axis_world * (ext_len * 0.5)

                # Priority 3: Fallback to Parent Vector (Better than nothing)
                else: 
                     parent_idx = parents[idx]
                     if parent_idx >= 0:
                         parent_pos = world_pos[:, parent_idx, :]
                         bone_vec = joint_pos - parent_pos
                         norm = np.linalg.norm(bone_vec, axis=-1, keepdims=True)
                         
                         valid_norm = norm > 1e-6
                         direction = np.zeros_like(bone_vec)
                         np.divide(bone_vec, norm, out=direction, where=valid_norm)
                         
                         name = self.joint_names[idx]
                         ext_len = 0.0
                         if 'head' in name: ext_len = limb_lengths.get('head', 0.20)
                         elif 'hand' in name: ext_len = limb_lengths.get('hand', 0.18)
                         elif 'foot' in name: ext_len = limb_lengths.get('foot', 0.20)
                         
                         seg_com = joint_pos + direction * (ext_len * 0.5)
                
            node_masses[:, idx] = m
            node_weighted_com[:, idx, :] = seg_com * m
            
        # Store for debug/CoM calc
        self.node_masses = node_masses
        self.node_weighted_com = node_weighted_com
            
        # 2. Bottom-Up Accumulation (Subtree Properties)
        # Iterate reverse (Leaves -> Root)
        # Exclude indices >= 24 if they exist? target_joint_count is 22 usually, but mapped to 24 skeleton.
        
        subtree_masses = node_masses.copy()
        subtree_weighted_com = node_weighted_com.copy()
        
        for idx in range(23, 0, -1): # Skip 0 (Root has no parent in array logic for accumulation)
            parent = parents[idx]
            if parent >= 0:
                subtree_masses[:, parent] += subtree_masses[:, idx]
                subtree_weighted_com[:, parent, :] += subtree_weighted_com[:, idx, :]
        
        # 3. Compute Torque
        # T = r_com_rel x F_gravity
        # r_com_rel = (Weighted_Sum / Total_Mass) - Joint_Pos
        
        # Avoid division by zero
        mask = subtree_masses > 0
        
        # Initialize Com Vectors (F, 24, 3)
        subtree_com = np.zeros_like(world_pos)
        subtree_com[mask] = subtree_weighted_com[mask] / subtree_masses[mask][:, np.newaxis]
        
        # For joints with mass=0, torque is 0 (already init to 0)
        
        # Calculate R vector
        r_vecs = subtree_com - world_pos
        
        # Calculate Force (M * g)
        # Broadcaset g_vec (3,) to (F, 24, 3)
        f_grav = subtree_masses[:, :, np.newaxis] * g_vec[np.newaxis, np.newaxis, :]
        
        # Cross Product
        t_grav_vecs = np.cross(r_vecs, f_grav)
        
        # Old Debug Block Removed
        
        # --- Apparent Gravity (Airborne Physics) ---
        # If in freefall, body accelerates with gravity (a ~ -g). Effective weight is 0.
        # If landing, body decelerates (a > 0). Effective weight is > mg.
        # FIX: Use ROOT Position (System Frame) instead of CoM (which includes internal Dynamics).
        # Using CoM causes internal limb swings to trigger global gravity scaling.
        
        # --- Apparent Gravity (Airborne Physics & Locomotion Effort) ---
        # 1. Inertial Force Logic (if enabled)
        
        current_root = world_pos[:, 0, :] # (F, 3)
        g_acc_vec = np.zeros_like(current_root)
        
        if options.enable_apparent_gravity:
            # Calculate Root Acceleration (System Acceleration)
            dt = options.dt
            # current_root already defined
            
            if not hasattr(self, 'prev_root_gravity'):
                 self.prev_root_gravity = current_root.copy()
                 self.prev_vel_gravity = np.zeros_like(current_root)
                 root_acc = np.zeros_like(current_root)
            else:
                 # Handle Teleport (Resets)
                 dist = np.linalg.norm(current_root - self.prev_root_gravity, axis=-1)
                 if np.any(dist > 2.0): 
                      self.prev_root_gravity = current_root.copy()
                      self.prev_vel_gravity = np.zeros_like(current_root)
                      root_acc = np.zeros_like(current_root)
                 else:
                      current_vel = (current_root - self.prev_root_gravity) / dt
                      root_acc = (current_vel - self.prev_vel_gravity) / dt
                      
                      self.prev_root_gravity = current_root.copy()
                      self.prev_vel_gravity = current_vel
                      
            # Construct Effective Gravity Vector: g_eff = g_base - a_root
            # D'Alembert's Principle: Inertial Force = -ma. Equivalent to gravity field -a.
            
            # Clamp acceleration to prevent noise scaling (e.g. +/- 30 m/s^2)
            root_acc = np.clip(root_acc, -30.0, 30.0)
            
            # Rate-of-change limit: max change in acceleration per frame
            # This prevents sudden acceleration spikes from mocap noise
            if hasattr(self, 'prev_root_acc_smooth'):
                dt_acc = 1.0 / self.framerate
                max_acc_change = 15.0 * dt_acc * 120  # 15 m/s^2/s at 120fps baseline
                acc_delta = root_acc - self.prev_root_acc_smooth
                acc_delta_clamped = np.clip(acc_delta, -max_acc_change, max_acc_change)
                root_acc = self.prev_root_acc_smooth + acc_delta_clamped
            
            # Smooth it (Stronger EMA) to remove mocap jitter
            if not hasattr(self, 'prev_root_acc_smooth'):
                 self.prev_root_acc_smooth = np.zeros_like(root_acc)
                 
            alpha_acc = 0.03  # Stronger low-pass filter (3% blend, ~33 frame settling)
            root_acc = self.prev_root_acc_smooth * (1.0 - alpha_acc) + root_acc * alpha_acc
            self.prev_root_acc_smooth = root_acc
            
            g_acc_vec = -root_acc
        else:
             # Reset history if disabled (to avoid stale start if re-enabled?)
             # self.prev_root_gravity = None # Might cause error
             pass
        
        # Combine with Base Gravity (g_vec)
        # g_vec is (3,)
        # g_acc_vec is (F, 3)
        g_vec_eff = g_vec[np.newaxis, :] + g_acc_vec # (F, 3)
        
        # Store for debug (magnitude scaling relative to 9.81)
        g_mag_eff = np.linalg.norm(g_vec_eff, axis=-1)
        self.g_factor = (g_mag_eff[0] / 9.81) if len(g_mag_eff) > 0 else 1.0
        
        # --- Update Gravity Force Vectors ---
        # f_grav was computed using static g_vec at Line 1555.
        # We must RECOMPUTE it using g_vec_eff.
        
        # f_grav = m * g_eff
        # subtree_masses: (F, 24)
        # g_vec_eff: (F, 3) -> (F, 1, 3)
        f_grav = subtree_masses[:, :, np.newaxis] * g_vec_eff[:, np.newaxis, :]
        
        # Recompute Torque (r x f_new)
        # r_vecs: (F, 24, 3)
        t_grav_vecs = np.cross(r_vecs, f_grav)
        
        # We NO LONGER scale by scalar g_factor. The vector logic handles it.
        # Removed: t_grav_vecs *= g_factor_reshaped
        
        # Note: 'f_grav' used for RNE subtraction later is derived from 'g_vec'.
        # We rely on 'contact_pressure' being scaled by GRF factor (process_frame logic) 
        # to scale the GRF term. 
        # So both T_grav and T_grf scale together. 
        
        # Optimization: Cache Full Body CoM (Root Subtree) for Floor Contact logic
        # subtree_com[:, 0, :] is the CoM of the whole tree (rooted at 0)
        # Note: If F=1, simple assignment. If F>1, whole array.
        self.current_com = subtree_com[:, 0, :] # (F, 3)
        self.current_total_mass = subtree_masses[:, 0] # (F,)
        
        # --- 4. Standing Correction (Ground Reaction Gravity) ---
        # Standard gravity calc treats Hips as "Roots" of the Legs (calculating lift torque).
        # When Standing, Hips support the Upper Body (calculating load torque).
        # We blend between these models based on Foot Contact Probability.
        
        # --- 4. Generalized Ground Reaction Forces (RNE) ---
        # Instead of heuristic "Support Logic", we calculate the actual Moment of Ground Reaction Forces.
        # Torque_net = Torque_gravity - Torque_GRF
        
        if hasattr(self, 'contact_pressure') and self.contact_pressure is not None:
             # 1. Compute GRF Vectors (With Friction/Alignment)
             # Instead of Pure Vertical GRF (High Torque in wide stance), we align GRF with Support Leg Axis.
             
             contact_masses = self.contact_pressure # (F, J)
             if contact_masses.ndim == 1: contact_masses = contact_masses[np.newaxis, :]
             
             f_grf_vecs = np.zeros_like(world_pos)
             
             # Leg Mapping: Toe/Ankle -> Hip
             # Arm Mapping: Wrist/Hand -> Shoulder (Fix Cartwheel Effort)
             leg_map = {
                 10: 1, 7: 1, 11: 2, 8: 2,           # Legs -> Hips
                 20: 16, 22: 16, 21: 17, 23: 17,     # Arms -> Shoulders
                 24: 1, 25: 2,                       # Toes -> Hips (if present)
                 26: 16, 27: 17,                     # Fingers -> Shoulders (if present)
                 28: 1, 29: 2                        # Heels -> Hips (Virtual, Index 28/29)
             }
             
             # Base Vertical Force (Magnitude)
             # contact_masses is Kg. Force = kg * g_eff.
             # Use g_vec_eff magnitude for scaling
             # Base Vertical Force (Magnitude)
             # contact_masses is Kg. Force = kg * g_eff.
             # Use g_vec_eff magnitude for scaling
             g_mag = np.linalg.norm(g_vec_eff, axis=-1, keepdims=True) + 1e-9 # (F, 1)
             f_vert_mag = contact_masses * g_mag # (F, J)
             
             # Loop over joints (Vectorized over F)
             for j in range(world_pos.shape[1]):
                 if np.any(contact_masses[:, j] > 0):
                      target_hip = leg_map.get(j, -1)
                      
                      # Alignment Logic (Only for Ankles to stabilize main leg strut)
                      # For Toes/Heels, we want Vertical Force to preserve Lever Arm Torque (Balance)
                      use_alignment = (target_hip != -1) and (options.floor_enable) and (j in [7, 8])
                      
                      if use_alignment:
                           v_leg = world_pos[:, target_hip, :] - world_pos[:, j, :] # (F, 3)
                           dy = v_leg[:, 1]
                           
                           # Valid Alignment Check (Leg not horizontal)
                           valid = dy > 0.1
                           
                           # Aligned Force: Scale so Y-component matches f_vert_mag
                           scale = f_vert_mag[:, j] / (dy + 1e-9)
                           f_aligned = v_leg * scale[:, np.newaxis]
                           
                           # Vertical Fallback (Up)
                           # g_vec_eff is Down. -g is Up.
                           # Normalized Direction
                           dir_up = -g_vec_eff / g_mag # (F, 3)
                           f_vertical = f_vert_mag[:, j, np.newaxis] * dir_up
                           
                           # Blend
                           mask_valid = valid
                           f_grf_vecs[mask_valid, j, :] = f_aligned[mask_valid]
                           f_grf_vecs[~mask_valid, j, :] = f_vertical[~mask_valid]
                           
                      else:
                           # Default Vertical
                           dir_up = -g_vec_eff / g_mag
                           f_grf_vecs[:, j, :] = f_vert_mag[:, j, np.newaxis] * dir_up

             # 2. Backward Pass (Accumulate GRF Moments)
             f_cum_grf = f_grf_vecs.copy()
             m_cum_grf = np.zeros_like(f_grf_vecs)
             
             parents = np.array(self._get_hierarchy())
             J = world_pos.shape[1]
             
             for j in range(J-1, 0, -1):
                 parent = parents[j]
                 if parent >= 0:
                      # Propagate Force
                      f_child = f_cum_grf[:, j, :]
                      f_cum_grf[:, parent, :] += f_child
                      
                      # Propagate Moment
                      # M_parent += M_child + (P_child - P_parent) x F_child
                      m_child = m_cum_grf[:, j, :]
                      r_bone = world_pos[:, j, :] - world_pos[:, parent, :]
                      t_lever = np.cross(r_bone, f_child)
                      
                      m_cum_grf[:, parent, :] += m_child + t_lever
                      
             # Combine with Gravity (Opposing)
             # Gravity Moment (t_grav) and GRF Moment (m_cum) should naturally oppose.
             # e.g. Gravity pulls CoM down (Adduction), GRF pushes Foot up (Abduction).
             # Signs should be opposite, so Addition is correct.
             t_grav_vecs += m_cum_grf
             
             # --- Structural Support Damping Removed ---
             # We rely on Friction Alignment (above) to stabilize stance.
             
             # --- Global CoP Stability Model ---
             # REVERTED: We now rely on correct ZMP-based Force Distribution to handle balance.
             # Pure Physics: Net Torque = Gravity Torque + Support Torque.
             # If balanced, these cancel naturally.
        
        return t_grav_vecs

    def _compute_floor_contacts_RENAMED(self, world_pos, pose_data_aa, options):
        """
        Simplified Floor Contact Logic (State-based).
        
        Rules:
        1. Entry: Height < Tolerance AND Velocity_Down approx 0 (Stopped).
        2. Exit: Height > Exit_Tolerance OR Velocity_Horiz > Slip_Limit.
        3. Distribution: ZMP-based Gaussian weighting.
        4. Geometry: Heel/Toe priority band.
        """
        F = world_pos.shape[0]
        if not options.floor_enable:
             return np.zeros((F, 24), dtype=np.float32)
             
        # Use tracked internal dimension (defaults to 1, but 2 if unrotated Z-up)
        y_dim = getattr(self, 'internal_y_dim', 1)
        dt = options.dt
        floor_height = options.floor_height
        tol_enter = options.floor_tolerance
        tol_exit = tol_enter * 1.5 # Hysteresis
        
        # --- 1. Compute Velocities ---
        if self.prev_world_pos is None:
             self.prev_world_pos = world_pos[0].copy() if world_pos.ndim == 3 else world_pos.copy()
             
        lin_vel = np.zeros_like(world_pos)
        if F > 1:
             lin_vel[0] = (world_pos[0] - self.prev_world_pos) / dt
             lin_vel[1:] = (world_pos[1:] - world_pos[:-1]) / dt
             self.prev_world_pos = world_pos[-1].copy()
             pass
        else:
             lin_vel = (world_pos - self.prev_world_pos) / dt
             self.prev_world_pos = world_pos.copy()
             
        vel_y = lin_vel[..., y_dim] # (F, 24) or (24,)
        
        vel_h_vec = lin_vel.copy()
        vel_h_vec[..., y_dim] = 0.0
        vel_h = np.linalg.norm(vel_h_vec, axis=-1)
        
        # --- 2. Determine Contact Candidates (State Machine) ---
        # Initialize with history if available
        if self.prev_contact_mask is None:
            self.prev_contact_mask = np.zeros(24, dtype=bool)
        
        # We need a boolean mask for logic
        current_contact_state = np.zeros((F, 24), dtype=bool)
        
        # Iterate (Vectorized per frame usually, but here we loop f for clarity/state)
        # Actually, for batch F>1, state propagation is tricky. 
        # But usually F=1 in streaming.
        
        curr_state = self.prev_contact_mask.astype(bool) # (24,)
        
        # Tip Positions (Heels/Toes/Hands)
        # Need to use 'temp_tips' which are computed in forward kinematics
        # But we need them aligned with world_pos frame index.
        # process_frame stores self.temp_tips (dict of joint -> pos array)
        
        # Ensure temp_tips exists
        tips_available = hasattr(self, 'temp_tips') and self.temp_tips is not None
        
        for f in range(F):
            # Per-Joint Checking
            for j in range(24):
                # Use Tip Position if available
                pos = world_pos[f, j, :]
                if tips_available and j in self.temp_tips:
                    pos = self.temp_tips[j][f]
                    
                h = pos[y_dim] - floor_height
                vy = vel_y[f, j]
                vh = vel_h[f, j]
                
                was_contact = curr_state[j]
                
                if was_contact:
                    # EXIT Conditions
                    # 1. Lifted
                    if h > tol_exit:
                        curr_state[j] = False
                    # 2. Sliding (Slip)
                    elif vh > 0.5: # 0.5 m/s slip limit
                        curr_state[j] = False
                    # 3. Active Lift (Jump/Rebound)
                    elif vy > 0.1: # Rising fast (> 10cm/s) -> Break Contact immediately
                        curr_state[j] = False
                    else:
                        # Sustain
                        curr_state[j] = True
                else:
                    # ENTRY Conditions
                    # 1. In Zone
                    if h < tol_enter:
                        # 2. Descent Stopped (Soft Landing)
                        # vy should be close to 0 or positive (rising CHECK)
                        # allow slight negative drift (-0.05) due to noise
                        # AND Prevent Re-Entry if Rising Fast (Oscillation Fix)
                        # Hysteresis: Entry < 0.05, Exit > 0.1.
                        if vy > -0.05 and vy < 0.05: 
                            curr_state[j] = True
                            
            # --- 3. Heel/Toe Geometric Resolution ---
            # If both Heel and Toe are candidates, apply geometric priority.
            # Left: Heel(7), Toe(10). Right: Heel(8), Toe(11).
            for (heel, toe) in [(7, 10), (8, 11)]:
                # Get Heights
                if tips_available and heel in self.temp_tips and toe in self.temp_tips:
                    h_heel = self.temp_tips[heel][f, y_dim]
                    h_toe = self.temp_tips[toe][f, y_dim]
                else:
                    # Fallback to joint positions
                    h_heel = world_pos[f, heel, y_dim]
                    h_toe = world_pos[f, toe, y_dim]

                # Diff: Positive = Heel Higher
                diff = h_heel - h_toe
                band = 0.05 # 5cm blending band
                
                r_heel = 0.5
                r_toe = 0.5
                
                if diff > band: # Heel is > 5cm above Toe -> Toe takes all
                    r_heel = 0.0
                    r_toe = 1.0
                elif diff < -band: # Toe is > 5cm above Heel -> Heel takes all
                    r_heel = 1.0
                    r_toe = 0.0
                else:
                    # Blend [-band, +band] -> [0, 2*band]
                    # t goes 0 (Heel Only) to 1 (Toe Only)
                    t = (diff + band) / (2.0 * band)
                    r_toe = t
                    r_heel = 1.0 - t
                    
                # Apply strictly? 
                # If a contact is OFF, it stays OFF.
                # If both are ON, we modulate them? 
                # Actually, the user wants "Heel gets no contact" if lifted.
                # So we act on the boolean state? 
                # Better: We keep boolean state as "Physical Candidate", 
                # and use these ratios for Mass Distribution Weighting.
                
                # Store ratios for this frame/pair to use in Weighting step
                # We can hack this by storing in a temporary dict
                pass 
                
            current_contact_state[f] = curr_state.copy()
            
        # Update History
        self.prev_contact_mask = curr_state.astype(float) # Store as float mask for compatibility
        
        # --- 4. Mass Distribution (ZMP) ---
        contact_masses = np.zeros((F, 24), dtype=np.float32)
        total_mass = self.total_mass_kg
        
        # Compute Ref Point (ZMP)
        if self.current_com is not None:
             com = self.current_com # (F, 3)
        else:
             com = self._compute_full_body_com(world_pos)
             
        # Compute COM Accel (Horizontal)
        # Use existing filter state
        com_vel = np.zeros_like(com)
        if F > 1:
            com_vel[1:] = (com[1:] - com[:-1]) / dt
            if self.prev_com_pos is not None: com_vel[0] = (com[0] - self.prev_com_pos) / dt
        elif self.prev_com_pos is not None:
            com_vel = (com - self.prev_com_pos) / dt
            
        accel_h = np.zeros_like(com) # Simplified: Ignore acceleration for stability first?
        # User requested: "Use center of mass and momentum"
        # ZMP = CoM - (h/g)*accel
        g = 9.81
        h_com = np.maximum(com[..., y_dim] - floor_height, 0.1)
        
        # Calculate Accel for ZMP
        if self.prev_com_vel is not None:
             accel = (com_vel - self.prev_com_vel) / dt
             accel_h = accel.copy()
             accel_h[..., y_dim] = 0.0
             
        zmp_offset = - (h_com[:, np.newaxis] / g) * accel_h
        target_point = com + zmp_offset
        
        # Explicitly set total_mass for distribution
        total_mass = self.total_mass_kg
        
        # DEBUG MASS
        # print(f"DEBUG: Inside _compute_floor_contacts. total_mass_kg: {self.total_mass_kg}")
        raise RuntimeError(f"CRASH TEST: Total Mass is {self.total_mass_kg}")
        
        # Update COM State
        if F > 0:
             self.prev_com_pos = com[-1].copy()
             self.prev_com_vel = com_vel[-1].copy()

        # Distribute
        for f in range(F):
            # Active Indices
            active_indices = np.where(current_contact_state[f])[0]
            if len(active_indices) == 0:
                continue
                
            weights = []
            
            # Recalculate Heel/Toe Ratios for this frame
            # (Repeating logic for clarity inside loop)
            geo_weights = np.ones(24)
            for (heel, toe) in [(7, 10), (8, 11)]:
                # Get Heights
                if tips_available and heel in self.temp_tips and toe in self.temp_tips:
                    h_heel = self.temp_tips[heel][f, y_dim]
                    h_toe = self.temp_tips[toe][f, y_dim]
                else:
                    h_heel, h_toe = world_pos[f, heel, y_dim], world_pos[f, toe, y_dim]
                
                diff = h_heel - h_toe
                band = 0.05
                if diff > band: 
                    geo_weights[heel] = 0.0
                    geo_weights[toe] = 1.0
                elif diff < -band:
                    geo_weights[heel] = 1.0
                    geo_weights[toe] = 0.0
                else:
                    t = (diff + band) / (2.0 * band)
                    geo_weights[toe] = t
                    geo_weights[heel] = 1.0 - t
            
            # Distance Weights
            sigma = 0.15 # Sharp (was 0.5)
            
            for idx in active_indices:
                # Position
                pos = world_pos[f, idx, :]
                if tips_available and idx in self.temp_tips:
                    pos = self.temp_tips[idx][f]
                    
                # Dist to ZMP
                dist_vec = pos - target_point[f]
                dist_vec[y_dim] = 0.0
                d_sq = np.sum(dist_vec**2)
                
                w_dist = np.exp(-d_sq / (2 * sigma**2))
                
                # Combine with Geometric Weight
                w_final = w_dist * geo_weights[idx]
                weights.append(w_final)
            
            total_w = sum(weights)
            if total_w > 1e-4:
                for i, idx in enumerate(active_indices):
                    contact_masses[f, idx] = total_mass * (weights[i] / total_w)
            
            # --- Refine Foot/Ankle Balance (Post-Pass: ZMP Projection) ---
            for (idx_a, idx_f) in [(7, 10), (8, 11)]:
                m_a = contact_masses[f, idx_a]
                m_f = contact_masses[f, idx_f]
                total_pair = m_a + m_f
                
                if total_pair < 0.1:
                    continue

                # 1. Geometric Validity (Height Logic)
                # Re-evaluate validity based on the 5cm band
                h_heel = 0.0
                h_toe = 0.0
                if tips_available and idx_f in self.temp_tips and idx_a in self.temp_tips:
                     h_heel = self.temp_tips[idx_a][f, y_dim]
                     h_toe = self.temp_tips[idx_f][f, y_dim]
                else:
                     h_heel = world_pos[f, idx_a, y_dim]
                     h_toe = world_pos[f, idx_f, y_dim]
                     
                diff = h_heel - h_toe
                band = 0.05
                
                # Default: Both valid
                valid_heel = 1.0
                valid_toe = 1.0
                
                if diff > band: # Heel High
                    valid_heel = 0.0
                    valid_toe = 1.0
                elif diff < -band: # Toe High
                    valid_heel = 1.0
                    valid_toe = 0.0
                else: # Mixing Band
                    t_geo = (diff + band) / (2.0 * band)
                    valid_toe = t_geo
                    valid_heel = 1.0 - t_geo
                    
                # 2. ZMP Demand (Project ZMP onto Foot Axis)
                # Use 'target_point' which includes CoM + Momentum (ZMP)
                zmp = target_point[f]
                
                plane_dims = [d for d in [0, 1, 2] if d != y_dim]
                p_heel = world_pos[f, idx_a, plane_dims]
                p_toe = world_pos[f, idx_f, plane_dims]
                p_zmp = zmp[plane_dims]
                
                if tips_available and idx_f in self.temp_tips and idx_a in self.temp_tips:
                     p_heel = self.temp_tips[idx_a][f][plane_dims]
                     p_toe = self.temp_tips[idx_f][f][plane_dims]

                vec_ht = p_toe - p_heel
                vec_hz = p_zmp - p_heel
                
                dist_sq = np.sum(vec_ht**2)
                demand_toe = 0.5
                demand_heel = 0.5
                
                if dist_sq > 1e-6:
                    # Projection
                    t_zmp = np.dot(vec_hz, vec_ht) / dist_sq
                    # Clamp [0, 1] means strictly between heel/toe.
                    t_zmp = np.clip(t_zmp, 0.0, 1.0)
                    
                    demand_toe = t_zmp
                    demand_heel = 1.0 - t_zmp
                
                # 3. Combine
                # Start with 'demand' (Physics preference), gated by 'validity' (Geometry)
                
                share_toe = demand_toe * valid_toe
                share_heel = demand_heel * valid_heel
                
                total_share = share_toe + share_heel
                
                if total_share > 1e-4:
                    normalized_toe = share_toe / total_share
                    normalized_heel = share_heel / total_share
                    
                    contact_masses[f, idx_f] = total_pair * normalized_toe
                    contact_masses[f, idx_a] = total_pair * normalized_heel
                    
        return contact_masses
    def set_axis_permutation(self, axis_permutation):
        """
        Parse and cache the axis permutation matrix.
        Args:
            axis_permutation (str): Format "x,y,z" or "y,z,-x" etc.
        """
        if not axis_permutation:
            self.perm_basis = None
            self.perm_basis_rot = None
            return

        try:
            parts = [p.strip().lower() for p in axis_permutation.split(',')]
            if len(parts) != 3:
                print(f"Error: Invalid axis_permutation '{axis_permutation}'. Expected 3 parts.")
                self.perm_basis = None
                self.perm_basis_rot = None
                return

            basis = np.zeros((3, 3))
            
            for i, p in enumerate(parts): # i is target index (0=X, 1=Y, 2=Z)
                sign = -1.0 if '-' in p else 1.0
                axis_char = p.replace('-', '').strip()
                src_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis_char, -1)
                
                if src_idx != -1:
                    basis[i, src_idx] = sign
            
            self.perm_basis = basis
            
            # Check Determinant for Rotation Matrix
            det = np.linalg.det(basis)
            basis_for_rot = basis
            
            if abs(det - 1.0) > 0.01:
                # msg = f"Warning: Axis Permutation '{axis_permutation}' matrix has det={det:.2f}. "
                if det < 0:
                    # Reflection (Left-Handed). Flip Z col for rotation.
                    basis_for_rot = basis.copy()
                    basis_for_rot[:, 2] *= -1.0
            
            self.perm_basis_rot = basis_for_rot
            
        except Exception as e:
             print(f"Error parsing axis permutation: {e}")
             self.perm_basis = None
             self.perm_basis_rot = None

    def _prepare_trans_and_pose(self, pose_data, trans_data, options):
        """
        Handles input reshaping, coordinate system conversion, and initial type conversion.
        Returns:
            trans_data (F, 3)
            pose_data_aa (F, 24, 3)
            quats (F, 24, 4)
        """
        # Unpack Options
        input_type = options.input_type
        input_up_axis = options.input_up_axis
        axis_permutation = options.axis_permutation
        quat_format = options.quat_format

        pose_data = np.array(pose_data)
        trans_data = np.array(trans_data)
        
        # Reshape trans_data to (F, 3)
        if trans_data.ndim == 1:
            if trans_data.size >= 3:
                trans_data = trans_data[:3].reshape(1, 3) # (1, 3)
            else:
                trans_data = np.zeros((1, 3)) # Fallback
        elif trans_data.ndim == 2:
            if trans_data.shape[1] > 3:
                trans_data = trans_data[:, :3] # Slice extra components
            elif trans_data.shape[1] < 3:
                 pass
        
        # --- Robust Input Reshap/Padding ---
        
        # 1. Handle Flattened Inputs
        if pose_data.ndim == 1:
            if pose_data.size == 66: # 22*3 Axis Angle
                pose_data = pose_data.reshape(1, 22, 3)
            elif pose_data.size == 88: # 22*4 Quats
                pose_data = pose_data.reshape(1, 22, 4)
            elif pose_data.size == 72: # 24*3
                pose_data = pose_data.reshape(1, 24, 3)
            elif pose_data.size == 96: # 24*4
                pose_data = pose_data.reshape(1, 24, 4)
            else:
                 pose_data = pose_data[np.newaxis, ...] # Fallback
                 
        # 2. Handle Single Frame 2D Inputs
        if pose_data.ndim == 2:
            if pose_data.shape[0] == 22: # (22, C)
                 pose_data = pose_data[np.newaxis, ...]
            elif pose_data.shape[0] == 24: # (24, C)
                 if pose_data.shape[1] in [3, 4]:
                     pose_data = pose_data[np.newaxis, ...]

        # 3. Handle 22-Joint Padding (F, 22, C) -> (F, 24, C)
        if pose_data.ndim == 3 and pose_data.shape[1] == 22:
            F = pose_data.shape[0]
            C = pose_data.shape[2]
            p24 = np.zeros((F, 24, C))
            p24[:, :22, :] = pose_data
            
            # Pad Identity if Quats
            if C == 4:
                if quat_format == 'xyzw':
                     p24[:, 22:, 3] = 1.0
                else: # wxyz
                     p24[:, 22:, 0] = 1.0
            
            pose_data = p24
            
        F = pose_data.shape[0]
        
        # 4. Handle AMASS / Extra Joints (Crop to 24)
        if pose_data.shape[-1] > 72 and input_type == 'axis_angle':
            temp = pose_data.reshape(F, -1, 3)
            pose_data = temp[:, :24, :].reshape(F, -1)
            
        elif pose_data.shape[-1] > 96 and input_type == 'quat':
             temp = pose_data.reshape(F, -1, 4)
             pose_data = temp[:, :24, :].reshape(F, -1)
        
        # Convert to Axis Angle
        if input_type == 'axis_angle':
            if pose_data.shape[-1] == 72:
                pose_data_aa = pose_data.reshape(F, 24, 3)
            else:
                pose_data_aa = pose_data
                
            flat_pose = pose_data_aa.reshape(-1, 3)
            r = R.from_rotvec(flat_pose)
            quats = r.as_quat().reshape(F, 24, 4)
            
        elif input_type == 'quat':
             quats = pose_data.reshape(F, 24, 4).copy()
             
             # Handle Format
             if quat_format == 'wxyz':
                 quats = np.roll(quats, -1, axis=-1)
             
             flat_q = quats.reshape(-1, 4)
             r = R.from_quat(flat_q)
             pose_data_aa = r.as_rotvec().reshape(F, 24, 3)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
            
        # --- Handle Coordinate System Conversion ---
        # Default Internal Up Axis is Y (1)
        self.internal_y_dim = 1
        
        if axis_permutation is not None:
             self.set_axis_permutation(axis_permutation)
             
        if hasattr(self, 'perm_basis') and self.perm_basis is not None:
             basis = self.perm_basis
             basis_for_rot = self.perm_basis_rot
             
             # Transform Translation
             trans_data = trans_data @ basis.T
             
             # Transform Root Orientation
             r_perm = R.from_matrix(basis_for_rot)
             r_root = R.from_rotvec(pose_data_aa[:, 0, :])
             r_root_new = r_perm * r_root
             pose_data_aa[:, 0, :] = r_root_new.as_rotvec()
             
             # Regenerate quats
             flat_pose = pose_data_aa.reshape(-1, 3)
             r_all = R.from_rotvec(flat_pose)
             quats = r_all.as_quat().reshape(F, 24, 4)
             
             r_all = R.from_rotvec(flat_pose)
             quats = r_all.as_quat().reshape(F, 24, 4)
             
        # Post-Permutation Alignment
        # If the resulting data is Z-up (as indicated by options), rotate it to Y-up.
        if input_up_axis == 'Z':
            # Convert Z-up to Y-up
            tx, ty, tz = trans_data[..., 0], trans_data[..., 1], trans_data[..., 2]
            trans_data = np.stack([tx, tz, -ty], axis=-1)
            
            r_conv = R.from_euler('x', -90, degrees=True)
            r_root = R.from_rotvec(pose_data_aa[:, 0, :])
            r_root_new = r_conv * r_root
            pose_data_aa[:, 0, :] = r_root_new.as_rotvec()
            
            flat_pose = pose_data_aa.reshape(-1, 3)
            r_all = R.from_rotvec(flat_pose)
            quats = r_all.as_quat().reshape(F, 24, 4)
            
            # Rotated to Y-up
            self.internal_y_dim = 1
            
        elif input_up_axis == 'Y':
            pass
            
        else:
             print(f"Warning: input_up_axis='{input_up_axis}' not fully supported. Assuming Y-up.")
             
        return trans_data, pose_data_aa, quats

    def _compute_forward_kinematics(self, trans_data, quats):
        """
        Computes World Positions and Tip Locations via Forward Kinematics.
        Args:
            trans_data (F, 3): Root translation.
            quats (F, 24, 4): Joint rotations.
        Returns:
            world_pos (F, 30, 3): Global positions.
            global_rots (List[R]): Global rotation objects per joint (relative to world).
            tips (dict): Index -> Position (F, 3) for end effectors (Feet, Hands).
        """
        F = trans_data.shape[0]
        # Extending to 30 output joints (24 original + 6 virtual: Toes, Fingers, Heels)
        world_pos = np.zeros((F, 30, 3))
        
        # Optimization: Create ONE Rotation object for all joints at once to speed up init
        # We need frames for a specific joint to be contiguous for easy slicing later.
        # quats input is (F, 24, 4), so we transpose to (24, F, 4)
        quats_perm = quats.transpose(1, 0, 2).reshape(-1, 4) # (24*F, 4)
        all_rots = R.from_quat(quats_perm)
        
        # We store slices for each joint to avoid re-slicing (though slicing R is cheapish)
        # Actually we just slice on the fly to keep code clean.
        
        global_rots = [None] * 30
        parents = self._get_hierarchy()

        for i in range(30):
            parent = parents[i]
            
            if i < 24:
                # Slice the global rotation object
                # Because we transposed to (24, F, 4), the frames for joint i are at [i*F : (i+1)*F]
                start = i * F
                end = (i + 1) * F
                rot_local = all_rots[start:end]
            else:
                # Virtual Joint (Fixed extension)
                # Local rotation is Identity.
                rot_local = None # Flag to skip multiplication
            
            if parent == -1:
                global_rots[i] = rot_local
                world_pos[:, i, :] = trans_data # (F, 3)
            else:
                if rot_local is not None:
                    global_rots[i] = global_rots[parent] * rot_local
                else:
                    # Identity local: Global = Parent Global
                    global_rots[i] = global_rots[parent]
                
                # Calculate Offset (using 30-element skeleton_offsets)
                local_pos = self.skeleton_offsets[i]

                # Apply Rotation (Parent's Global Rotation) to the Local Offset
                offset_vec = global_rots[parent].apply(local_pos)
                world_pos[:, i, :] = world_pos[:, parent, :] + offset_vec
                
        # --- Tip Projection for Contact Detection ---
        tips = {} # index -> pos (F, 3)
        
        # Use Virtual End-Effectors (Indices 24-29) for accurate contact
        # L_Foot (10) -> L_Toe (24)
        # R_Foot (11) -> R_Toe (25)
        # L_Hand (22) -> L_Tip (26)
        # R_Hand (23) -> R_Tip (27)
        
        # Verify we have enough joints (28)
        if world_pos.shape[1] >= 28:
            # User Contact Indices Convention:
            # 7/8: L/R Ankle (Native).
            # 10/11: L/R Foot/ToeBase (Native).
            # 22/23: L/R Hand (Native) - Do NOT overwrite.
            # 24/25: L/R Toe Tip (Virtual).
            # 26/27: L/R Finger Tip (Virtual).
            
            # Map Indices 24-27 to Virtual Tips
            tips[24] = world_pos[:, 24, :] # L_Toe (Virtual) -> Index 24
            tips[25] = world_pos[:, 25, :] # R_Toe (Virtual) -> Index 25
            tips[26] = world_pos[:, 26, :] # L_Finger (Virtual) -> Index 26
            tips[27] = world_pos[:, 27, :] # R_Finger (Virtual) -> Index 27
            
            # Remove Legacy overrides if they exist
            # (We do not want to override 10/11 with tips if they are explicitly Foot joints)
        else:
             # Fallback if virtual joints missing (should not happen with new logic)
             pass
                  
        # Store tips and world_pos for later contact logic/debug
        self.temp_tips = tips 
        self.last_world_pos = world_pos 
        
        return world_pos, global_rots, tips




    def _compute_angular_kinematics(self, F, pose_data_aa, quats, options, use_filter=True, state_suffix=''):
        """
        Computes angular velocity and acceleration.
        Dual Path Support: Can optionally bypass filter and use separate state history.
        
        Args:
            F (int): Frame count.
            pose_data_aa (F, 24, 3): Pose axis-angle.
            quats (F, 24, 4): Pose quaternions.
            options (SMPLProcessingOptions): Usage: dt, filter_min_cutoff, filter_beta.
            use_filter (bool): Whether to use OneEuroFilter (True for Contact Path, False for Effort Path).
            state_suffix (str): Suffix for state variables (e.g. '', '_effort').
            
        Returns:
            ang_vel (F, 24, 3): Angular velocity vectors.
            ang_acc (F, 24, 3): Angular acceleration vectors.
        """
        dt = options.dt
        
        # Resolve State Variable Names
        name_prev_pose_aa = 'prev_pose_aa' + state_suffix
        name_prev_pose_q = 'prev_pose_q' + state_suffix
        name_prev_vel_aa = 'prev_vel_aa' + state_suffix
        
        # Access State
        prev_pose_aa = getattr(self, name_prev_pose_aa, None)
        prev_pose_q = getattr(self, name_prev_pose_q, None)
        prev_vel_aa = getattr(self, name_prev_vel_aa, None)
        
        # KEY FIX: Always enter streaming block if F==1. 
        # Do not rely on hasattr() because the attribute might not exist yet for new suffixes.
        if F == 1:
            # Streaming mode
            curr_aa = pose_data_aa[0] # (24, 3)
            
            if prev_pose_aa is None:
                # First frame, no velocity/accel
                ang_vel = np.zeros_like(curr_aa)
                ang_acc = np.zeros_like(curr_aa)
                
                # Init State (using setattr for dynamic names)
                setattr(self, name_prev_pose_aa, curr_aa)
                setattr(self, name_prev_vel_aa, ang_vel)
                
                # Check for Shape Mismatch (Re-init filter if needed)
                if use_filter and options.enable_one_euro_filter and self.pose_filter is None:
                     self.pose_filter = OneEuroFilter(min_cutoff=options.filter_min_cutoff, beta=options.filter_beta, framerate=self.framerate)
            else:
                dt = 1.0 / self.framerate if self.framerate > 0 else 0.016
                
                filtered_q = None
                
                # One Euro Filter for Pose
                if use_filter and options.enable_one_euro_filter:
                    if self.pose_filter is None:
                        # Initialize
                        self.pose_filter = OneEuroFilter(min_cutoff=options.filter_min_cutoff, beta=options.filter_beta, framerate=self.framerate)
                    
                    # Update params dynamicallly (if changed via UI)
                    self.pose_filter._mincutoff = options.filter_min_cutoff
                    self.pose_filter._beta = options.filter_beta
                    
                    # Filter the signal (Pose Quaternions to handle wrapping)
                    curr_q = quats[0].flatten() # (96,)
                    
                    # Check for Shape Mismatch (Re-init filter if needed)
                    if self.pose_filter._x.last_value is not None:
                        if self.pose_filter._x.last_value.shape != curr_q.shape:
                            # Reset filter state completely
                            self.pose_filter = OneEuroFilter(min_cutoff=options.filter_min_cutoff, beta=options.filter_beta, framerate=self.framerate)
                            setattr(self, name_prev_pose_q, None)
                            prev_pose_q = None
                            
                    # Align Quaternions to avoid flips (Double Cover)
                    if self.pose_filter._x.last_value is not None:
                        prev_filtered_q = self.pose_filter._x.last_value
                        c_rs = curr_q.reshape(-1, 4)
                        p_rs = prev_filtered_q.reshape(-1, 4)
                        dots = np.sum(c_rs * p_rs, axis=1) # (24,)
                        mask = dots < 0
                        if np.any(mask):
                            c_rs[mask] *= -1.0
                            curr_q = c_rs.flatten()
                    
                    # Filter
                    filtered_q_flat = self.pose_filter(curr_q)
                    filtered_q = filtered_q_flat.reshape(24, 4)
                else:
                    # Bypass Filter
                    curr_q = quats[0] # (24, 4)
                    
                    # Ensure continuity (Unwrap) just like Filter does
                    # This prevents numerical discontinuities if Quats flip sign
                    # Although Rotation object handles it, manual unwrap is safer for debugging/consistency
                    if prev_pose_q is not None:
                         # prev_pose_q is the LAST STORED STATE (Unfiltered)
                         # We need to access it properly.
                         # Logic below accesses it via `getattr`.
                         pass 
                         
                    # We can't easily access 'prev_pose_q' here because it's loaded later in the function?
                    # No, it's loaded at line 2506 *after* filtering.
                    # To unwrap, we need the previous value NOW.
                    
                    # Access Previous State
                    prev_state_q = getattr(self, name_prev_pose_q, None)
                    
                    if prev_state_q is not None:
                        # Align to previous
                         c_rs = curr_q.reshape(-1, 4)
                         p_rs = prev_state_q.reshape(-1, 4)
                         dots = np.sum(c_rs * p_rs, axis=1)
                         mask = dots < 0
                         if np.any(mask):
                             c_rs[mask] *= -1.0
                             curr_q = c_rs 
                             
                    filtered_q = curr_q
                
                # Normalize Filtered Quats
                norms = np.linalg.norm(filtered_q, axis=1, keepdims=True)
                filtered_q /= (norms + 1e-8)
                
                # Velocity computation using FILTERED Quaternions
                r_curr = R.from_quat(filtered_q)
                
                # Need previous FILTERED quat (from State)
                if prev_pose_q is None:
                    ang_vel = np.zeros((24, 3))
                    raw_ang_acc = np.zeros((24, 3))
                    # Initialize history
                    setattr(self, name_prev_pose_q, filtered_q)
                else:
                    r_prev = R.from_quat(prev_pose_q)
                    r_diff = r_curr * r_prev.inv()
                    diff_vec = r_diff.as_rotvec()
                    
                    ang_vel = diff_vec.reshape(24, 3) / dt
                    
                    # Update History
                    setattr(self, name_prev_pose_q, filtered_q)
                    
                    # Acceleration
                    if prev_vel_aa is None:
                        raw_ang_acc = np.zeros_like(ang_vel)
                    else:
                        raw_ang_acc = (ang_vel - prev_vel_aa) / dt
                        
                    # Update Velocity History
                    setattr(self, name_prev_vel_aa, ang_vel)
                
                # Smoothing implicitly handled by OneEuroFilter on Pose (if enabled)
                ang_acc = raw_ang_acc
                
                # Update AA History
                # We track raw AA for spike detection logic usually?
                # Or filtered AA? Original code tracked curr_aa (Raw/Input).
                setattr(self, name_prev_pose_aa, curr_aa)
                
                # Wait, I might have just assigned filtered_q (Quat) to prev_pose_aa (AA) in my thought process?
                # Let's be careful.
                


        else:
            # Batch Mode (F > 1)
            # Calculate derivatives across the batch
            dt = 1.0 / self.framerate
            
            # Robust Angular Velocity using Quaternions (Handling wrapping/discontinuities)
            # Vectorized using linear R array
            flat_quats = quats.reshape(-1, 4)
            r_all = R.from_quat(flat_quats)
            
            # Backward Difference: vel[t] = (rot[t] * rot[t-1].inv) / dt
            # We skip the first frame (index 0) as it has no prev
            # Slices: 24 (1 frame) to End vs 0 to End-24
            
            r_curr = r_all[24:]
            r_prev = r_all[:-24]
            
            r_diff = r_curr * r_prev.inv()
            vel_flat = r_diff.as_rotvec() / dt
            
            ang_vel = np.zeros((F, 24, 3))
            ang_vel[1:] = vel_flat.reshape(F-1, 24, 3)
            ang_vel[0] = ang_vel[1] # Pad start
            
            # Smoothing (Batch)
            smoothing = 0.0 # Disabled for now as param was removed.
            if smoothing > 0.0:
                 # Gaussian filter or similar?
                 # For consistency with streaming (Exponential Moving Average), we could simulate EMA.
                 # Or just use scipy.ndimage.gaussian_filter1d
                 from scipy.ndimage import gaussian_filter1d
                 sigma = smoothing * 10.0 # Heuristic mapping
                 ang_vel = gaussian_filter1d(ang_vel, sigma=sigma, axis=0)

            # 2. Angular Acceleration
            ang_acc = np.gradient(ang_vel, dt, axis=0)
            
            if smoothing > 0.0:
                 ang_acc = gaussian_filter1d(ang_acc, sigma=sigma, axis=0)

            # Acceleration Clamping (Batch)
            # Acceleration Clamping REMOVED per task.
                    
        return ang_vel, ang_acc



    def _compute_joint_torques(self, F, ang_acc, world_pos, parents, global_rots, pose_data_aa, tips, options, contact_forces=None):
        """
        Computes Joint Torques, Efforts, and Inertias.
        Iterates through joints to calculate dynamic vs gravity torques.
        
        Args:
            F (int): Frame count.
            ang_acc (F, 24, 3): Angular acceleration.
            world_pos (F, 24, 3): World positions.
            parents (list): Parent indices.
            global_rots (list[R]): Global rotation objects per joint.
            pose_data_aa (F, 24, 3): Pose for passive limits.
            tips (dict): Cached tip positions (head, hands, feet).
            contact_forces (F, 24, 3): Contact forces (gravity distribution).
            
        Returns:
            torques, inertias, efforts_net, efforts_dyn, efforts_grav, t_dyn_vecs, t_grav_vecs, t_passive_vecs
        """
        # Output containers
        torques_vec = np.zeros((F, self.target_joint_count, 3))
        inertias = np.zeros((F, self.target_joint_count))
        efforts_net = np.zeros((F, self.target_joint_count))
        efforts_dyn = np.zeros((F, self.target_joint_count))
        efforts_grav = np.zeros((F, self.target_joint_count))
        
        # Vectors debugging arrays
        t_net_vecs = np.zeros((F, self.target_joint_count, 3))
        t_dyn_vecs = np.zeros((F, self.target_joint_count, 3))
        t_grav_vecs = np.zeros((F, self.target_joint_count, 3))
        t_passive_vecs = np.zeros((F, self.target_joint_count, 3))
        
        
        
        # Inverse Rotations for Torque Projection (World -> Local)
        # 2. Torque / Effort Calculation (Vectorized)
        
        # Pre-compute Gravity Torques
        t_grav_vecs = self._compute_gravity_torques(world_pos, options, global_rots, tips)
        
        # Pre-compute Contact Torques (World Frame)
        t_contact_vecs = np.zeros((F, self.target_joint_count, 3))
        if contact_forces is not None:
             # Iterate all contact joints (including Tips/Virtual if present)
             num_contact_indices = contact_forces.shape[1]
             for c in range(num_contact_indices):
                  # Check if any force present in batch
                  f_c = contact_forces[:, c, :] # (F, 3)
                  if np.all(np.max(np.abs(f_c), axis=1) < 1e-6): continue
                  
                  p_c = world_pos[:, c, :] # (F, 3)
                  
                  # Traverse up
                  curr = c
                  while curr != -1:
                       if curr < self.target_joint_count:
                            p_curr = world_pos[:, curr, :] # (F, 3)
                            r = p_c - p_curr # (F, 3)
                            tau = np.cross(r, f_c) # (F, 3)
                            t_contact_vecs[:, curr, :] += tau
                       
                       # Move to parent
                       if curr < len(parents):
                            curr = parents[curr]
                       else:
                            break
        


        # Optimization: Pre-compute Inverses of Global Rotations
        # Many joints share parents (e.g. Pelvis->Hips, Spine->Collars), so computing inv() inside loop is redundant.
        global_rot_invs = [r.inv() for r in global_rots]

        # --- PASS 1: Compute Raw Dynamic Torques and Inertias ---
        t_dyn_raw = np.zeros((F, self.target_joint_count, 3))
        
        for j in range(self.target_joint_count):
            # Inertia (F,)
            I_eff = self._compute_subtree_inertia(j, world_pos, None, self.limb_data['lengths'], self.limb_data['masses'])
            inertias[:, j] = I_eff
            
            # Alpha (F, 3)
            alpha_vec = ang_acc[:, j, :]
            
            # Dynamic Torque (F, 3) - raw computation
            torque_dyn = I_eff[:, np.newaxis] * alpha_vec
            t_dyn_raw[:, j, :] = torque_dyn
        
        # --- Apply Rate Limiting to Raw Dynamic Torques ---
        # This limits the maximum change per frame to prevent noise spikes
        t_dyn_limited = self._apply_torque_rate_limiting(t_dyn_raw, options)
        
        # --- PASS 2: Compute Net Torques, Passive, and Efforts using Limited Dynamic ---
        for j in range(self.target_joint_count):
            name = self.joint_names[j]
            
            # Use rate-limited dynamic torque
            torque_dyn = t_dyn_limited[:, j, :]
            t_dyn_vecs[:, j, :] = torque_dyn
            
            torque_grav = t_grav_vecs[:, j, :]
            torque_contact = t_contact_vecs[:, j, :]
            
            # Frame Correction:
            # torque_dyn is LOCAL/PARENT Frame (derived from Local AA).
            # torque_grav is WORLD Frame.
            # We must convert Gravity to Local before subtracting.
            
            # --- Transform to Parent Frame ---
            parent_idx = parents[j]
            if parent_idx != -1:
                # Inverse of Parent Rotation (Cached)
                r_parent_inv = global_rot_invs[parent_idx]
                
                # Transform Gravity to Local
                t_grav_local = r_parent_inv.apply(torque_grav)
                t_contact_local = r_parent_inv.apply(torque_contact)
                
                # Dynamic is already Local
                t_dyn_local = torque_dyn
                
                # Net Local
                t_net_local = t_dyn_local - t_grav_local - t_contact_local
                
            else:
                # Root: Parent is World (Identity)
                t_net_local = torque_dyn - torque_grav - torque_contact
                t_dyn_local = torque_dyn
                t_grav_local = torque_grav
                t_contact_local = torque_contact
            
            # Store transformed vectors for debug outputs if needed?
            # t_dyn_vecs is pre-filled with torque_dyn (Local)
            # t_grav_vecs is pre-filled with World.
            # We might want to return consistent frames? 
            # The return dict has 'torques_vec' (Net Local). 
            # 'torques_dyn_vec' (Currently Local).
            # 'torques_grav_vec' (Currently World).
            # Ideally debugging vectors should be in the same frame as Net?
            # But changing return types might break other viz. 
            # Let's keep t_grav_vecs (World) in the output for now, as it's "Raw Gravity".
            
            # --- Passive Limits ---
            t_passive_local = np.zeros_like(t_net_local)
            if options.enable_passive_limits:
                 # Current local pose for this joint
                 curr_pose_aa = pose_data_aa[:, j, :]
                 t_passive_local = self._compute_passive_torque(name, t_net_local, curr_pose_aa)
                 
                 # --- OPTIMAL PASSIVE SUPPORT (Minimizing Active Effort) ---
                 # Treat the calculated Passive Torque (t_passive_local) as a "Capacity".
                 # The structure CAN provide up to this amount, but only if needed to resist Net Load.
                 # It never pushes "actively" against the load (unless bouncing, which we ignore for effort).
                 # P_effective = clip(Net, min(0, P_model), max(0, P_model)).
                 
                 # Components where P_model is Negative (Extension Stop, Gravity Support, etc)
                 mask_neg = t_passive_local < 0
                 # For these components, P_eff is in [P_model, 0].
                 # We clip Net to this range.
                 t_passive_local[mask_neg] = np.clip(t_net_local[mask_neg], t_passive_local[mask_neg], 0)
                 
                 # Components where P_model is Positive (Flexion Stop, etc)
                 mask_pos = t_passive_local >= 0
                 # For these components, P_eff is in [0, P_model].
                 t_passive_local[mask_pos] = np.clip(t_net_local[mask_pos], 0, t_passive_local[mask_pos])
                 
                 # Result: 
                 # If Net is 0 (Sideways): P_eff = clip(0, [P,0]) = 0. Active = 0.
                 # If Net opposes P: e.g. Net=+2, P=-6. P_eff = clip(2, [-6,0]) = 0. Active = +2.
                 # If Net matches P (within): Net=-2, P=-6. P_eff = clip(-2, [-6,0]) = -2. Active = 0.
                 # If Net exceeds P: Net=-10, P=-6. P_eff = clip(-10, [-6,0]) = -6. Active = -4.
            
            # Active Active (Net - Passive)
            t_active_local = t_net_local - t_passive_local
            
            torques_vec[:, j, :] = t_active_local
            
            # --- Effort Calculation (Normalized) ---
            # Max torque for this joint (Vector)
            t_max_vec = self.max_torque_array[j] # (3,)
            
            # Use abs(torque) / vector_limit elementwise
            # Add epsilon to prevent div by zero
            denom = t_max_vec + 1e-6
            
            eff_net_vec = np.abs(t_active_local) / denom # (F, 3) 
            eff_dyn_vec = np.abs(t_dyn_local) / denom
            eff_grav_vec = np.abs(t_grav_local) / denom
            
            # Scalar Effort: L2 Norm of the normalized vector
            # If t_max was isotropic (L), this is norm(t/L) = norm(t)/L. Compatible.
            efforts_net[:, j] = np.linalg.norm(eff_net_vec, axis=1)
            efforts_dyn[:, j] = np.linalg.norm(eff_dyn_vec, axis=1)
            efforts_grav[:, j] = np.linalg.norm(eff_grav_vec, axis=1)
            
            # Store vectors (in Local Frame?)
            # t_net_vecs was accumulating t_net_local (which is post-passive subtraction: Active)
            t_net_vecs[:, j, :] = t_active_local
            t_dyn_vecs[:, j, :] = t_dyn_local
            t_grav_vecs[:, j, :] = t_grav_local # Updated: Now Local Frame (Summable)
            t_passive_vecs[:, j, :] = t_passive_local

        return torques_vec, inertias, efforts_net, efforts_dyn, efforts_grav, t_dyn_vecs, t_grav_vecs, t_passive_vecs



    def _apply_torque_rate_limiting(self, t_dyn_vecs, options):
        """
        Apply rate limiting to dynamic torque vectors to prevent glitches.
        
        Limits the maximum change in dynamic torque per frame based on per-joint
        biomechanical limits, modulated by the rate_limit_strength parameter.
        
        When clamping would reduce the delta by more than 90% (indicating a likely
        teleport or noise spike), the joint's torque is held steady at the previous
        value to avoid propagating any artifact.
        
        Args:
            t_dyn_vecs (F, 24, 3): Raw dynamic torque vectors.
            options (SMPLProcessingOptions): Contains enable_rate_limiting and rate_limit_strength.
            
        Returns:
            t_dyn_limited (F, 24, 3): Rate-limited dynamic torque vectors.
        """
        if not options.enable_rate_limiting:
            # Update state but don't limit
            self.prev_dynamic_torque = t_dyn_vecs.copy()
            return t_dyn_vecs
        
        F = t_dyn_vecs.shape[0]
        t_dyn_limited = t_dyn_vecs.copy()
        
        # Initialize previous state if needed
        if self.prev_dynamic_torque is None:
            self.prev_dynamic_torque = np.zeros((self.target_joint_count, 3))
        
        # Ensure shape compatibility
        if self.prev_dynamic_torque.shape[0] != t_dyn_vecs.shape[1]:
            self.prev_dynamic_torque = np.zeros((t_dyn_vecs.shape[1], 3))
        
        # Ensure rate limits exist
        if not hasattr(self, 'torque_rate_limits') or self.torque_rate_limits is None:
            self.torque_rate_limits = self._compute_torque_rate_limits()
        
        # Scale rate limits by framerate and strength parameter
        # Base limits are calibrated for 60 FPS
        # Higher framerate = smaller dt = must scale down the per-frame limit
        fps_scale = self.framerate / 60.0  # e.g., 120 FPS -> 2.0
        strength = max(0.01, options.rate_limit_strength)  # Prevent division issues
        
        # Effective limit per frame (N·m)
        # At higher framerate, each frame has smaller dt -> smaller allowed change
        # strength > 1.0 -> more permissive (allows faster changes)
        # strength < 1.0 -> more conservative (smoother)
        effective_limits = (self.torque_rate_limits / fps_scale) * strength
        
        # Threshold for considering a delta as a "spike" requiring full suppression
        # If delta exceeds the limit by this multiplier, it's clearly non-physical
        SPIKE_MULTIPLIER = 1.5  # If delta > 1.5x the allowed limit, suppress entirely
        NEIGHBOR_MULTIPLIER = 1.0  # Adjacent joints use 1.0x threshold when neighbor spikes
        
        # Jitter detection parameters
        JITTER_WINDOW = 8  # frames to track sign changes
        JITTER_THRESHOLD = 0.3  # sign flips per frame threshold (more = jittery)
        JITTER_DAMPING = 0.3  # damping factor for jittery joints (0 = full damping, 1 = no damping)
        
        num_joints = min(self.target_joint_count, t_dyn_vecs.shape[1])
        
        # Initialize jitter detection state if needed
        if self.torque_sign_history is None:
            self.torque_sign_history = np.zeros((JITTER_WINDOW, num_joints, 3))
            self.jitter_history_idx = 0
        
        # Ensure jitter history shape matches
        if self.torque_sign_history.shape[1] != num_joints:
            self.torque_sign_history = np.zeros((JITTER_WINDOW, num_joints, 3))
            self.jitter_history_idx = 0
        
        # Build adjacency map from hierarchy (parents + children)
        parents_list = self._get_hierarchy()
        
        # Build children lookup
        children = {j: [] for j in range(num_joints)}
        for j in range(num_joints):
            parent = parents_list[j]
            if parent >= 0 and parent < num_joints:
                children[parent].append(j)
        
        # Per-frame noise tracking
        frame_spikes = 0
        frame_rate_limits = 0
        frame_jitter = 0
        frame_innovation_clamps = 0
        
        for f in range(F):
            curr_torque = t_dyn_vecs[f]
            prev_torque = self.prev_dynamic_torque
            
            # Delta torque per joint
            delta = curr_torque - prev_torque
            delta_mags = np.linalg.norm(delta, axis=1)  # (num_joints,)
            curr_mags = np.linalg.norm(curr_torque, axis=1)
            prev_mags = np.linalg.norm(prev_torque, axis=1)
            
            # --- PASS 1: Detect primary spikes ---
            # Only trigger on INCREASES - sudden decreases are usually legitimate motion ending
            primary_spikes = set()
            for j in range(num_joints):
                is_large_delta = delta_mags[j] > effective_limits[j] * SPIKE_MULTIPLIER
                is_increasing = curr_mags[j] > prev_mags[j]  # Only suppress increases
                if is_large_delta and is_increasing:
                    primary_spikes.add(j)
                    frame_spikes += 1
                    # Track severity: how much the delta exceeded the threshold
                    excess = delta_mags[j] - effective_limits[j] * SPIKE_MULTIPLIER
                    self.noise_stats.spike_severity += excess
                    self.noise_stats.joint_spike_counts[j] += 1
                    self.noise_stats.joint_spike_severity[j] += excess
            
            # --- PASS 2: Propagate to neighbors (parent + children) ---
            neighbor_suspect = set()
            for j in primary_spikes:
                # Add parent
                parent = parents_list[j]
                if parent >= 0 and parent < num_joints:
                    neighbor_suspect.add(parent)
                # Add children
                for child in children[j]:
                    neighbor_suspect.add(child)
            
            # --- PASS 3: Apply suppression with adjusted thresholds ---
            for j in range(num_joints):
                if j in primary_spikes:
                    # Primary spike: full suppression - hold at previous
                    t_dyn_limited[f, j, :] = prev_torque[j, :]
                elif j in neighbor_suspect:
                    # Neighbor of spike: use stricter threshold (only on increases)
                    neighbor_limit = effective_limits[j] * NEIGHBOR_MULTIPLIER
                    is_neighbor_increasing = curr_mags[j] > prev_mags[j]
                    if delta_mags[j] > neighbor_limit and is_neighbor_increasing:
                        # Neighbor exceeds stricter threshold and is increasing: suppress
                        t_dyn_limited[f, j, :] = prev_torque[j, :]
                    else:
                        # Normal rate limiting
                        for axis in range(3):
                            if delta[j, axis] > effective_limits[j]:
                                t_dyn_limited[f, j, axis] = prev_torque[j, axis] + effective_limits[j]
                                frame_rate_limits += 1
                                self.noise_stats.joint_rate_limit_counts[j] += 1
                            elif delta[j, axis] < -effective_limits[j]:
                                t_dyn_limited[f, j, axis] = prev_torque[j, axis] - effective_limits[j]
                            else:
                                t_dyn_limited[f, j, axis] = curr_torque[j, axis]
                else:
                    # Normal joint: standard rate limiting
                    for axis in range(3):
                        if delta[j, axis] > effective_limits[j]:
                            t_dyn_limited[f, j, axis] = prev_torque[j, axis] + effective_limits[j]
                            frame_rate_limits += 1
                            self.noise_stats.joint_rate_limit_counts[j] += 1
                        elif delta[j, axis] < -effective_limits[j]:
                            t_dyn_limited[f, j, axis] = prev_torque[j, axis] - effective_limits[j]
                        else:
                            t_dyn_limited[f, j, axis] = curr_torque[j, axis]
            
            # --- PASS 4: Jitter Detection and Damping ---
            # Update sign history buffer
            curr_signs = np.sign(t_dyn_limited[f])  # (num_joints, 3)
            self.torque_sign_history[self.jitter_history_idx] = curr_signs
            self.jitter_history_idx = (self.jitter_history_idx + 1) % JITTER_WINDOW
            
            # Compute sign-flip rate per joint
            # Count how many times sign changes between consecutive history entries
            sign_changes = np.sum(np.abs(np.diff(self.torque_sign_history, axis=0)) > 0, axis=(0, 2))  # (num_joints,)
            flip_rate = sign_changes / (JITTER_WINDOW - 1)  # flips per frame
            
            # Apply damping to jittery joints
            for j in range(num_joints):
                if flip_rate[j] > JITTER_THRESHOLD:
                    # High jitter detected - blend toward previous (damping)
                    # More jitter = more damping
                    jitter_excess = (flip_rate[j] - JITTER_THRESHOLD) / (1.0 - JITTER_THRESHOLD + 1e-6)
                    jitter_excess = min(1.0, jitter_excess)  # cap at 1.0
                    damping = 1.0 - jitter_excess * (1.0 - JITTER_DAMPING)
                    t_dyn_limited[f, j, :] = damping * t_dyn_limited[f, j, :] + (1.0 - damping) * prev_torque[j, :]
                    frame_jitter += 1
                    self.noise_stats.joint_jitter_counts[j] += 1
            
            # --- PASS 5: SmartClampKF Filtering ---
            # Apply Kalman filter with innovation clamping for final smoothing
            # This provides principled noise reduction while preventing teleportation
            
            if options.enable_kf_smoothing:
                # Filter parameters (user-tuned values)
                RESPONSIVENESS = 10.0   # Higher = faster tracking
                SMOOTHNESS = 1.0        # Higher = smoother output
                CLAMP_RADIUS = 15.0     # Max innovation per frame (N·m)
                
                # Initialize filter if needed
                if self.dynamic_torque_kf is None:
                    dt = 1.0 / self.framerate
                    self.dynamic_torque_kf = NumpySmartClampKF(dt, num_joints, 3)
                    self.dynamic_torque_kf.update_params(RESPONSIVENESS, SMOOTHNESS, CLAMP_RADIUS, dt)
                
                # Predict and Update
                self.dynamic_torque_kf.predict()
                
                # Track innovation before update to count clamps
                pre_update = t_dyn_limited[f].copy()
                t_dyn_limited[f] = self.dynamic_torque_kf.update(t_dyn_limited[f])
                
                # Count innovation clamps (KF clamped the update)
                innovation_mag = np.linalg.norm(pre_update - t_dyn_limited[f], axis=1)
                clamp_count = np.sum(innovation_mag > CLAMP_RADIUS * 0.9)  # Near clamp limit
                frame_innovation_clamps += clamp_count
            
            # Update prev for next frame in batch
            self.prev_dynamic_torque = t_dyn_limited[f].copy()
            
            # Update cumulative noise stats for this frame
            self.noise_stats.total_frames += 1
            self.noise_stats.spike_detections += frame_spikes
            self.noise_stats.rate_limit_clamps += frame_rate_limits
            self.noise_stats.jitter_damping_events += frame_jitter
            self.noise_stats.innovation_clamps += frame_innovation_clamps
            
            # Reset per-frame counters for next frame
            frame_spikes = 0
            frame_rate_limits = 0
            frame_jitter = 0
            frame_innovation_clamps = 0
        
        return t_dyn_limited

    def process_frame(self, pose_data, trans_data, options, effort_pose_data=None):
        """
        Process a single frame or batch of frames.
        Tracks state for torque calculation if frames are passed one by one.
        quat_format: 'xyzw' (default, Scipy) or 'wxyz' (Scalar first). Only used if input_type='quat'.
        effort_pose_data: Optional separate pose stream for Calculation of Effort (AngAcc).
        """
        # Prepare Data (Reshape, Permute, Convert)
        trans_data, pose_data_aa, quats = self._prepare_trans_and_pose(
            pose_data, trans_data, options
        )
        
        # Reset Per-Frame Caches
        self.current_com = None
        self.current_total_mass = None
        
        F = trans_data.shape[0]
        # 1. Forward Kinematics (Vectorized)
        world_pos, global_rots, tips = self._compute_forward_kinematics(trans_data, quats)
        parents = self._get_hierarchy()





        # --- Angular Kinematics (Main Path: Contact/Gravity) ---
        # Uses OneEuroFilter if enabled
        ang_vel, ang_acc = self._compute_angular_kinematics(
            F, pose_data_aa, quats, options, use_filter=options.enable_one_euro_filter, state_suffix=''
        )
        
        # --- Angular Kinematics (Dual Path: Effort) ---
        ang_acc_for_effort = ang_acc # Default to main path
        
        if effort_pose_data is not None:
             # Process Effort Pose (Assume same Trans/Options)
             # Use a dummy trans since we only care about rotations/quats
             dummy_trans = trans_data
             _, effort_aa, effort_quats = self._prepare_trans_and_pose(
                 effort_pose_data, dummy_trans, options
             )
             
             # Compute Kinematics for Effort
             # Uses filter if enabled, separate state suffix
             _, effort_ang_acc = self._compute_angular_kinematics(
                 F, effort_aa, effort_quats, options, use_filter=options.enable_one_euro_filter, state_suffix='_effort'
             )
             ang_acc_for_effort = effort_ang_acc


        if ang_acc_for_effort.ndim == 2:
             ang_acc_for_effort = ang_acc_for_effort[np.newaxis, ...]
             
        if ang_acc.ndim == 2:
             ang_acc = ang_acc[np.newaxis, ...]
        
        # --- Probabilistic Contact (Moved Early for Dynamics) ---
        # --- Probabilistic Contact (Moved Early for Dynamics) ---
        self._update_physics_state(
             world_pos, pose_data_aa, options
        )
        
        # --- Velocity Prep for Fusion ---
        smoothed_vel_vec = self.prob_smoothed_vel # (J, 3)
        if F == 1:
             vel_y_in = smoothed_vel_vec[np.newaxis, :, getattr(self, 'internal_y_dim', 1)]
             # Horizontal
             yd = getattr(self, 'internal_y_dim', 1)
             pd = [0, 2] if yd == 1 else [0, 1]
             vel_h_in = np.linalg.norm(smoothed_vel_vec[np.newaxis, :, pd], axis=-1)
        else:
             # Batch - simplified
             vel_y_in = np.zeros((F, world_pos.shape[1]))
             vel_h_in = np.zeros((F, world_pos.shape[1]))
             
        contact_probs_fusion = self._compute_probabilistic_contacts_fusion(
             F, world_pos.shape[1], world_pos, vel_y_in, vel_h_in, options.floor_height, options
        )
        
        # --- Weighted Mass Distribution (for RNE GRF) ---
        # 1. Total Contact Mass (kg)
        total_contact_mass = self.total_mass_kg
        
        # 2. Distribute based on Fusion Probs
        # self.contact_pressure stores MASS per joint (kg)
        self.contact_pressure = np.zeros((F, world_pos.shape[1]))
        
        for f in range(F):
             probs = contact_probs_fusion[f] # (J,)
             
             # ZMP-Based Weighting (Physics)
             # Basic Prob is purely geometric (touching). 
             # Load Bearing depends on ZMP alignment.
             
             zmp_f = self.current_zmp[f] if self.current_zmp.ndim > 1 else self.current_zmp
             yd = getattr(self, 'internal_y_dim', 1)
             pd = [0, 2] if yd == 1 else [0, 1]
             zmp_hz = zmp_f[pd]
             
             w_dist = np.zeros_like(probs)
             
             # Calculate IDW Weights
             for j in range(probs.shape[0]):
                  p = probs[j]
                  if p < 0.01: continue
                  
                  # Pos
                  if j in tips: pos = tips[j][f]
                  else: pos = world_pos[f, j]
                  
                  pos_hz = pos[pd]
                  dist_sq = np.sum((pos_hz - zmp_hz)**2)
                  
                  # Weight = P / (Dist^2 + epsilon)
                  # Epsilon 0.005 (approx 7cm radius peak)
                  w_dist[j] = p / (dist_sq + 0.005)
                  
             # Normalize
             w_sum = np.sum(w_dist)
             
             if w_sum > 1e-6:
                  weights = w_dist / w_sum
                  
                  # Hard Height Cutoff (Prevent Ghost Contact at > 15cm)
                  heights = world_pos[f, :, yd] - options.floor_height
                  mask_high = heights > 0.15
                  weights[mask_high] = 0.0
                  
                  # Re-normalize after cutoff
                  w_sum_2 = np.sum(weights)
                  if w_sum_2 > 0:
                       weights = weights / w_sum_2
                       self.contact_pressure[f] = weights * total_contact_mass
                  else:
                       self.contact_pressure[f] = 0.0
             else:
                  self.contact_pressure[f] = 0.0      
        # --- Contact Pressure Smoothing ---
        # Consistent smoothing to reduce ZMP-based weight oscillation
        # Using a moderate alpha to balance responsiveness and stability
        
        ALPHA_CP = 0.15  # 15% new per frame (~7 frame settling time at 120fps)
        
        # Initialize smoothed state if needed
        if not hasattr(self, 'prev_contact_pressure_smooth') or self.prev_contact_pressure_smooth is None:
            self.prev_contact_pressure_smooth = self.contact_pressure.copy()
        
        # Ensure shape matches
        if self.prev_contact_pressure_smooth.shape != self.contact_pressure.shape:
            self.prev_contact_pressure_smooth = self.contact_pressure.copy()
        
        for f in range(F):
            curr = self.contact_pressure[f]
            prev = self.prev_contact_pressure_smooth[0] if F == 1 else self.prev_contact_pressure_smooth[f]
            
            # Simple consistent blend (no asymmetric mode switching)
            smoothed = prev * (1.0 - ALPHA_CP) + curr * ALPHA_CP
            self.contact_pressure[f] = smoothed
            
            # Update prev for next frame in batch
            if F == 1:
                self.prev_contact_pressure_smooth = smoothed[np.newaxis, :].copy()
            else:
                self.prev_contact_pressure_smooth[f] = smoothed

        # --- Joint Torque & Effort Calculation ---
        # Key: Pass contact_forces!
        torques_vec, inertias, efforts_net, efforts_dyn, efforts_grav, t_dyn_vecs, t_grav_vecs, t_passive_vecs = self._compute_joint_torques(
            F, ang_acc_for_effort, world_pos, parents, global_rots,
            pose_data_aa, tips, options, contact_forces=None
        )
        

        
        # Scalar torque magnitude for output
        # torques is now removed as per user request


            
        output_quats = quats[:, :self.target_joint_count, :]
        

        
        # Update tip history
        if hasattr(self, 'temp_tips') and self.temp_tips:
             self.last_tip_positions = {k: v[-1].copy() for k, v in self.temp_tips.items()}

        # --- Output Dictionary ---
        # --- Output Dictionary ---
        res = {
            'pose': output_quats if options.return_quats else pose_data_aa[:, :self.target_joint_count, :],
            'trans': trans_data,
            'contact_probs': contact_probs_fusion, # Legacy alias
            'contact_probs_fusion': contact_probs_fusion, # New
            'torques_vec': torques_vec,
            'torques_grav_vec': t_grav_vecs,
            'torques_passive_vec': t_passive_vecs,

            'torques_dyn_vec': t_dyn_vecs,
            'inertias': inertias,
            'efforts_dyn': efforts_dyn,
            'efforts_grav': efforts_grav,
            'efforts_net': efforts_net,
            'positions': world_pos,

            'contact_pressure': self.contact_pressure
        }
        
        return res


    def align_to_up_axis(self, trans, rotations, input_axis='Y', target_axis='Z'):
        """
        Helper to rotate root pose and translation to new up axis.
        """
        if input_axis == target_axis:
            return trans, rotations

        # Rotation matrix to go from Y-up to Z-up (rotate -90 deg around X-axis)
        # Y -> Z, Z -> -Y, X -> X
        if input_axis == 'Y' and target_axis == 'Z':
            # Rx(-90)
            rot_convert = R.from_euler('x', -90, degrees=True)
        elif input_axis == 'Z' and target_axis == 'Y':
            # Rx(90)
            rot_convert = R.from_euler('x', 90, degrees=True)
        else:
            logging.warning("Unsupported axis conversion")
            return trans, rotations
            
        # Apply to translation
        trans_out = rot_convert.apply(trans)
        
        # Apply to root rotation (index 0)
        # global_rot_new = rot_convert * global_rot_old
        r_root = R.from_quat(rotations[:, 0, :])
        r_root_new = rot_convert * r_root
        
        rotations_out = rotations.copy()
        rotations_out[:, 0, :] = r_root_new.as_quat()
        
        return trans_out, rotations_out

    def get_limb_physics_data(self):
        return self.limb_data

    def get_joint_limits(self):
        """
        Returns approximate biomechanical limits for the first 22 joints.
        Values are in RADIANS.
        Format: {joint_name: [[min_x, max_x], [min_y, max_y], [min_z, max_z]]}
        
        Note: SMPL local coordinate systems usually have the primary rotation axis 
        aligned such that flexion/extension is the dominant component. 
        These are rough approximations useful for constraints.
        """
        d2r = np.deg2rad
        
        # Ranges: [min, max]
        # Common ranges (approximate)
        # 0: pelvis (root) - usually unlimited (free joint)
        # 1/2: L/R hips - ball joints, large range
        # 4/5: L/R knees - hinge, primarily flexion (0 to ~140/150)
        # 1/2/3/6/9... spines - limited ball joints
        
        limits = {}
        
        # Default loose limit for unspecified/ball joints
        full_range = [-np.pi, np.pi]
        limited_ball = [d2r(-45), d2r(45)]
        
        for name in self.joint_names[:self.target_joint_count]:
            # Default to full range for root, or reasonable limits for others
            if name == 'pelvis':
                 limits[name] = [full_range, full_range, full_range]
                 continue
            
            x_lim = limited_ball 
            y_lim = limited_ball 
            z_lim = limited_ball
            
            if 'knee' in name:
                # Knees are hinges. In SMPL, X-axis is usually flexion.
                # Flexion 0 to 150 deg approx.
                # Allow small range for twist/abduction (laxity)
                x_lim = [d2r(0), d2r(150)] 
                y_lim = [d2r(-10), d2r(10)]
                z_lim = [d2r(-10), d2r(10)]
                
            elif 'elbow' in name:
                # Elbows: Hinge. Flexion 0 to 150.
                # Warning: Check SMPL axis. Often Z or Y is main axis? 
                # Assuming X is primary for consistency or symmetric to knee in many rigs, 
                # but SMPL template is T-pose.
                # Let's assume standard SMPL: 
                # Be careful, we will provide broad 'primary Hinge' limits.
                # User asked for 'approximate', so we establish the pattern.
                 x_lim = [d2r(-150), d2r(0)] # or 0 to 150 depending on sign convention
                 # Actually, usually better to provide symmetric large range if unsure of sign
                 # or verify. For now, we'll use a specific range but warn.
                 # Let's use 0 to 160 magnitude.
                 x_lim = [d2r(-160), d2r(160)] 
                 y_lim = [d2r(-20), d2r(20)]
                 z_lim = [d2r(-20), d2r(20)]
                 
            elif 'hip' in name:
                # Ball joint, wide range
                x_lim = [d2r(-120), d2r(30)] # Flexion/Ext
                y_lim = [d2r(-50), d2r(50)]  # Abduction
                z_lim = [d2r(-50), d2r(50)]  # Rotation
                
            elif 'shoulder' in name:
                x_lim = [d2r(-180), d2r(180)] # Huge range
                y_lim = [d2r(-90), d2r(90)]
                z_lim = [d2r(-90), d2r(90)]

            elif 'spine' in name:
                # Stiffer
                stiff = [d2r(-30), d2r(30)]
                x_lim = stiff
                y_lim = stiff
                z_lim = stiff
                
            limits[name] = [x_lim, y_lim, z_lim]
            
        return limits
