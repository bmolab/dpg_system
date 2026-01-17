import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import pickle
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
    enable_passive_limits: bool = False
    
    # --- Filtering / Signal Processing ---
    enable_one_euro_filter: bool = True
    filter_min_cutoff: float = 1.0
    filter_beta: float = 0.0
    
    # --- Spike Detection / Clamping ---
    spike_threshold: float = 0.0 # Output Output Hysteresis
    input_spike_threshold: float = 0.0 # Teleport Detection
    jerk_threshold: float = 0.0 # Jerk-based spike
    
    # --- Floor / Environment ---
    floor_enable: bool = False
    floor_height: float = 0.0
    floor_tolerance: float = 0.15
    heel_toe_bias: float = 0.0

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
            'neck': {'limit': np.radians(30), 'k': 20.0}, 
            'spine': {'limit': np.radians(30), 'k': 50.0}, 
            
            # Collar: Asymmetric Hinge limits for structural support
            # Supporting 'sag' while allowing shrugs/rotation.
            'left_collar': {'type': 'hinge', 'axis': 2, 'min': -0.2, 'max': 0.8, 'k': 100.0}, 
            'right_collar': {'type': 'hinge', 'axis': 2, 'min': -0.2, 'max': 0.8, 'k': 100.0},
            
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
        self.prev_contact_mask = None
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
            # 27: R_Fingertip (Child of 37/R_Hand)
            parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 10, 11, 22, 37]
            self.parents = parents # Persist for RNE Physics
            
            # Note: We need 28 offsets now
            model_offsets = np.zeros((28, 3))
            
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

            # --- Virtual Extensions (Indices 24-27) ---
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
                
            # If SMPL-H, Right offset is at 23 (but using 37 data).
            r_knuckle_vec = model_offsets[23] 
            if np.linalg.norm(r_knuckle_vec) > 1e-4:
                r_dir = r_knuckle_vec / np.linalg.norm(r_knuckle_vec)
            else:
                r_dir = np.array([-1.0, 0.0, 0.0])
                
            finger_len = 0.08 # Approx 8cm from Knuckle to Tip
            
            model_offsets[26] = l_dir * finger_len
            model_offsets[27] = r_dir * finger_len
            
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

    def _compute_probabilistic_contacts(self, world_pos, pose_data_aa, options):
        """
        Estimates contact probability P(C) for each joint [0..1].
        P(C) = P_prox * P_kin * P_load * P_geo
        """
        if not options.floor_enable:
             return np.zeros((world_pos.shape[0], world_pos.shape[1]), dtype=np.float32)
             
        F = world_pos.shape[0]
        J = world_pos.shape[1] # 24 or 28
        floor_height = options.floor_height
        dt = options.dt
        # Use tracked internal dimension
        y_dim = getattr(self, 'internal_y_dim', 1) # Internal physics is always Y-up
        
        # State Initialization / Resizing
        if not hasattr(self, 'prob_contact_probs') or self.prob_contact_probs is None or self.prob_contact_probs.shape[0] != J:
             self.prob_contact_probs = np.zeros(J, dtype=np.float32)
             
        if not hasattr(self, 'prob_smoothed_vel') or self.prob_smoothed_vel is None or self.prob_smoothed_vel.shape[0] != J:
             # Explicitly (J, 3) for F=1 state
             self.prob_smoothed_vel = np.zeros((J, 3), dtype=np.float32)
             # Also reset prev_world_pos if size changes to avoid mismatch
             self.prob_prev_world_pos = None

        # Velocities
        if not hasattr(self, 'prob_prev_world_pos') or self.prob_prev_world_pos is None:
             lin_vel = np.zeros_like(world_pos)
             self.prob_prev_world_pos = world_pos[0].copy() if world_pos.ndim == 3 else world_pos.copy()
        elif F > 1:
             # Batch mode approximation
             lin_vel = np.zeros_like(world_pos)
             lin_vel[1:] = (world_pos[1:] - world_pos[:-1]) / dt
             self.prob_prev_world_pos = world_pos[-1].copy()
        else:
             lin_vel = (world_pos - self.prob_prev_world_pos) / dt
             self.prob_prev_world_pos = world_pos.copy()
             
        # Temporal Smoothing for Velocity (Trend Analysis)
        # alpha_v = 0.3 means history dominates (0.7).
        # Helps filter out inter-frame jitter.
        alpha_v = 0.3
        
        if F == 1:
             # lin_vel is (1, J, 3). We want (J, 3) for storage.
             current_vel = lin_vel[0] # (J, 3)
             self.prob_smoothed_vel = self.prob_smoothed_vel * (1.0 - alpha_v) + current_vel * alpha_v
             # Broadcast back to (1, J, 3)
             lin_vel = self.prob_smoothed_vel[np.newaxis, ...] # (1, J, 3)
        else:
             # Batch mode: No state smoothing possible across frames in one call easily
             pass

             
        vel_y = lin_vel[..., y_dim]
        vel_h_vec = lin_vel.copy()
        vel_h_vec[..., y_dim] = 0.0
        vel_h = np.linalg.norm(vel_h_vec, axis=-1)
        
        # --- Factors ---
        frame_probs = np.zeros((F, J), dtype=np.float32)
        
        # Tips Access
        tips_available = hasattr(self, 'temp_tips') and self.temp_tips is not None
        
        for f in range(F):
            # 1. ZMP Load Prior (Balance)
            # Use current CoM/ZMP state
            # Slice world_pos to 24 for CoM calc if necessary, but _compute_full_body_com usually handles slicing internal?
            # Actually _compute_full_body_com calculates mass weighted average.
            # If we pass 28 joints to it, and masses are 24, it might crash there too?
            # Let's check com calc safely.
            # But here we use 'self.current_com' which is typically pre-calced (on 24 e.g.).
            if self.current_com is not None:
                com = self.current_com # (F, 3) 
                # (Assuming F=1 or current_com matches F)
                if com.ndim == 2: c = com[f]
                else: c = com
            else:
            # Fallback: slice to 24 for safety
                c = self._compute_full_body_com(world_pos[f:f+1, :24, :])[0]

            # --- Dynamic ZMP Calculation ---
            # ZMP = CoM_proj - (h/g) * a_horz
            # Requires acceleration tracking.

            # 1. State Initialization
            if not hasattr(self, 'prob_prev_com') or self.prob_prev_com is None:
                self.prob_prev_com = c.copy()
                self.prob_prev_com_vel = np.zeros_like(c)
                self.prob_prev_com_acc = np.zeros_like(c)
                com_vel = np.zeros_like(c)
                com_acc = np.zeros_like(c)
            else:
                # 2. Kinematics
                if F > 1:
                     # Batch mode approximation (less accurate per frame if gap)
                     # For now, treat as step
                     com_vel = (c - self.prob_prev_com) / dt
                else:
                     com_vel = (c - self.prob_prev_com) / dt
                     
                # 3. Acceleration & Smoothing (EMA)
                # Raw Acc
                # Handle startup first frame
                raw_acc = (com_vel - self.prob_prev_com_vel) / dt
                
                # Heavy smoothing to reject noise (Alpha ~ 0.05 is too slow for Jumps)
                # Jumps happen in 30-60 frames. Alpha 0.05 lags by 20 frames.
                # Increase to 0.5 (Responsive)
                alpha_acc = 0.5
                com_acc = self.prob_prev_com_acc * (1.0 - alpha_acc) + raw_acc * alpha_acc
                
                # Update State
                self.prob_prev_com = c.copy()
                self.prob_prev_com_vel = com_vel.copy()
                self.prob_prev_com_acc = com_acc.copy()
                
            # 4. Calculate Offset
            plane_dims = [d for d in [0, 1, 2] if d != y_dim]
            
            # h = CoM Height relative to floor
            com_h = c[y_dim] - floor_height
            g = 9.81
            
            # Offset = - (h/g) * a
            # We only care about horizontal acceleration for ZMP shift
            acc_horz = com_acc[plane_dims] # 2D
            zmp_offset = - (com_h / g) * acc_horz
            
            # Static Projection
            p_zmp_static = c[plane_dims]
            p_zmp = p_zmp_static + zmp_offset
            
            # Store 3D ZMP for Output (Projected to Floor)
            if not hasattr(self, 'current_zmp') or self.current_zmp is None:
                 # Use 'c' which is guaranteed to be the CoM
                 if c.ndim == 1: shape = (3,)
                 elif c.ndim == 2: shape = (c.shape[0], 3)
                 else: shape = (3,)
                 self.current_zmp = np.zeros(shape)
            
            # Pack back into 3D (assuming floor height Y)
            if self.current_zmp.ndim == 1:
                self.current_zmp[plane_dims] = p_zmp
                self.current_zmp[y_dim] = floor_height
            else:
                self.current_zmp[f, plane_dims] = p_zmp
                self.current_zmp[f, y_dim] = floor_height
            
            # --- Stability Heuristic (Feet Balance) ---
            # Check how well feet support the CoM. If balanced, dampen hand sensitivity.
            # If failing (unstable), increase hand sensitivity (allow catch).
            feet_indices = [7, 8, 10, 11]
            min_foot_dist = 100.0
            
            # Helper to get Hz Pos of any joint
            def get_hz_pos(idx):
                 if tips_available and idx in self.temp_tips: p = self.temp_tips[idx][f]
                 else: p = world_pos[f, idx]
                 return p[plane_dims]
                 
            for f_idx in feet_indices:
                d = np.linalg.norm(get_hz_pos(f_idx) - p_zmp)
                if d < min_foot_dist: min_foot_dist = d
                
            # Score 1.0 (Stable) -> 0.0 (Unstable). Threshold ~0.25m.
            stability_score = np.exp(- (min_foot_dist / 0.25)**2)
            
            # Hand Sigma: 0.05 (Stable) -> 0.10 (Unstable)
            sigma_hand_dynamic = 0.05 + 0.05 * (1.0 - stability_score)
            
            # 2. Joint Loop
            current_p = np.zeros(J)
            for j in range(J):
                # Pos
                if tips_available and j in self.temp_tips:
                    pos = self.temp_tips[j][f]
                else:
                    pos = world_pos[f, j]
                
                h = pos[y_dim] - floor_height
                
                # Apply Virtual Heel Offset if Ankle (7, 8) or Wrist (20, 21)
                # Note: Hands (22, 23) do NOT get bias, allowing fingertips to touch accurately.
                if j in [7, 8, 20, 21]:
                    h -= options.heel_toe_bias
                
                vy = vel_y[f, j] if vel_y.ndim > 1 else vel_y[j]
                vh = vel_h[f, j] if vel_h.ndim > 1 else vel_h[j]
                
                # A. Proximity P(h)
                # Gaussian falloff
                # Default Sigma 0.10m
                sigma_h = 0.10
                
                # Use Dynamic Sigma for Hands
                if j in [20, 21, 22, 23]:
                    sigma_h = sigma_hand_dynamic
                
                if h <= 0:
                    p_prox = 1.0
                else:
                    p_prox = np.exp(-0.5 * (h / sigma_h)**2)
                
                # Clip for large distances (optimization)
                if h > 0.20: p_prox = 0.0
                
                # B. Kinematics P(v)
                # Issue: During swing (arc), Vy or Vh might dip individually while V_total is high.
                # Component-wise checks allow False Positives.
                # Solution: Use Velocity Magnitude Gating.
                
                v_total_mag = np.sqrt(vy**2 + vh**2)
                
                v_safe = 0.20 # Relaxed from 0.15
                v_cut = 0.50  # Relaxed from 0.30
                
                # Note: We apply Velocity Gating EVEN if Underground (h <= 0).
                # Why? Mocap errors or interpolation sometimes cause feet to "clip" 
                # through the floor during a fast swing.
                # If we forced Contact=1.0 just because h<0, we'd get a false positive slam.
                # Real contact (standing/pushing) implies low velocity relative to the planted foot.
                
                if v_total_mag < v_safe:
                    p_vel = 1.0
                elif v_total_mag > v_cut:
                    p_vel = 0.0
                else:
                    # Smooth Decay
                    t = (v_total_mag - v_safe) / (v_cut - v_safe)
                    p_vel = np.exp(-5.0 * t**2) # Sharp decay
                
                # Asymmetric Lift Logic with Hysteresis
                # User Request: "If contact has not been made yet, and CoM is rising and potential contact point is rising, then contact is not established."
                # "If contact has already been establish and is ongoing, minor upward movement should not break contact."
                
                if not hasattr(self, 'prob_prev_in_contact') or self.prob_prev_in_contact is None:
                     self.prob_prev_in_contact = np.zeros(J, dtype=bool)
                     
                was_contact = self.prob_prev_in_contact[j]
                
                # Thresholds
                # Exit (Breaking Contact): Loose. Allow jitter up to 0.20 m/s.
                thresh_exit = 0.20 
                # Entry (Making Contact): Strict. Must be stable (0.02 m/s).
                thresh_entry = 0.02
                
                # Logic
                failed_gate = False
                
                if was_contact:
                     # In Contact: Only break if lifting distinctly
                     if vy > thresh_exit:
                          failed_gate = True
                else:
                     # Not In Contact: Strict Entry
                     # 1. Foot must be stable (~0 velocity)
                     if vy > thresh_entry:
                          failed_gate = True
                          
                     # 2. Global Rise Check (Prevent grabbing while launching)
                     # If CoM is launching UP, and Foot is loose, deny contact.
                     # com_vel is computed above.
                     if com_vel[y_dim] > 0.1 and vy > 0.01: # Even slight foot rise during CoM launch is suspicious
                          failed_gate = True
                          
                # 3. Acceleration Gate (Apex / Freefall)
                # If body is accelerating downwards at near-gravity (Freefall), and velocity is low (Apex),
                # We cannot be supported. Spurious contact usually happens here because Height is low and Vel is 0.
                # Adjusted threshold to -2.0 based on file analysis (Data shows ~ -2.5m/s^2 at apex).
                if com_acc[y_dim] < -2.0 and v_total_mag < 0.5:
                     failed_gate = True
                          
                if failed_gate:
                    p_vel = 0.0
                    
                p_contact_j = p_prox * p_vel
                
                # --- Contact Debounce (Confirmation) ---
                # User Request: "Need at least two consecutive frames of contact to confirm"
                if not hasattr(self, 'prob_contact_duration') or self.prob_contact_duration is None:
                     self.prob_contact_duration = np.zeros(J, dtype=int)
                     
                if p_contact_j > 0.5:
                     self.prob_contact_duration[j] += 1
                else:
                     self.prob_contact_duration[j] = 0
                     
                # Debounce: Suppress if duration < 2
                if self.prob_contact_duration[j] < 2:
                     p_contact_j = 0.0
                
                # Update State for Next Frame
                # Using 0.5 as binary threshold for "Established Contact" state
                # Note: This creates a feedback loop across frames (Hysteresis state).
                is_contact = (p_contact_j > 0.5)
                self.prob_prev_in_contact[j] = is_contact
                
                # Use p_contact_j for subsequent logic assignments...
                # Note: original code assigned p_vy and p_slip separately.
                # We unify them into p_vel.
                
                # Combine
                current_p[j] = p_contact_j
                
                frame_probs[f, j] = p_contact_j
            
            # 3. Heel/Toe Geometry Override (Specific to Feet)
            # Apply the 5cm band logic as a Probability Mask
            for (heel, toe) in [(7, 10), (8, 11)]:
                 # Get P_raw
                 p_h = current_p[heel]
                 p_t = current_p[toe]
                 
                 # Heights
                 if tips_available and heel in self.temp_tips: h_h = self.temp_tips[heel][f, y_dim]
                 else: h_h = world_pos[f, heel, y_dim]
                 if tips_available and toe in self.temp_tips: h_t = self.temp_tips[toe][f, y_dim]
                 else: h_t = world_pos[f, toe, y_dim]
                 
                 # Apply Virtual Heel Offset
                 h_h -= options.heel_toe_bias
                 
                 diff = h_h - h_t
                 band = 0.10 # Relaxed to 10cm to prevent dropout on pitched feet
                 
                 # Mask [0..1]
                 mask_heel = 1.0
                 mask_toe = 1.0
                 
                 if diff > band: # Heel High
                      mask_heel = 0.0
                 elif diff < -band: # Toe High
                      mask_toe = 0.0
                 else:
                      # Inside the band (-10cm to +10cm), consider both valid.
                      # This solves the issue where Ankle depth breaks Toe contact.
                      pass
                      
                 current_p[heel] *= mask_heel
                 current_p[toe] *= mask_toe
                 
            # 4. Temporal Smoothing (Bayesian Update)
            # P_new = alpha * P_obs + (1-alpha) * P_old
            if f == 0:
                 self.prob_contact_probs = current_p # Initialize
            else:
                 # Alpha = Learning Rate. 
                 # 0.5 = Faster response (User requested less "persistence")
                 alpha = 0.5
                 self.prob_contact_probs = self.prob_contact_probs * (1.0 - alpha) + current_p * alpha
            frame_probs[f] = self.prob_contact_probs
            
            # --- 5. Contact Pressure (Load Distribution) ---
            # Distinguish "Touching" from "Load Bearing" using ZMP proximity.
            # Pressure ~ P_contact * (1 / distance_to_zmp^2)
            if not hasattr(self, 'contact_pressure') or self.contact_pressure is None or self.contact_pressure.shape != frame_probs.shape:
                 self.contact_pressure = np.zeros_like(frame_probs)
            
            # Grounded Set: Ankles(7,8), Toes(10,11 - Wait 10,11 are Feet, 24,25 are Toes)
            # We use all joints that have contact > 0
            # Get ZMP for this frame
            zmp_f = self.current_zmp[f] if self.current_zmp.ndim > 1 else self.current_zmp # (3,)
            zmp_hz = zmp_f[plane_dims] # 2D
            
            # Accumulate weights
            total_weight = 0.0
            weights = np.zeros(J)
            
            # Compute Beta (Lying Factor)
            # If Indices [0, 3, 6, 9, 12, 15] have significant contact sum?
            core_contact_sum = np.sum(self.prob_contact_probs[[0, 3, 6, 9, 12, 15]])
            # If Pelvis(0) or Spine(3) is down, beta -> 1.0
            beta = np.clip(core_contact_sum, 0.0, 1.0)
            
            # Refined Weight calculation with Beta
            total_weight = 0.0
            for j in range(J):
                p = self.prob_contact_probs[j]
                if p < 0.01: continue
                
                # Get Joint Horizontal Pos
                if tips_available and j in self.temp_tips: 
                    pos = self.temp_tips[j][f]
                else: 
                    pos = world_pos[f, j]
                pos_hz = pos[plane_dims]
                dist_sq = np.sum((pos_hz - zmp_hz)**2)
                
                w_idw = p / (dist_sq + 0.005)
                # w_mass = p * 50.0 
                # ISSUE: p is very high for Pelvis/Spine (Deep penetration), low for Knees.
                # If we use p directly, Pelvis hogs all support. Knees get none.
                # Result: Hips lift legs (High Effort).
                # FIX: Uniform Distribution (or close to it) ensures all touching parts support themselves.
                w_mass = 10.0 if p > 0.01 else 0.0
                
                # Blend
                w_final = w_idw * (1.0 - beta) + w_mass * beta
                
                weights[j] = w_final
                total_weight += w_final
            
            # Normalize
            # Normalize
            if total_weight > 1e-6:
                self.contact_pressure[f] = weights / total_weight
                
                # --- 6. Dynamic GRF Scaling ---
                # Scale Pressure by Vertical Acceleration (F = m * (g + a))
                # If jumping (pushing off), a_y > 0 -> Pressure > 1.0 (High Effort)
                # If falling (airborne), a_y ~ -g -> Pressure ~ 0.0
                # If landing, a_y >> 0 -> Pressure >> 1.0 (Impact)
                
                acc_vert = com_acc[y_dim]
                # Factor = (g + a) / g
                # Clip lower bound to 0 (can't have negative pressure)
                grf_factor = max(0.0, (g + acc_vert) / g)
                
                self.contact_pressure[f] *= grf_factor
                
            else:
                self.contact_pressure[f] = 0.0 # No pressure if no contact
                 
        return frame_probs

    def _compute_max_torque_profile(self):
        """
        Returns a dictionary of approximate theoretical maximum isometric torque (N-m) 
        per joint dimension, for "Effort" normalization.
        
        Based very roughly on biomechanics literature (e.g. Chaffin).
        This is a simplification (assumes healthy adult).
        """
        # Conservative estimates
        # Format: [Flexion/Ext, Abd/Add, Rot] (roughly X, Y, Z in local)
        # But we calculate torque magnitude. So we need a scalar 'Max Torque Magnitude'.
        
        # Max Torque (Scalar N-m)
        max_t = {
            'pelvis': 500.0, # Massive core capability
            'spine': 400.0,
            'hip': 300.0,
            'knee': 250.0, # Quadriceps are strong (ext)
            'ankle': 150.0, # Adjusted for Weight Bearing (Plantarflexion is strong)
            'foot': 40.0,
            'neck': 100.0, # Increased (50->100) to lower gravity effort
            'head': 30.0,  # Increased (15->30) to lower gravity effort
            'collar': 500.0, # High capability (structural support) to reduce resting effort
            'shoulder': 120.0,
            'elbow': 80.0,
            'wrist': 20.0,
            'hand': 10.0
        }
        
        # Adjust for gender roughly? 
        # Females approx 60-70% of males in upper body, 70-80% lower.
        # Neutral defaults to male-ish or mid-range.
        scale = 1.0
        if self.gender == 'female':
            scale = 0.7
            
        return {k: v * scale for k, v in max_t.items()}

    def _compute_max_torque_array(self):
        """
        Convert dictionary profile to per-joint max torque array.
        Returns:
            arr (np.array): (24,) Max torque for each joint.
        """
        arr = np.zeros(24)
        for i in range(24):
            name = self.joint_names[i]
            max_t = 100.0 # Default
            
            # Priority Logic (matches original process_frame)
            if 'pelvis' in name: max_t = self.max_torques.get('pelvis', 500.0)
            elif 'hip' in name: max_t = self.max_torques.get('hip', 300.0)
            elif 'knee' in name: max_t = self.max_torques.get('knee', 250.0)
            elif 'ankle' in name: max_t = self.max_torques.get('ankle', 40.0)
            elif 'foot' in name: max_t = self.max_torques.get('foot', 30.0)
            elif 'spine' in name: max_t = self.max_torques.get('spine', 400.0)
            elif 'neck' in name: max_t = self.max_torques.get('neck', 50.0)
            elif 'head' in name: max_t = self.max_torques.get('head', 15.0)
            elif 'collar' in name: max_t = self.max_torques.get('collar', 500.0)
            elif 'shoulder' in name: max_t = self.max_torques.get('shoulder', 120.0)
            elif 'elbow' in name: max_t = self.max_torques.get('elbow', 80.0)
            elif 'wrist' in name: max_t = self.max_torques.get('wrist', 20.0)
            elif 'hand' in name: max_t = self.max_torques.get('hand', 10.0)
            
            arr[i] = max_t
        return arr

    def set_max_torque(self, joint_name_filter, value):
        """
        Manually updates max torque for joints matching the filter.
        Args:
            joint_name_filter (str): Substring to match (e.g., 'neck', 'ankle').
            value (float): New max torque value in N-m.
        """
        count = 0
        # 1. Update Profile Dict
        for k in self.max_torques:
            if joint_name_filter in k:
                self.max_torques[k] = float(value)
                count += 1
        
        # 2. Update Cached Array
        if hasattr(self, 'max_torque_array'):
            self.max_torque_array = self._compute_max_torque_array()
            
        return count

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


    def _get_hierarchy(self):
        """Defines parent-child relationships for SMPL 24 joints."""
        # Parent indices for standard SMPL
        # -1 means root
        parents = [-1,  0,  0,  0,  1,  2, 
                    3,  4,  5,  6,  7,  8, 
                    9,  9,  9, 12, 13, 14, # Shoulders (16,17) parented to Collars (13,14)
                   16, 17, 18, 19, 20, 21,
                   10, 11, 22, 23] # Virtual: L_Toe, R_Toe, L_Tip, R_Tip
                   
        # Note: Collars (13, 14) are still children of Spine3 (9).
        # But Shoulders (16, 17) now bypass them physically for mass load calculation.
        if not hasattr(self, '_cached_parents'):
            self._cached_parents = parents
            
        return self._cached_parents

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
            
            if m <= 0: continue

            # Find child node to define rod direction
            child_nodes = [c for c, p in enumerate(parents) if p == idx]
            
            com_pos = world_positions[..., idx, :] # Default fallback
            
            if len(child_nodes) > 0:
                # Take average of children as end point
                end_pos = np.mean([world_positions[..., c, :] for c in child_nodes], axis=0)
                com_pos = (world_positions[..., idx, :] + end_pos) * 0.5
            else:
                pass

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
        
        if not options.add_gravity:
            return t_grav_vecs
            
        # Optimization: Gravity only applies to the main 24 body joints.
        # Virtual end-effectors (24-27) have zero mass and no torque.
        # Slice world_pos to ensure shape consistency with mass arrays.
        world_pos_full = world_pos
        world_pos = world_pos_full[:, :24, :]
            
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
        node_masses = np.zeros((F, 24))
        node_weighted_com = np.zeros((F, 24, 3))
        
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
        
        # --- Apparent Gravity (Airborne Physics) ---
        # If in freefall, body accelerates with gravity (a ~ -g). Effective weight is 0.
        # If landing, body decelerates (a > 0). Effective weight is > mg.
        # FIX: Use ROOT Position (System Frame) instead of CoM (which includes internal Dynamics).
        # Using CoM causes internal limb swings to trigger global gravity scaling.
        
        dt = options.dt
        current_root = world_pos[:, 0, :] # (F, 3)
        
        if not hasattr(self, 'prev_root_gravity'):
             self.prev_root_gravity = current_root.copy()
             self.prev_vel_gravity = np.zeros_like(current_root)
             root_acc = np.zeros_like(current_root)
        else:
             # Vel
             # Handle Teleport (Resets)
             dist = np.linalg.norm(current_root - self.prev_root_gravity, axis=-1)
             if np.any(dist > 2.0): # 2 meters per frame jump? Reset.
                  self.prev_root_gravity = current_root.copy()
                  self.prev_vel_gravity = np.zeros_like(current_root)
                  root_acc = np.zeros_like(current_root)
             else:
                  current_vel = (current_root - self.prev_root_gravity) / dt
                  # Acc
                  root_acc = (current_vel - self.prev_vel_gravity) / dt
                  
                  # Update History
                  self.prev_root_gravity = current_root.copy()
                  self.prev_vel_gravity = current_vel
             
        # Calculate Scaling Factor: (g + a_y) / g
        y_dim = getattr(self, 'internal_y_dim', 1)
        g_mag = 9.81
        a_y = root_acc[..., y_dim]
        
        # G_factor needs to match shape of t_grav_vecs for broadcasting (F, 1, 1) or (F, 24, 3)? ((F,))
        g_factor = np.clip((g_mag + a_y) / g_mag, 0.0, 5.0) # Clamp 0 to 5g
        
        # Store for debug
        self.g_factor = g_factor[0] if len(g_factor) > 0 else 1.0
        
        # Apply to Gravity Torque
        # (F,) -> (F, 1, 1)
        g_factor_reshaped = g_factor[:, np.newaxis, np.newaxis]
        t_grav_vecs *= g_factor_reshaped
        
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
             # 1. Compute GRF Vectors at each Joint
             J = world_pos.shape[1] # Define J
             
             m_total = self.current_total_mass # (F,)
             f_total_weight = m_total[:, np.newaxis] * g_vec[np.newaxis, :] # (F, 3)
             
             pressures = self.contact_pressure # (F, J)
             if pressures.ndim == 1: pressures = pressures[np.newaxis, :]
             
             # GRF Force Vectors per Joint (Point Forces)
             # (F, J, 3)
             f_grf_vecs = pressures[..., np.newaxis] * f_total_weight[:, np.newaxis, :]
             
             # 2. Backward Pass (Accumulate GRF Moments up the tree)
             # We need to know:
             # - Cumulative Supported Force below joint j (F_cum)
             # - Cumulative Moment of those forces about joint j (M_cum)
             
             # Initialize
             f_cum_grf = f_grf_vecs.copy()
             m_cum_grf = np.zeros_like(f_grf_vecs) # Moments about the joint itself
             
             # Parents array for traversal
             parents = self.parents 
             
             # Backward Pass (Leaf -> Root)
             # Iterate J-1 down to 1 (0 is root)
             for j in range(J-1, 0, -1):
                 parent = parents[j]
                 
                 # Propagate TO Parent
                 # Force propagates directly
                 f_j = f_cum_grf[:, j, :]
                 f_cum_grf[:, parent, :] += f_j
                 
                 # Moment propagates:
                 # M_parent += M_joint + (Pos_joint - Pos_parent) x F_joint
                 
                 # Vector r from Parent to Joint
                 r = world_pos[:, j, :] - world_pos[:, parent, :]
                 
                 # Additional Moment from Force at j acting on parent lever arm
                 dt_force = np.cross(r, f_j)
                 
                 # Add child's internal moment + lever moment
                 m_cum_grf[:, parent, :] += m_cum_grf[:, j, :] + dt_force
                 
             # 3. Apply to Gravity Torques
             # The existing 't_grav_vecs' calculates the moment of the SUBTREE MASS about the joint.
             # 'm_cum_grf' is now the moment of the SUBTREE GROUND FORCES about the joint.
             # Net Load = Gravity Load - Support Load
             
             # Note on Sign Convention:
             # Gravity Torque (Lift) usually attempts to ROTATE the limb DOWN.
             # Support Torque (GRF) attempts to ROTATE the limb UP (Support).
             # If completely supported (Lying), they cancel.
             # If Standing, GRF >> Leg Weight -> Negative Torque (Support Load).
             
             # Apply
             # Roll Attenuation (Sagittal Dominance) still valuable for perceptual "Effort"?
             # RNE is physically correct, but "Effort" is subjective.
             # Let's trust pure physics first. If Standing:
             # M_grav (Leg) is small. M_grf (Whole Body) is huge.
             # Net = Huge Support Torque. This is correct (Squatting is hard).
             # But "Standing Straight" should be low effort?
             # If aligned, Moment arm is small. So M_grf is small?
             # Yes. If joints are stacked, Cross Product is small.
             # Yes. If joints are stacked, Cross Product is small.
             # RNE handles alignment automatically!
             
             t_grav_vecs -= m_cum_grf[:, :t_grav_vecs.shape[1], :]
        
        return t_grav_vecs

    def _compute_floor_contacts(self, world_pos, pose_data_aa, options):
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
            world_pos (F, 24, 3): Global positions.
            global_rots (List[R]): Global rotation objects per joint (relative to world).
            tips (dict): Index -> Position (F, 3) for end effectors (Feet, Hands).
        """
        F = trans_data.shape[0]
        # Extending to 28 output joints (24 original + 4 virtual)
        world_pos = np.zeros((F, 28, 3))
        
        # Optimization: Create ONE Rotation object for all joints at once to speed up init
        # We need frames for a specific joint to be contiguous for easy slicing later.
        # quats input is (F, 24, 4), so we transpose to (24, F, 4)
        quats_perm = quats.transpose(1, 0, 2).reshape(-1, 4) # (24*F, 4)
        all_rots = R.from_quat(quats_perm)
        
        # We store slices for each joint to avoid re-slicing (though slicing R is cheapish)
        # Actually we just slice on the fly to keep code clean.
        
        global_rots = [None] * 28
        parents = self._get_hierarchy()

        for i in range(28):
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
                
                # Calculate Offset (using 28-element skeleton_offsets)
                local_pos = self.skeleton_offsets[i]

                # Apply Rotation (Parent's Global Rotation) to the Local Offset
                offset_vec = global_rots[parent].apply(local_pos)
                world_pos[:, i, :] = world_pos[:, parent, :] + offset_vec
                
        # --- Tip Projection for Contact Detection ---
        tips = {} # index -> pos (F, 3)
        
        # Use Virtual End-Effectors (Indices 24-27) for accurate contact
        # L_Foot (10) -> L_Toe (24)
        # R_Foot (11) -> R_Toe (25)
        # L_Hand (22) -> L_Tip (26)
        # R_Hand (23) -> R_Tip (27)
        
        # Verify we have enough joints (28)
        if world_pos.shape[1] >= 28:
            tips[10] = world_pos[:, 24, :]
            tips[11] = world_pos[:, 25, :]
            tips[22] = world_pos[:, 26, :]
            tips[23] = world_pos[:, 27, :]
        else:
             # Fallback if virtual joints missing (should not happen with new logic)
             pass
                  
        # Store tips and world_pos for later contact logic/debug
        self.temp_tips = tips 
        self.last_world_pos = world_pos 
        
        return world_pos, global_rots, tips

    def _detect_input_spikes(self, pose_data_aa, world_pos, options):
        """
        Detects anomalies in input data (Teleport, Glitch, Wobble).
        Updates internal state (prev_pose_raw, etc.).
        Returns:
            input_spike_detected (bool): True if spike detected.
        """
        input_spike_detected = False
        
        # Only run spike detection if streaming (frame by frame)
        if pose_data_aa.shape[0] == 1:
             # 1. Teleport Detection (Large Step)
             if options.input_spike_threshold > 0.0 and self.prev_pose_raw is not None:
                 diff = np.abs(pose_data_aa - self.prev_pose_raw)
                 diff_norm = np.linalg.norm(diff, axis=2)
                 
                 # Scale Threshold by dt (Normalize to 60 FPS)
                 # Unit is effectively "Degrees per Frame @ 60 FPS"
                 scale_factor = options.dt * 60.0
                 th_rad = np.radians(options.input_spike_threshold) * scale_factor
                 
                 if np.max(diff_norm) > th_rad:
                     if not self.last_frame_spiked:
                         input_spike_detected = True
                         self.last_frame_spiked = True # Guard ON
                     else:
                         self.last_frame_spiked = True # Sustain Guard
                 else:
                     self.last_frame_spiked = False
             
             # 2. Accel / Jerk Detection (Glitch/Wobble)
             # Always track state to support dynamic toggling
             if self.prev_pose_raw is not None:
                   # Lazy init for hot-reload
                   if not hasattr(self, 'prev_vel_aa_raw'): self.prev_vel_aa_raw = None
                   if not hasattr(self, 'prev_acc_raw'): self.prev_acc_raw = None
                   
                   if self.prev_vel_aa_raw is None:
                       self.prev_vel_aa_raw = np.zeros_like(pose_data_aa)
                   
                   # V_new
                   v_new = pose_data_aa - self.prev_pose_raw
                   
                   # A_new
                   acc_new = v_new - self.prev_vel_aa_raw
                   self.prev_vel_aa_raw = v_new
                   
                   # Check Thresholds
                   # Only Jerk Threshold remains
                   if options.jerk_threshold > 0.0:
                       # Jerk Check
                       if self.prev_acc_raw is not None:
                            jerk_vec = acc_new - self.prev_acc_raw
                            jerk_mag = np.linalg.norm(jerk_vec, axis=2)
                            jerk_th_rad = np.radians(options.jerk_threshold)
                            
                            mask_jerk = jerk_mag > jerk_th_rad
                            
                            y_dim = 2 if options.input_up_axis == 'Z' else 1
                            
                            # --- Floor Contact Logic ---
                            if options.floor_enable:
                                 y_vals = world_pos[..., y_dim] # (1, 24)
                                 
                                 on_floor = y_vals < (options.floor_height + options.floor_tolerance)
                                 mask_jerk = np.logical_and(mask_jerk, ~on_floor)
                            
                            
                            if np.any(mask_jerk):
                                 input_spike_detected = True
                                 
                   self.prev_acc_raw = acc_new
                  
             # Update Pose History
             self.prev_pose_raw = pose_data_aa.copy()
             
        else:
             # Batch Mode - Reset State
             pass
             
        return input_spike_detected


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

    def _compute_floor_support_torques(self, world_pos, contact_masses, tips, options):
        """
        Compute torque vectors from floor support (Anti-Gravity).
        T = r x F.
        F = mass * -g_vec. (Upward force).
        r = (Weighted_Pos / Mass) - Joint_Pos.
        """
        F = world_pos.shape[0]
        
        # Optimization: Slice to 24 body joints (ignore virtual tips)
        world_pos_full = world_pos
        world_pos = world_pos_full[:, :24, :]
        
        t_floor_vecs = np.zeros((F, self.target_joint_count, 3))
        
        # If no floor, return zero
        if not options.floor_enable:
            return t_floor_vecs
            
        # Determine Up Vector (Opposite to Gravity)
        # Gravity is Down. Support is Up. 
        # F_support = m * -g.
        g_mag = 9.81
        internal_up = getattr(self, 'internal_y_dim', 1)
        if internal_up == 2:
            g_vec = np.array([0.0, 0.0, -g_mag])
        else:
            g_vec = np.array([0.0, -g_mag, 0.0])
            
        support_accel = -g_vec # Upwards 9.81
        
        # 1. Accumulate Mass and Center of Pressure (Weighted Pos)
        # Initialize with Contact Mass at each node
        node_contact_mass = contact_masses.copy() # (F, 24)
        node_weighted_pos = np.zeros((F, 24, 3))
        
        # Calculate Weighted Positions
        for idx in range(24):
            # If Tip (Ankle/Foot/Hand), use Tip Position for better lever arm
            pos = world_pos[:, idx, :] # Default
            
            # Tip Override
            if idx in [7, 8, 10, 11, 22, 23]:
                 if tips is not None and idx in tips:
                      pos = tips[idx]
                      
            node_weighted_pos[:, idx, :] = pos * node_contact_mass[:, idx][:, np.newaxis]
            
        # 2. Bottom-Up Accumulation (Subtree)
        parents = self._get_hierarchy()
        subtree_contact_mass = node_contact_mass.copy()
        subtree_weighted_pos = node_weighted_pos.copy()
        
        for idx in range(23, 0, -1):
            parent = parents[idx]
            if parent >= 0:
                subtree_contact_mass[:, parent] += subtree_contact_mass[:, idx]
                subtree_weighted_pos[:, parent, :] += subtree_weighted_pos[:, idx, :]
                
        # 3. Compute Torque
        # T = r x F
        # r = CoP - Joint_Pos
        # CoP = Weighted_Pos / Total_Mass
        
        mask = subtree_contact_mass > 1e-6
        subtree_cop = np.zeros_like(world_pos)
        subtree_cop[mask] = subtree_weighted_pos[mask] / subtree_contact_mass[mask][:, np.newaxis]
        
        # If mass is 0, torque is 0.
        r_vecs = subtree_cop - world_pos
        
        # Force = Mass * Support_Accel
        f_support = subtree_contact_mass[:, :, np.newaxis] * support_accel[np.newaxis, np.newaxis, :]
        
        # Torque
        t_floor_vecs = np.cross(r_vecs, f_support)
        
        return t_floor_vecs

    def _compute_joint_torques(self, F, ang_acc, world_pos, parents, global_rots, pose_data_aa, contact_masses, tips, options):
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
            options (SMPLProcessingOptions): Usage: add_gravity, input_up_axis, enable_passive_limits.
            
        Returns:
            torques_vec (F, 24, 3): Net torque vectors (local/parent frame).
            inertias (F, 24): Effective inertia magnitude.
            efforts_net (F, 24): Net Effort (Normalized Torque).
            efforts_dyn (F, 24): Dynamic Effort.
            efforts_grav (F, 24): Gravity Effort.
        """
        # 2. Torque / Effort Calculation (Vectorized)
        torques_vec = np.zeros((F, self.target_joint_count, 3))
        
        inertias = np.zeros((F, self.target_joint_count))
        
        efforts_dyn = np.zeros((F, self.target_joint_count))
        efforts_grav = np.zeros((F, self.target_joint_count))
        efforts_net = np.zeros((F, self.target_joint_count))
        
        # Vectors
        t_dyn_vecs = np.zeros((F, self.target_joint_count, 3))
        
        # Pre-compute Gravity Torques
        t_grav_vecs = self._compute_gravity_torques(world_pos, options, global_rots, tips)
        
        # Pre-compute Floor Support Torques
        t_floor_vecs = self._compute_floor_support_torques(world_pos, contact_masses, tips, options)

        # Optimization: Pre-compute Inverses of Global Rotations
        # Many joints share parents (e.g. Pelvis->Hips, Spine->Collars), so computing inv() inside loop is redundant.
        global_rot_invs = [r.inv() for r in global_rots]

        for j in range(self.target_joint_count):
            name = self.joint_names[j]
            # Inertia (F,)
            I_eff = self._compute_subtree_inertia(j, world_pos, None, self.limb_data['lengths'], self.limb_data['masses'])
            inertias[:, j] = I_eff
            
            # Alpha (F, 3)
            alpha_vec = ang_acc[:, j, :]
            
            # Dynamic Torque (F, 3)
            torque_dyn = I_eff[:, np.newaxis] * alpha_vec
            t_dyn_vecs[:, j, :] = torque_dyn
            
            torque_grav = t_grav_vecs[:, j, :]
            
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
                
                # Dynamic is already Local
                t_dyn_local = torque_dyn
                
                # Net Local
                t_net_local = t_dyn_local - t_grav_local
                
            else:
                # Root: Parent is World (Identity)
                t_net_local = torque_dyn - torque_grav
                t_dyn_local = torque_dyn
                t_grav_local = torque_grav
            
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
            # Max torque for this joint
            t_max = self.max_torque_array[j]
            
            # Magnitudes of ACTIVE/Effective Torque
            m_net = np.linalg.norm(t_active_local, axis=1) # Active Net
            m_dyn = np.linalg.norm(t_dyn_local, axis=1) # Dynamic (Raw)
            m_grav = np.linalg.norm(t_grav_local, axis=1) # Gravity (Raw)
            
            efforts_net[:, j] = m_net / t_max
            efforts_dyn[:, j] = m_dyn / t_max
            efforts_grav[:, j] = m_grav / t_max
            
        return torques_vec, inertias, efforts_net, efforts_dyn, efforts_grav, t_dyn_vecs, t_grav_vecs, t_floor_vecs

    def _apply_output_spike_rejection(self, efforts_net, input_spike_detected, options):
        """
        Filters sudden spikes in effort output (Teleport prevention).
        Uses input_spike_detected flag and internal threshold.
        Updates self.prev_efforts.
        Returns filtered efforts_net.
        """
        masked_output = False
        
        if input_spike_detected:
            masked_output = True 
            
        if options.spike_threshold > 0.0:
            if self.prev_efforts is None:
                self.prev_efforts = np.zeros_like(efforts_net)
                
            # Check for Output-based spikes only if Input trigger didn't catch it
            if not masked_output:
                if efforts_net.shape == self.prev_efforts.shape:
                    diff = np.abs(efforts_net - self.prev_efforts)
                    mask = diff > options.spike_threshold
                    if np.any(mask):
                        efforts_net[mask] = self.prev_efforts[mask]
                        
        # Global Override for Input Detection
        if masked_output:
            if self.prev_efforts is not None:
                if efforts_net.shape == self.prev_efforts.shape:
                    efforts_net = self.prev_efforts.copy()
            else:
                 efforts_net[:] = 0.0 # Safety
                 
        self.prev_efforts = efforts_net.copy()
        
        return efforts_net
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

        # --- Input Spike Detection (Teleport) ---
        input_spike_detected = self._detect_input_spikes(
            pose_data_aa, world_pos, options
        )

        # --- Floor Contact Detection (BEFORE Torques now) ---
        contact_mask = self._compute_floor_contacts(
            world_pos, pose_data_aa, options
        )

        # --- Angular Kinematics (Main Path: Contact/Gravity) ---
        # Uses OneEuroFilter if enabled
        ang_vel, ang_acc = self._compute_angular_kinematics(
            F, pose_data_aa, quats, options, use_filter=False, state_suffix=''
        ) # was true
        
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
             # Bypass Filter (Assume User pre-filtered) -> use_filter=False
             # Use separate state history -> state_suffix='_effort'
             _, effort_ang_acc = self._compute_angular_kinematics(
                 F, effort_aa, effort_quats, options, use_filter=False, state_suffix='_effort'
             )
             ang_acc_for_effort = effort_ang_acc
             eq = np.all(np.equal(ang_acc_for_effort, ang_acc))


        if ang_acc_for_effort.ndim == 2:
             ang_acc_for_effort = ang_acc_for_effort[np.newaxis, ...]
             
        if ang_acc.ndim == 2:
             ang_acc = ang_acc[np.newaxis, ...]
        
        # --- Joint Torque & Effort Calculation ---
        # Now pass contact_mask and tips
        # Key: Pass ang_acc_for_effort!
        torques_vec, inertias, efforts_net, efforts_dyn, efforts_grav, t_dyn_vecs, t_grav_vecs, t_floor_vecs = self._compute_joint_torques(
            F, ang_acc_for_effort, world_pos, parents, global_rots,
            pose_data_aa, contact_mask, tips, options
        )
        
        # --- Probabilistic Contact (Parallel Output) ---
        contact_probs = self._compute_probabilistic_contacts(
             world_pos, pose_data_aa, options
        )
        
        # Scalar torque magnitude for output
        torques = np.linalg.norm(torques_vec, axis=-1)

        # --- Output Spike Rejection ---
        efforts_net = self._apply_output_spike_rejection(efforts_net, input_spike_detected, options)
            
        output_quats = quats[:, :self.target_joint_count, :]
        
        # --- State Updates ---
        # Update contact mask history (for hysteresis)
        self.prev_contact_mask = contact_mask[-1].copy() if F > 0 else self.prev_contact_mask
        
        # Update tip history
        if hasattr(self, 'temp_tips') and self.temp_tips:
             self.last_tip_positions = {k: v[-1].copy() for k, v in self.temp_tips.items()}

        # --- Output Dictionary ---
        res = {
            'pose': output_quats if options.return_quats else pose_data_aa[:, :self.target_joint_count, :],
            'trans': trans_data,
            'torques': torques,
            'torques_vec': torques_vec,
            'torques_grav_vec': t_grav_vecs,
            'torques_floor_vec': t_floor_vecs,
            'torques_dyn_vec': t_dyn_vecs,
            'inertias': inertias,
            'efforts_dyn': efforts_dyn,
            'efforts_grav': efforts_grav,
            'efforts_net': efforts_net,
            'positions': world_pos,
            'floor_contact': contact_mask,
            'contact_probs': contact_probs
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
