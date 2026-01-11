import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import pickle
from dataclasses import dataclass
import logging
from dpg_system.one_euro_filter import OneEuroFilter

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

    def _compute_limb_properties(self):
        """
        Approximates limb lengths and masses.
        If a model is loaded, we could be more precise. 
        Without a model, we use anthropometric tables scaled by height/gender assumptions.
        """
        
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
        
        # Approximate lengths (in meters) for a standard ~1.7m human.
        # We can try to adjust this based on betas[0] (height correlation) lightly if we want.
        # Beta[0] is roughly height/scaling. +1 sigma is taller.
        # A very rough approximation: scale = 1.0 + (beta[0] * 0.02)
        scale_factor = 1.0
        if self.betas is not None and len(self.betas) > 0:
            scale_factor += self.betas[0] * 0.02
        
        defaults = {
            'pelvis_width': 0.25,
            'upper_leg': 0.45,
            'lower_leg': 0.42,
            'foot': 0.20, # length
            'spine_segment': 0.12, # approx per spine joint
            'neck': 0.10,
            'head': 0.20,
            'shoulder_width': 0.40,
            'upper_arm': 0.30,
            'lower_arm': 0.28,
            'hand': 0.18
        }
        
        lengths = {k: v * scale_factor for k, v in defaults.items()}
        
        # Masses
        m = self.total_mass_kg
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
        
        return {'lengths': lengths, 'masses': masses}

    def _compute_skeleton_offsets(self):
        """
        Compute static local offsets for the skeleton based on limb lengths.
        Returns:
            offsets (np.array): (24, 3) Local position vectors for each joint in parent frame.
        """
        offsets = np.zeros((24, 3))
        
        for i in range(24):
            node_name = self.joint_names[i]
            
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
                    length = 0.05
                else:
                    length = 0.15
                dir_x = 1.0 if 'left' in node_name else -1.0
                lx = dir_x
                ly = 0.0
                lz = 0.0
                
            elif 'elbow' in node_name or 'wrist' in node_name:
                dir_x = 1.0 if 'left' in node_name else -1.0
                lx, ly, lz = dir_x, 0.0, 0.0
                
            elif 'hand' in node_name:
                length = 0.08 
                dir_x = 1.0 if 'left' in node_name else -1.0
                lx, ly, lz = dir_x, 0.0, 0.0
                
            elif 'foot' in node_name:
                length = 0.15 
                # Tuning V3.5
                lx, ly, lz = 0.0, 0.5, 0.8
            
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
            'neck': 50.0, # Adjusted up from 30 (User felt effort was too high)
            'head': 15.0,
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


    def _get_hierarchy(self):
        """Defines parent-child relationships for SMPL 24 joints."""
        # Parent indices for standard SMPL
        # -1 means root
        parents = [-1,  0,  0,  0,  1,  2, 
                    3,  4,  5,  6,  7,  8, 
                    9,  9,  9, 12, 13, 14, # Shoulders (16,17) parented to Collars (13,14)
                   16, 17, 18, 19, 20, 21]
                   
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
    def _compute_gravity_torques(self, world_pos, options):
        """
        Compute gravity torque vectors for all joints.
        
        Args:
            world_pos (np.array): (F, 24, 3) World positions of joints.
            options (SMPLProcessingOptions): Usage: add_gravity, input_up_axis.
            
        Returns:
            t_grav_vecs (np.array): (F, 24, 3) Gravity torque vectors.
        """
        F = world_pos.shape[0]
        t_grav_vecs = np.zeros((F, self.target_joint_count, 3))
        
        if not options.add_gravity:
            return t_grav_vecs
            
        # Determine Gravity Vector
        g_mag = 9.81
        if options.input_up_axis == 'Z':
            g_vec = np.array([0.0, 0.0, -g_mag])
        else: 
            g_vec = np.array([0.0, -g_mag, 0.0])
            
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
            # If no children (Leaf), assume Joint Position? Or project along bone?
            # Current logic in _compute_subtree_com uses Mean(Children).
            
            child_nodes = [c for c, p in enumerate(parents) if p == idx]
            
            joint_pos = world_pos[:, idx, :]
            if len(child_nodes) > 0:
                # Vectorized Mean of Children
                # Optimization: Pre-compute child lists? 
                # For 24 joints, list comp is fast enough per frame?
                # Creating list 24 times * F? No, iterate idx, handle all F.
                
                # Using 'sum' + divide is faster than mean logic manually?
                child_pos_sum = np.zeros((F, 3))
                for c in child_nodes:
                    child_pos_sum += world_pos[:, c, :]
                
                end_pos = child_pos_sum / len(child_nodes)
                seg_com = (joint_pos + end_pos) * 0.5
            else:
                # Leaf Logic (e.g. Wrist) -> End Effectors might be handled elsewhere or just use joint
                # In old logic: "end_pos" wasn't defined if no children, so com_pos = world_pos
                seg_com = joint_pos
                
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
        
        # Optimization: Cache Full Body CoM (Root Subtree) for Floor Contact logic
        # subtree_com[:, 0, :] is the CoM of the whole tree (rooted at 0)
        # Note: If F=1, simple assignment. If F>1, whole array.
        self.current_com = subtree_com[:, 0, :] # (F, 3)
        self.current_total_mass = subtree_masses[:, 0] # (F,)
        
        return t_grav_vecs

    def _compute_floor_contacts(self, world_pos, pose_data_aa, options):
        """
        Compute floor contact masses (pressure) based on position, velocity, and dynamics.
        Factors in:
        - Base Proximity
        - Velocity Guards (Up/Down/Horizontal)
        - CoM / ZMP Stability
        - Tip Contacts (Toes/Heels/Hands)
        - Geometric Biases (Heel-Toe)
        """
        F = world_pos.shape[0]
        
        if not options.floor_enable:
             return np.zeros((F, 24), dtype=np.float32)
             
        y_dim = 2 if options.input_up_axis == 'Z' else 1
        
        # Aliases
        dt = options.dt
        floor_height = options.floor_height
        floor_tolerance = options.floor_tolerance
        heel_toe_bias = options.heel_toe_bias
        
        # 1. Base Joint Contact Quality (Vectorized Default)
        dist_to_floor = world_pos[..., y_dim] - floor_height
        dist_to_floor = np.maximum(dist_to_floor, 0.0)
        base_contact = 1.0 - (dist_to_floor / max(floor_tolerance, 1e-4))
        base_contact = np.clip(base_contact, 0.0, 1.0)
        
        # 2. Velocity Guard (Lift Detection + Horizontal Slip)
        if self.prev_world_pos is None:
             self.prev_world_pos = world_pos[0].copy() if world_pos.ndim == 3 else world_pos.copy()
             
        # dt passed in is 1.0/max(self.framerate, 1.0) usually? 
        # In process_frame it was calculated: dt = 1.0/max(self.framerate, 1.0)
        # We'll use the passed dt.
        
        lin_vel = np.zeros_like(world_pos)
        
        # Batch Mode Velocity
        if world_pos.shape[0] > 1:
             # First frame vs prev state
             lin_vel[0] = (world_pos[0] - self.prev_world_pos) / dt
             # Subsequent frames
             lin_vel[1:] = (world_pos[1:] - world_pos[:-1]) / dt
             
             # Update State to Last Frame of Batch
             self.prev_world_pos = world_pos[-1].copy()
        else:
             # Streaming Mode
             lin_vel = (world_pos - self.prev_world_pos) / dt
             self.prev_world_pos = world_pos.copy()
        
        # Upward Velocity Penalty
        vel_up = lin_vel[..., y_dim]
        v_min_up, v_max_up = 0.05, 1.0 # 5cm/s to 100cm/s
        vel_up_penalty = (vel_up - v_min_up) / (v_max_up - v_min_up)
        vel_up_penalty = np.clip(vel_up_penalty, 0.0, 1.0)
        
        # Downward Velocity Penalty (Contact Watch - suppress freefall contact)
        vel_down = -vel_up
        v_min_down, v_max_down = 0.20, 1.50
        vel_down_penalty = (vel_down - v_min_down) / (v_max_down - v_min_down)
        vel_down_penalty = np.clip(vel_down_penalty, 0.0, 1.0)
        
        # Horizontal Velocity Penalty (Sliding/Swinging)
        vel_horiz_vec = lin_vel.copy()
        vel_horiz_vec[..., y_dim] = 0.0 # Remove vertical component
        vel_horiz = np.linalg.norm(vel_horiz_vec, axis=2)
        
        v_min_h, v_max_h = 0.2, 4.0 # 20cm/s to 400cm/s (Relaxed for running stops)
        vel_horiz_penalty = (vel_horiz - v_min_h) / (v_max_h - v_min_h)
        vel_horiz_penalty = np.clip(vel_horiz_penalty, 0.0, 1.0)
        
        # Guards
        velocity_factor = (1.0 - vel_up_penalty) * (1.0 - vel_down_penalty) * (1.0 - vel_horiz_penalty)
        
        # 3. CoM Weighting (Center of Gravity Shift)
        if self.current_com is not None:
            com = self.current_com
        else:
            com = self._compute_full_body_com(world_pos) # (F, 3)
        
        # --- ZMP Dynamics (Dynamic CoM) ---
        # 1. Calculate CoM Velocity and Acceleration
        com_vel = np.zeros_like(com)
        if F > 1:
             # Vectorized (Axis 0 is Time)
             com_vel[1:] = (com[1:] - com[:-1]) / dt
             if self.prev_com_pos is not None:
                 com_vel[0] = (com[0] - self.prev_com_pos) / dt
        elif self.prev_com_pos is not None:
             com_vel[0] = (com[0] - self.prev_com_pos) / dt
             
        # Filter Velocity? Or compute Accel from Vel diff
        com_accel = np.zeros_like(com)
        if F > 1:
             com_accel[1:] = (com_vel[1:] - com_vel[:-1]) / dt
             if self.prev_com_vel is not None:
                 com_accel[0] = (com_vel[0] - self.prev_com_vel) / dt
        elif self.prev_com_vel is not None:
             # Calculate raw acceleration (streaming)
             com_accel_curr = (com_vel - self.prev_com_vel) / dt
             com_accel = com_accel_curr # Assign to array (broadcasting if array 1)
             
        # --- Tuning: Amplify Acceleration ---
        com_accel *= 5.0  
        
        # Filter Acceleration (OneEuro)
        if hasattr(self, 'com_accel_filter'):
            # In batch, filter sequentially
            for f_idx in range(F):
                 com_accel[f_idx] = self.com_accel_filter(com_accel[f_idx])
        
        # Update State
        if F > 0:
             self.prev_com_pos = com[-1].copy()
             self.prev_com_vel = com_vel[-1].copy()
             
        # 2. Calculate ZMP Offset
        g = 9.81
        com_h = com[..., y_dim] - floor_height
        com_h = np.maximum(com_h, 0.1) # Min height 10cm safety
        
        # Accel Horizontal
        accel_h = com_accel.copy()
        accel_h[..., y_dim] = 0.0
        
        zmp_offset = - (com_h[:, np.newaxis] / g) * accel_h
        
        # Target Point for Balance
        balance_target = com + zmp_offset
        # ----------------------------------

        # Horizontal Distance to BALANCE TARGET (ZMP)
        diff = world_pos - balance_target[:, np.newaxis, :]
        diff[..., y_dim] = 0.0
        dist_sq = np.sum(diff**2, axis=2)
        
        # Gaussian Falloff
        sigma = 1.0 # Broaden
        com_factor = np.exp(-dist_sq / (2 * sigma**2))
        com_factor = 0.3 + 0.7 * com_factor # Min weight 0.3
        
        com = balance_target
        
        # Combine Base Contact Quality
        contact_mask = base_contact * velocity_factor * com_factor
        
        # --- Mass Distribution Logic ---
        total_mass = self.total_mass_kg
        contact_masses = np.zeros((F, 24), dtype=np.float32)
        
        # Calculate Root Speed (Pelvis) for Adaptive Guard
        root_vel_h = lin_vel[:, 0].copy()
        root_vel_h[..., y_dim] = 0.0
        root_speed = np.linalg.norm(root_vel_h, axis=1) # (F,)
        
        for f in range(F):
            # Adaptive Guard Threshold
            v_root_t = np.clip((root_speed[f] - 0.1) / (0.5 - 0.1), 0.0, 1.0)
            hold_guard_max = 2.5 * (1.0 - v_root_t) + 1.0 * v_root_t
            candidates = [] 
            
            # 1. Base Joints
            for i in range(24):
                if contact_mask[f, i] > 0.01:
                    # Store
                    candidates.append({'idx': i, 'qual': contact_mask[f, i], 'd_sq': dist_sq[f, i], 'is_tip': False})
                    
            # 2. Add Tip Contacts
            for idx in [7, 8, 10, 11, 22, 23]:
                # 7,8=Heel, 10,11=Toe, 22,23=Hand
                 if idx in self.temp_tips:
                     tip_pos = self.temp_tips[idx] # (F, 3)
                     
                     # Tip Velocity
                     tv_up = lin_vel[f, idx, y_dim]
                     tv_up_pen = np.clip(tv_up / 0.5, 0.0, 1.0) 
                     
                     prev_contact_f = self.prev_contact_mask if F > 0 else np.zeros(24)
                     if self.prev_contact_mask is None: prev_contact_f = np.zeros(24)
                     
                     was_contact = prev_contact_f[idx] > 0.01 
                     was_contact_tip = was_contact
                     
                     # Use 'temp_tips' history if available?
                     tv_h = np.linalg.norm([lin_vel[f, idx, d] for d in range(3) if d != y_dim])
                     
                     v_min_d, v_max_d = 0.05, 0.50 
                     tv_down_pen = np.clip((-tv_up - v_min_d)/(v_max_d - v_min_d), 0.0, 1.0)
                     
                     # Streaming Hysteresis for Descent Guard
                     if pose_data_aa.shape[0] == 1:
                         if not hasattr(self, 'prev_tip_pen'): self.prev_tip_pen = {}
                         
                         decay = np.exp(-dt / 0.1) # 100ms Decay
                         last_pen = self.prev_tip_pen.get(idx, 0.0)
                         tv_down_pen = max(tv_down_pen, last_pen * decay)
                         self.prev_tip_pen[idx] = tv_down_pen
                     
                     v_min_h, v_max_h = 0.05, 1.50
                     tv_h_pen = np.clip((tv_h - v_min_h)/(v_max_h - v_min_h), 0.0, 1.0)
                     
                     # Dynamic Tolerance
                     entry_scale = 1.0 - 0.95 * tv_down_pen 
                     eff_tol_tip = floor_tolerance if was_contact_tip else (floor_tolerance * entry_scale)
                     
                     d_floor = tip_pos[f, y_dim] - floor_height
                     d_floor = np.maximum(d_floor, 0.0)
                     qual = 1.0 - (d_floor / max(eff_tol_tip, 1e-4))
                     qual = np.clip(qual, 0.0, 1.0)

                     if qual > 0.01:
                          if was_contact_tip:
                               # Holding
                               tv_h_hold_pen = np.clip((tv_h - 0.2)/(hold_guard_max - 0.2), 0.0, 1.0)
                               qual *= (1.0 - tv_up_pen) * (1.0 - tv_h_hold_pen)
                          else:
                               # Entry
                               qual *= (1.0 - tv_up_pen) * (1.0 - tv_h_pen)
                     
                     if qual > 0.01:
                         # Dist
                         d = tip_pos[f] - com[f]
                         d[y_dim] = 0.0 # Remove vertical component
                         d_sq = np.sum(d**2)
                         
                         current_q = contact_mask[f, idx]
                         contact_mask[f, idx] = max(current_q, qual)

                         candidates.append({'idx': idx, 'qual': qual, 'd_sq': d_sq, 'is_tip': True})
            
            if not candidates:
                continue

            # 3. Calculate Weights
            weights = []
            sigma = 0.30 
            denom = 2 * sigma * sigma
            
            for c in candidates:
                dist_factor = np.exp(-c['d_sq'] / denom) 
                w = c['qual'] * dist_factor
                weights.append(w)
                
            total_w = sum(weights)
            
            if total_w > 0:
                for i, c in enumerate(candidates):
                    mass_share = total_mass * (weights[i] / total_w)
                    target_idx = c['idx']
                    contact_masses[f, target_idx] += mass_share
            
            # --- Refine Foot/Ankle Balance (Post-Pass) ---
            for (idx_a, idx_f) in [(7, 10), (8, 11)]:
                m_a = contact_masses[f, idx_a]
                m_f = contact_masses[f, idx_f]
                total_pair = m_a + m_f
                
                w_foot = 0.5
                
                if total_pair > 0.1: 
                    # 1. Base Weight (Lean / Flat Logic)
                    plane_dims = [d for d in [0, 1, 2] if d != y_dim]
                    p_a = world_pos[f, idx_a, plane_dims]
                    p_f = world_pos[f, idx_f, plane_dims]
                    p_c = com[f, plane_dims]
                    
                    # Project CoM onto line A-F
                    vec_af = p_f - p_a
                    len_sq = np.sum(vec_af**2)
                    
                    if len_sq > 1e-6:
                        vec_ac = p_c - p_a
                        t = np.dot(vec_ac, vec_af) / len_sq
                        w_foot_base = np.clip(t, 0.0, 1.0)
                    else:
                        w_foot_base = 0.5
                        
                    # 2. Geometric Bias (Height Comparison)
                    geo_toe_bias = 0.0
                    geo_heel_bias = 0.0
                    toe_bias = 0.0
                    heel_bias = 0.0
                    
                    h_toe = 0.0
                    h_heel = 0.0
                    
                    if hasattr(self, 'temp_tips') and idx_f in self.temp_tips and idx_a in self.temp_tips:
                        h_toe = self.temp_tips[idx_f][f, y_dim]
                        h_heel = self.temp_tips[idx_a][f, y_dim]
                        
                        height_diff = (h_toe - h_heel) + heel_toe_bias 
                        
                        if height_diff < -0.005:
                             geo_toe_bias = np.clip((-height_diff - 0.005) / 0.035, 0.0, 1.0)
                        
                        if height_diff > 0.005:
                             geo_heel_bias = np.clip((height_diff - 0.005) / 0.045, 0.0, 1.0)
                             
                    if geo_toe_bias > 0: toe_bias = geo_toe_bias
                    if geo_heel_bias > 0: heel_bias = geo_heel_bias
                    
                    # Maintain Pitch Calculation for Velocity Logic (Reaching/Flattening)
                    foot_pitch = 0.0
                    if hasattr(self, 'temp_tips') and idx_f in self.temp_tips:
                        t_pos = self.temp_tips[idx_f][f]
                        v_ft = t_pos - world_pos[f, idx_f]
                        v_ft_h = np.linalg.norm([v_ft[d] for d in range(3) if d != y_dim])
                        if v_ft_h > 0.01:
                            foot_pitch = np.arctan2(v_ft[y_dim], v_ft_h) * (180.0 / np.pi)
                            
                    # Calculate Pitch Velocity
                    if f > 0:
                        pitch_vel = foot_pitch - self.prev_foot_pitch_arr[idx_f]
                    else:
                        pitch_vel = 0.0
                    self.prev_foot_pitch_arr[idx_f] = foot_pitch
                        
                    # --- Bias Suppression by ZMP (Physics Wins) ---
                    if len_sq > 1e-6:
                        vec_af_norm = vec_af / np.sqrt(len_sq)
                        zmp_off_vec = zmp_offset[f, plane_dims]
                        zmp_proj = np.dot(zmp_off_vec, vec_af_norm)
                        
                        # Heel Suppression (ZMP Forward > 2cm)
                        if heel_bias > 0.0 and zmp_proj > 0.02: 
                             suppress = np.clip((zmp_proj - 0.02) / 0.08, 0.0, 1.0)
                             heel_bias *= (1.0 - suppress)
                             
                             dynamic_toe = suppress
                             toe_bias = max(toe_bias, dynamic_toe)
                             
                        # Toe Suppression (ZMP Backward < -2cm)
                        if toe_bias > 0.0 and zmp_proj < -0.02:
                             suppress = np.clip((-zmp_proj - 0.02) / 0.08, 0.0, 1.0)
                             toe_bias *= (1.0 - suppress)
                             
                    # --- Ankle Rotation / Reaching Logic ---
                    if idx_f in [10, 11] and pitch_vel < -1.0: 
                         if h_toe < 0.15:
                                 reaching_bias = np.clip((-pitch_vel - 1.0) / 4.0, 0.0, 1.0)
                                 toe_bias = max(toe_bias, reaching_bias)
                             
                    # --- Flattening Bias (Heel Drop) ---
                    if foot_pitch < 0.0 and pitch_vel > 0.0:
                        flattening_bias = np.clip((-foot_pitch) / 10.0, 0.0, 1.0) 
                        toe_bias = max(toe_bias, flattening_bias)
                    
                    # Blend
                    w_foot_target = w_foot_base * (1.0 - toe_bias - heel_bias) + (1.0 * toe_bias) + (0.0 * heel_bias)
                    
                    # --- Temporal Smoothing (Anti-Jitter) ---
                    alpha_smooth = 0.2
                    pair_idx = 0 if idx_f == 10 else 1
                    
                    if f == 0 and self.prev_w_foot is not None:
                         # Use carried over state
                         w_foot = self.prev_w_foot[pair_idx] * (1.0 - alpha_smooth) + w_foot_target * alpha_smooth
                         
                    # Update State
                    self.prev_w_foot[pair_idx] = w_foot

                    # 3. Hover Guard
                    if (toe_bias < 0.1 and heel_bias < 0.1):
                        if min(h_heel, h_toe) > floor_height + 0.05:
                            contact_masses[f, idx_f] = 0.0
                            contact_masses[f, idx_a] = 0.0
                        else:
                            contact_masses[f, idx_f] = total_pair * w_foot
                            contact_masses[f, idx_a] = total_pair * (1.0 - w_foot)
                    else:
                        # Bias active (Toe or Heel Reach) -> Allow contact 
                        contact_masses[f, idx_f] = total_pair * w_foot
                        contact_masses[f, idx_a] = total_pair * (1.0 - w_foot)

        
        # --- Contact Smoothing (Filtered Floor Pressure) ---
        if pose_data_aa.shape[0] == 1:
            if not hasattr(self, 'contact_mass_filter'):
                 self.contact_mass_filter = OneEuroFilter(min_cutoff=1.0, beta=0.005, framerate=self.framerate)
                 
            # Filter (24,)
            curr_contacts = contact_masses[0]
            if self.contact_mass_filter._x.last_value is None:
                 filtered_contacts = self.contact_mass_filter(curr_contacts)
                 filtered_contacts = curr_contacts # Passthrough first frame
            else:
                 filtered_contacts = self.contact_mass_filter(curr_contacts)
                 
            contact_masses[0] = filtered_contacts

        # Convert to float32 mask
        contact_mask = contact_masses.astype(np.float32)

        # Update contact mask history (for hysteresis)
        self.prev_contact_mask = contact_mask[-1].copy() if F > 0 else self.prev_contact_mask
        
        return contact_mask
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
             
        elif input_up_axis == 'Z':
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
        world_pos = np.zeros((F, 24, 3))
        
        # Optimization: Create ONE Rotation object for all joints at once to speed up init
        # We need frames for a specific joint to be contiguous for easy slicing later.
        # quats input is (F, 24, 4), so we transpose to (24, F, 4)
        quats_perm = quats.transpose(1, 0, 2).reshape(-1, 4) # (24*F, 4)
        all_rots = R.from_quat(quats_perm)
        
        # We store slices for each joint to avoid re-slicing (though slicing R is cheapish)
        # Actually we just slice on the fly to keep code clean.
        
        global_rots = [None] * 24
        parents = self._get_hierarchy()

        for i in range(24):
            parent = parents[i]
            
            # Slice the global rotation object
            # Because we transposed to (24, F, 4), the frames for joint i are at [i*F : (i+1)*F]
            start = i * F
            end = (i + 1) * F
            rot_local = all_rots[start:end]
            
            if parent == -1:
                global_rots[i] = rot_local
                world_pos[:, i, :] = trans_data # (F, 3)
            else:
                global_rots[i] = global_rots[parent] * rot_local
                
                # Calculate Offset
                local_pos = self.skeleton_offsets[i]

                # Apply Rotation (Parent's Global Rotation) to the Local Offset
                offset_vec = global_rots[parent].apply(local_pos)
                world_pos[:, i, :] = world_pos[:, parent, :] + offset_vec
                
        # --- Tip Projection for Contact Detection ---
        tips = {} # index -> pos (F, 3)
        
        # Feet (10, 11)
        # Calibrated Offset (Frame 0 Analysis with Reflection)
        # Vector: [0, -0.14, 0.05] (Local Y-down/Z-forward mix)
        if hasattr(self, 'foot_offset') and self.foot_offset is not None:
             foot_offset = self.foot_offset
        else:
             foot_offset = np.array([0.0, -0.14, 0.05])
             
        for idx in [10, 11]:
            if idx < 24:
                tips[idx] = world_pos[:, idx, :] + global_rots[idx].apply(foot_offset)

        # Heels (7, 8 - Ankles) - Standard Heel Drop: 5cm
        heel_offset = np.array([0.0, -0.05, 0.0])
        for idx in [7, 8]:
            tips[idx] = world_pos[:, idx, :] + global_rots[idx].apply(heel_offset)
            
        # Hands (22, 23)
        hand_len = self.limb_data['lengths']['hand'] - 0.08
        
        for idx, sign in [(22, 1.0), (23, -1.0)]: # Left (+), Right (-)
            if idx < 24:
                 hand_offset = np.array([sign, 0.0, 0.0]) * max(hand_len, 0.1)
                 tips[idx] = world_pos[:, idx, :] + global_rots[idx].apply(hand_offset)
                 
                  
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

    def _compute_angular_kinematics(self, F, pose_data_aa, quats, options):
        """
        Computes Angular Velocity and Acceleration.
        Handles both Streaming (OneEuroFilter) and Batch (Vectorized) modes.
        Updates internal state (prev_pose_q, prev_vel_aa, etc.).
        
        Args:
            F (int): Number of frames.
            pose_data_aa (F, 24, 3): Pose axis-angle.
            quats (F, 24, 4): Pose quaternions.
            options (SMPLProcessingOptions): Usage: dt, filter_min_cutoff, filter_beta, accel_clamp.
            
        Returns:
            ang_vel (F, 24, 3): Angular velocity vectors.
            ang_acc (F, 24, 3): Angular acceleration vectors.
        """
        dt = options.dt
        
        if F == 1 and hasattr(self, 'prev_pose_aa'):
            # Streaming mode
            curr_aa = pose_data_aa[0] # (24, 3)
            
            if self.prev_pose_aa is None:
                # First frame, no velocity/accel
                ang_vel = np.zeros_like(curr_aa)
                ang_acc = np.zeros_like(curr_aa)
                self.prev_pose_aa = curr_aa
                self.prev_vel_aa = ang_vel
                
                # Check for Shape Mismatch (Re-init filter if needed) -> Logic moved inside OneEuro Init check?
                # We need to initialize filter if not exists
                if self.pose_filter is None:
                     self.pose_filter = OneEuroFilter(min_cutoff=options.filter_min_cutoff, beta=options.filter_beta, framerate=self.framerate)
            else:
                dt = 1.0 / self.framerate if self.framerate > 0 else 0.016
                
                # One Euro Filter for Pose
                if options.enable_one_euro_filter:
                    if self.pose_filter is None:
                        # Initialize
                        self.pose_filter = OneEuroFilter(min_cutoff=options.filter_min_cutoff, beta=options.filter_beta, framerate=self.framerate)
                    
                    # Update params dynamicallly (if changed via UI)
                    self.pose_filter._mincutoff = options.filter_min_cutoff
                    self.pose_filter._beta = options.filter_beta
                    
                    # Filter the signal (Pose Quaternions to handle wrapping)
                    # Flatten to (96,) for filter
                    # Convert AA to Quat
                    curr_q = quats[0].flatten() # (96,)
                    
                    # Check for Shape Mismatch (Re-init filter if needed)
                    if self.pose_filter._x.last_value is not None:
                        if self.pose_filter._x.last_value.shape != curr_q.shape:
                            # Reset filter state completely
                            self.pose_filter = OneEuroFilter(min_cutoff=options.filter_min_cutoff, beta=options.filter_beta, framerate=self.framerate)
                            self.prev_pose_q = None # Reset our history too
                            
                    # Align Quaternions to avoid flips (Double Cover)
                    if self.pose_filter._x.last_value is not None:
                        prev_filtered_q = self.pose_filter._x.last_value
                        # Vectorized Dot Product
                        # Reshape to (24, 4)
                        c_rs = curr_q.reshape(-1, 4)
                        p_rs = prev_filtered_q.reshape(-1, 4)
                        dots = np.sum(c_rs * p_rs, axis=1) # (24,)
                        mask = dots < 0
                        if np.any(mask):
                            c_rs[mask] *= -1.0
                            curr_q = c_rs.flatten()
                    
                    # Filter
                    filtered_q_flat = self.pose_filter(curr_q)
                else:
                    # Bypass Filter
                    curr_q = quats[0].flatten()
                    filtered_q_flat = curr_q
                
                # Normalize Filtered Quats
                filtered_q = filtered_q_flat.reshape(24, 4)
                norms = np.linalg.norm(filtered_q, axis=1, keepdims=True)
                filtered_q /= (norms + 1e-8)
                
                # Velocity computation using FILTERED Quaternions
                r_curr = R.from_quat(filtered_q)
                
                # Need previous FILTERED quat
                if not hasattr(self, 'prev_pose_q') or self.prev_pose_q is None:
                    ang_vel = np.zeros((24, 3))
                    raw_ang_acc = np.zeros((24, 3))
                    # Initialize history
                    self.prev_pose_q = filtered_q
                else:
                    r_prev = R.from_quat(self.prev_pose_q)
                    r_diff = r_curr * r_prev.inv()
                    diff_vec = r_diff.as_rotvec()
                    
                    ang_vel = diff_vec.reshape(24, 3) / dt
                    
                    # Acceleration
                    if self.prev_vel_aa is None:
                        raw_ang_acc = np.zeros_like(ang_vel)
                    else:
                        raw_ang_acc = (ang_vel - self.prev_vel_aa) / dt
                
                # Smoothing implicitly handled by OneEuroFilter on Pose
                ang_acc = raw_ang_acc
                
                # Acceleration Clamping REMOVED per task.
                
                # Update State with FILTERED values
                self.prev_pose_q = filtered_q
                self.prev_vel_aa = ang_vel
                self.prev_acc_aa = ang_acc
                
            # Expand dims to (F, 24, 3) for return
            ang_vel = ang_vel[np.newaxis, ...]
            ang_acc = ang_acc[np.newaxis, ...]

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
        t_floor_vecs = np.zeros((F, self.target_joint_count, 3))
        
        # If no floor, return zero
        if not options.floor_enable:
            return t_floor_vecs
            
        # Determine Up Vector (Opposite to Gravity)
        # Gravity is Down. Support is Up. 
        # F_support = m * -g.
        g_mag = 9.81
        if options.input_up_axis == 'Z':
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
        t_grav_vecs = self._compute_gravity_torques(world_pos, options)
        
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
            torque_floor = t_floor_vecs[:, j, :]
            
            # Net Torque Vector (World Step 1)
            # T_muscle = I*alpha - (T_gravity + T_floor)
            torque_net_world = torque_dyn - (torque_grav + torque_floor)
            
            # --- Transform to Parent Frame ---
            parent_idx = parents[j]
            if parent_idx != -1:
                # Inverse of Parent Rotation (Cached)
                r_parent_inv = global_rot_invs[parent_idx]
                
                # Transform Vectors
                t_net_local = r_parent_inv.apply(torque_net_world)
                
                # Also transform components if needed? 
                t_dyn_local = r_parent_inv.apply(torque_dyn)
                t_grav_local = r_parent_inv.apply(torque_grav)
                
            else:
                t_net_local = torque_net_world
                t_dyn_local = torque_dyn
                t_grav_local = torque_grav
            
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
    def process_frame(self, pose_data, trans_data, options):
        """
        Process a single frame or batch of frames.
        Tracks state for torque calculation if frames are passed one by one.
        quat_format: 'xyzw' (default, Scipy) or 'wxyz' (Scalar first). Only used if input_type='quat'.
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

        # --- Angular Kinematics ---
        ang_vel, ang_acc = self._compute_angular_kinematics(
            F, pose_data_aa, quats, options
        )

        if ang_acc.ndim == 2:
             ang_acc = ang_acc[np.newaxis, ...]
        
        # --- Joint Torque & Effort Calculation ---
        # Now pass contact_mask and tips
        torques_vec, inertias, efforts_net, efforts_dyn, efforts_grav, t_dyn_vecs, t_grav_vecs, t_floor_vecs = self._compute_joint_torques(
            F, ang_acc, world_pos, parents, global_rots,
            pose_data_aa, contact_mask, tips, options
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
            'floor_contact': contact_mask
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
