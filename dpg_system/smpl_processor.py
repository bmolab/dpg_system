import numpy as np
from scipy.spatial.transform import Rotation as R
import logging
from dpg_system.one_euro_filter import OneEuroFilter

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
            'neck': {'limit': np.radians(30), 'k': 20.0}, # Stiff neck limits (Lowered to 30 deg per user test)
            'spine': {'limit': np.radians(30), 'k': 50.0}, # Stiff spine
            'shoulder': {'limit': np.radians(90), 'k': 10.0}, # Flexible shoulder
            'elbow': {'limit': np.radians(160), 'k': 50.0}, # Hyperextension stop
            'knee': {'limit': np.radians(160), 'k': 50.0},
            'hip': {'limit': np.radians(90), 'k': 30.0},
            'default': {'limit': np.radians(180), 'k': 1.0} # Non-limiting
        }
        
        # We are interested in the first 22 joints (indices 0-21)
        self.target_joint_count = 22
        
        self.limb_data = self._compute_limb_properties()
        self.max_torques = self._compute_max_torque_profile()
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
            'ankle': 40.0, # Adjusted down from 150 (Dorsiflexion is weaker than Plantarflexion)
            'foot': 20.0,
            'neck': 50.0, # Adjusted up from 30 (User felt effort was too high)
            'head': 15.0,
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

    def set_max_torque(self, joint_name_filter, value):
        """
        Manually updates max torque for joints matching the filter.
        Args:
            joint_name_filter (str): Substring to match (e.g., 'neck', 'ankle').
            value (float): New max torque value in N-m.
        """
        count = 0
        for k in self.max_torques:
            if joint_name_filter in k:
                self.max_torques[k] = float(value)
                count += 1
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


    def _get_hierarchy(self):
        """Defines parent-child relationships for SMPL 24 joints."""
        # Parent indices for standard SMPL
        # -1 means root
        parents = [-1,  0,  0,  0,  1,  2, 
                    3,  4,  5,  6,  7,  8, 
                    9,  9,  9, 12,  9,  9, # Shoulders (16,17) parented to Spine3 (9) NOT Collars (13,14)
                   16, 17, 18, 19, 20, 21]
                   
        # Note: Collars (13, 14) are still children of Spine3 (9).
        # But Shoulders (16, 17) now bypass them physically for mass load calculation.
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
                
        limit_angle = limits['limit']
        k = limits['k']
        
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

    def process_frame(self, pose_data, trans_data, input_type='axis_angle', return_quats=False, 
                     input_up_axis='Y', smoothing=0.0, add_gravity=False, quat_format='xyzw', enable_passive_limits=False,
                     filter_min_cutoff=1.0, filter_beta=0.0, accel_clamp=0.0, spike_threshold=0.0, input_spike_threshold=0.0, accel_spike_threshold=0.0, jerk_threshold=0.0,
                     floor_enable=False, floor_height=0.0, floor_tolerance=0.15):
        """
        Process a single frame or batch of frames.
        Tracks state for torque calculation if frames are passed one by one.
        quat_format: 'xyzw' (default, Scipy) or 'wxyz' (Scalar first). Only used if input_type='quat'.
        """
        pose_data = np.array(pose_data)
        trans_data = np.array(trans_data)
        
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
            elif pose_data.shape[0] == 24: # (24, C) or (F, 72/96)? 
                 # If dim1 is 3 or 4, it's (24, C).
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
                # Identity depends on format
                if quat_format == 'xyzw':
                     p24[:, 22:, 3] = 1.0
                else: # wxyz
                     p24[:, 22:, 0] = 1.0
            
            pose_data = p24
            
            pose_data = p24
            
        F = pose_data.shape[0]
        
        # 4. Handle AMASS / Extra Joints (Crop to 24)
        # If Axis Angle (N, 3): 24*3 = 72
        if pose_data.shape[-1] > 72 and input_type == 'axis_angle':
            # Assumed flattened AA
            # Reshape to (-1, 3) -> take first 24 -> flatten back or keep?
            # Safe way:
            temp = pose_data.reshape(F, -1, 3)
            pose_data = temp[:, :24, :].reshape(F, -1)
            
        elif pose_data.shape[-1] > 96 and input_type == 'quat':
            # Assumed flattened Quat
             temp = pose_data.reshape(F, -1, 4)
             pose_data = temp[:, :24, :].reshape(F, -1)
        
        # Convert to Axis Angle
        # We work internally with Axis Angle for velocity (rad/s)
        
        if input_type == 'axis_angle':
            if pose_data.shape[-1] == 72:
                pose_data_aa = pose_data.reshape(F, 24, 3)
            else:
                pose_data_aa = pose_data
                
            # For output consistency
            flat_pose = pose_data_aa.reshape(-1, 3)
            r = R.from_rotvec(flat_pose)
            quats = r.as_quat().reshape(F, 24, 4)
            
        elif input_type == 'quat':
             quats = pose_data.reshape(F, 24, 4).copy()
             
             # Handle Format
             if quat_format == 'wxyz':
                 # Convert WXYZ -> XYZW for Scipy
                 # [w, x, y, z] -> [x, y, z, w]
                 # Roll -1
                 quats = np.roll(quats, -1, axis=-1)
             
             flat_q = quats.reshape(-1, 4)
             r = R.from_quat(flat_q)
             pose_data_aa = r.as_rotvec().reshape(F, 24, 3)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
            
        # --- Handle Coordinate System Conversion ---
        if input_up_axis == 'Z':
            # Convert Z-up to Y-up (Rotate -90 deg X)
            # 1. Transform Translation
            # (x, y, z) -> (x, z, -y)
            # Manually or via Rotation Matrix?
            # Manual is faster/clearer for axis swap.
            # trans_data shape (F, 3).
            tx, ty, tz = trans_data[..., 0], trans_data[..., 1], trans_data[..., 2]
            trans_data = np.stack([tx, tz, -ty], axis=-1)
            
            # 2. Transform Root Orientation (Joint 0)
            # R_new = R_conv * R_root
            r_conv = R.from_euler('x', -90, degrees=True)
            
            # Get current root rots
            r_root = R.from_rotvec(pose_data_aa[:, 0, :])
            r_root_new = r_conv * r_root
            
            # Update pose_data_aa
            pose_data_aa[:, 0, :] = r_root_new.as_rotvec()
            
            # Re-sync quats for downstream?
            # Yes, code uses 'quats' derived from pose_data_aa.
            # But wait, lines 530/534 create 'quats' variable.
            # If I modify 'pose_data_aa', I must update 'quats'.
            # Or delay creation of 'quats' until after?
            # 'quats' is used in Line 553.
            # So I must update 'quats' or regenerate it.
            # Since 'quats' logic was branched (lines 531 vs 534), regeneration is safest.
            
            flat_pose = pose_data_aa.reshape(-1, 3)
            r_all = R.from_rotvec(flat_pose)
            quats = r_all.as_quat().reshape(F, 24, 4)
            
            # Now that data is Y-up, update state so downstream uses Y-dim
            input_up_axis = 'Y'
            
        elif input_up_axis == 'Y':
            pass # No conversion
            
        else:
             print(f"Warning: input_up_axis='{input_up_axis}' not fully supported. Assuming Y-up.")
                   # 1. Forward Kinematics (Vectorized)
        world_pos = np.zeros((F, 24, 3))
        
        # Pre-convert local quats to list of R objects for speed/indexing
        # quats is (F, 24, 4)
        local_rots_list = [R.from_quat(quats[:, i, :]) for i in range(24)]
        global_rots = [None] * 24
        parents = self._get_hierarchy()

        for i in range(24):
            parent = parents[i]
            rot_local = local_rots_list[i]
            
            if parent == -1:
                global_rots[i] = rot_local
                world_pos[:, i, :] = trans_data # (F, 3)
            else:
                global_rots[i] = global_rots[parent] * rot_local
                
                # Find length & offset
                node_name = self.joint_names[i]
                length = 0.1
                if 'knee' in node_name: length = self.limb_data['lengths']['upper_leg']
                elif 'ankle' in node_name: length = self.limb_data['lengths']['lower_leg']
                elif 'foot' in node_name: length = 0.05
                elif 'elbow' in node_name: length = self.limb_data['lengths']['upper_arm']
                elif 'wrist' in node_name: length = self.limb_data['lengths']['lower_arm']
                elif 'hand' in node_name: length = self.limb_data['lengths']['hand']
                elif 'spine' in node_name: length = self.limb_data['lengths']['spine_segment']
                elif 'neck' in node_name or 'head' in node_name: length = self.limb_data['lengths']['neck']

                # Offset direction
                # Default Down (Legs)
                offset_local = np.array([0.0, -1.0, 0.0])
                if 'spine' in node_name or 'neck' in node_name or 'head' in node_name:
                     offset_local = np.array([0.0, 1.0, 0.0])
                
                if 'shoulder' in node_name or 'collar' in node_name: 
                    offset_local = np.array([1.0, 0.0, 0.0]) if 'left' in node_name else np.array([-1.0, 0.0, 0.0])
                elif 'elbow' in node_name or 'wrist' in node_name:
                    offset_local = np.array([1.0, 0.0, 0.0]) if 'left' in node_name else np.array([-1.0, 0.0, 0.0])
                elif 'hand' in node_name:
                    # Wrist to Hand (Knuckles): ~8-10cm, not full hand length
                    length = 0.08 
                    offset_local = np.array([1.0, 0.0, 0.0]) if 'left' in node_name else np.array([-1.0, 0.0, 0.0])
                
                # Apply Rotation
                offset_vec = global_rots[parent].apply(offset_local * length)
                world_pos[:, i, :] = world_pos[:, parent, :] + offset_vec
                
        # --- Tip Projection for Contact Detection ---
        # Calculate Virtual Tips for Hands and Feet
        # Foot (10/11): Base is ~5cm from Ankle. Add ~15cm for Toes.
        # Hand (22/23): Base is ~8cm from Wrist. Add ~10cm for Fingertips.
        
        tips = {} # index -> pos (F, 3)
        
        # Feet (10, 11)
        foot_len = self.limb_data['lengths']['foot'] - 0.05 # Remaining length
        foot_offset = np.array([0.0, -1.0, 0.0]) * max(foot_len, 0.1)
        
        for idx in [10, 11]:
            if idx < 24:
                # Use Global Rot of Foot Joint (10/11) to project Toes
                tips[idx] = world_pos[:, idx, :] + global_rots[idx].apply(foot_offset)

        # Hands (22, 23)
        hand_len = self.limb_data['lengths']['hand'] - 0.08
        
        for idx, sign in [(22, 1.0), (23, -1.0)]: # Left (+), Right (-)
            if idx < 24:
                 hand_offset = np.array([sign, 0.0, 0.0]) * max(hand_len, 0.1)
                 tips[idx] = world_pos[:, idx, :] + global_rots[idx].apply(hand_offset)
                
        # Store tips for contact check
        self.temp_tips = tips # Save for use in contact logic (hack/optimization)
                
        # --- Input Spike Detection (Teleport) ---
        input_spike_detected = False
        
        # Only run spike detection if streaming (frame by frame)
        if pose_data_aa.shape[0] == 1:
             # 1. Teleport Detection (Large Step)
             if input_spike_threshold > 0.0 and self.prev_pose_raw is not None:
                 diff = np.abs(pose_data_aa - self.prev_pose_raw)
                 diff_norm = np.linalg.norm(diff, axis=2)
                 th_rad = np.radians(input_spike_threshold)
                 
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
                   check_accel = (accel_spike_threshold > 0.0) or (jerk_threshold > 0.0)
                   
                   if check_accel:
                       # Accel Magnitude Check (Preliminary)
                       acc_mag = np.linalg.norm(acc_new, axis=2)
                       acc_th_rad = np.radians(accel_spike_threshold)
                       mask_spike = acc_mag > acc_th_rad
                       
                       # A. Sign Flip Check (Old Method)
                       if np.any(mask_spike) and accel_spike_threshold > 0.0:
                            if self.prev_acc_raw is not None:
                                 dot = np.sum(acc_new * self.prev_acc_raw, axis=2)
                                 mask_flip = dot < 0.0
                                 joint_reject = np.logical_and(mask_spike, mask_flip)
                                 if np.any(joint_reject):
                                      input_spike_detected = True
                                      
                       # B. Jerk Check (New Method)
                       if jerk_threshold > 0.0 and self.prev_acc_raw is not None:
                            jerk_vec = acc_new - self.prev_acc_raw
                            jerk_mag = np.linalg.norm(jerk_vec, axis=2)
                            jerk_th_rad = np.radians(jerk_threshold)
                            
                            
                            mask_jerk = jerk_mag > jerk_th_rad
                            
                            y_dim = 2 if input_up_axis == 'Z' else 1
                            
                            # --- Floor Contact Logic ---
                            if floor_enable:
                                 y_vals = world_pos[..., y_dim] # (1, 24)
                                 
                                 on_floor = y_vals < (floor_height + floor_tolerance)
                                 mask_jerk = np.logical_and(mask_jerk, ~on_floor)
                            
                            
                            if np.any(mask_jerk):
                                 # DEBUG
                                 input_spike_detected = True
                                 
                   self.prev_acc_raw = acc_new
                  
             # Update Pose History
             self.prev_pose_raw = pose_data_aa.copy()
             
        else:
             # Batch Mode - Reset State
             if pose_data_aa.shape[0] == 1: # Should not happen unless bad logic
                 pass
             else:
                 pass
                 # TODO: Batch logic?
                 # For now, just ensure we don't hold stale state if user calls batch then stream.
                 pass


        # --- Physics / Torque Calculation ---
        # 1. Calculate Angular Velocity & Acceleration
        # We need history.
        
        # Initialize torque array
        torques = np.zeros((F, self.target_joint_count)) # Scalar torque magnitude? Or Vector?
        # User asked for "torque required at each joint". Usually a scalar (magnitude) or vector.
        # Given "easier to rotate wrist vs shoulder", scalar magnitude of torque makes sense for "intensity".
        # But let's compute vector torque and return magnitude or vector.
        # Vector torque matches the axis of rotation.
        # Let's return N-m magnitude for simplicity unless requested. 
        # "torque required" -> usually N-m.
        
        # Handle state tracking for F=1 (streaming mode)
        # If F > 1, we assume it's a sequence and calculate derivatives within batch.
        
        if F == 1 and hasattr(self, 'prev_pose_aa'):
            # Streaming mode
            curr_aa = pose_data_aa[0] # (24, 3)
            
            if self.prev_pose_aa is None:
                # First frame, no velocity/accel
                ang_vel = np.zeros_like(curr_aa)
                ang_acc = np.zeros_like(curr_aa)
                self.prev_pose_aa = curr_aa
                self.prev_vel_aa = ang_vel
            else:
                dt = 1.0 / self.framerate
                
                # Velocity: (curr - prev) / dt
                # Note: Simple subtraction of AA vectors is approximate but okay for small steps.
                # Proper way: Relative rotation converted to axis-angle.
                # dR = R_curr * R_prev.T
                # vel = log(dR) / dt
                
                # One Euro Filter for Pose
                if self.pose_filter is None:
                    # Initialize
                    # Use provided parameters or defaults
                    self.pose_filter = OneEuroFilter(min_cutoff=filter_min_cutoff, beta=filter_beta, framerate=self.framerate)
                
                # Update params dynamicallly (if changed via UI)
                self.pose_filter._mincutoff = filter_min_cutoff
                self.pose_filter._beta = filter_beta
                
                # Filter the signal (Pose Quaternions to handle wrapping)
                # Flatten to (96,) for filter
                # Convert AA to Quat
                r = R.from_rotvec(curr_aa.reshape(-1, 3))
                curr_q = r.as_quat().flatten() # (96,)
                
                # Check for Shape Mismatch (Re-init filter if needed)
                # LowPassFilter inside OneEuroFilter stores _y (last value)
                if self.pose_filter._x.last_value is not None:
                    if self.pose_filter._x.last_value.shape != curr_q.shape:
                        # Reset filter state completely
                        self.pose_filter = OneEuroFilter(min_cutoff=filter_min_cutoff, beta=filter_beta, framerate=self.framerate)
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
                
                # Acceleration Clamping (Slew Rate Limit)
                if accel_clamp > 0.0:
                    acc_mags = np.linalg.norm(ang_acc, axis=1) # Shape (24,)
                    mask = acc_mags > accel_clamp
                    if np.any(mask):
                        scale = accel_clamp / (acc_mags[mask] + 1e-8)
                        ang_acc[mask] *= scale[:, np.newaxis]
                
                # Update State with FILTERED values
                self.prev_pose_q = filtered_q
                self.prev_vel_aa = ang_vel
                self.prev_acc_aa = ang_acc
                


        else:
            # Batch Mode (F > 1)
            # Calculate derivatives across the batch
            dt = 1.0 / self.framerate
            
            # 1. Angular Velocity
            # Need continuous representation or handle wrapping?
            # Axis-angle is generally continuous if not flipping. 
            # Ideally use Quaternions -> Relative Rotation -> Axis Angle.
            # But for batch, let's use the same finite diff logic or np.gradient on axis-angle for speed/simplicity
            # IF we assume continuous data. quaternion unwrap might be needed.
            # Let's stick to the robust relative rotation method but vectorized?
            # Calculating relative rotation for F frames is expensive. 
            # Approx: gradient of axis-angle.
            
            # Optimization: Just use np.gradient logic on flat AA.
            # vel = gradient(pose) / dt
            
            # Simple Centered Difference
            # ang_vel[i] = (pose[i+1] - pose[i-1]) / (2*dt)
            
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
            if accel_clamp > 0.0:
                # ang_acc is (F, 24, 3)
                # Calculate magnitudes (F, 24)
                acc_mags = np.linalg.norm(ang_acc, axis=2) 
                
                mask = acc_mags > accel_clamp
                
                if np.any(mask):
                    scales = accel_clamp / (acc_mags[mask] + 1e-8)
                    # mask has True where we want to clamp
                    # ang_acc[mask] selects those vectors (N, 3)
                    ang_acc[mask] *= scales[:, np.newaxis]

        # Per-joint Torque Calculation
        # Vectorized FK for Batch Support
        
        # Ensure 'ang_acc' is (F, 24, 3)
        if ang_acc.ndim == 2:
             ang_acc = ang_acc[np.newaxis, ...]
        
        # 1. Forward Kinematics (Computed Earlier)

        # Determine Gravity Vector
        g_vec = np.zeros(3)
        if add_gravity:
            g_mag = 9.81
            if input_up_axis == 'Z':
                g_vec = np.array([0.0, 0.0, -g_mag])
            else: 
                g_vec = np.array([0.0, -g_mag, 0.0])

        # 2. Torque / Effort Calculation (Vectorized)
        torques = np.zeros((F, self.target_joint_count))
        torques_vec = np.zeros((F, self.target_joint_count, 3))
        
        inertias = np.zeros((F, self.target_joint_count))
        
        efforts_dyn = np.zeros((F, self.target_joint_count))
        efforts_grav = np.zeros((F, self.target_joint_count))
        efforts_net = np.zeros((F, self.target_joint_count))
        
        # Vectors
        t_dyn_vecs = np.zeros((F, self.target_joint_count, 3))
        t_grav_vecs = np.zeros((F, self.target_joint_count, 3))

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
            
            torque_grav = np.zeros((F, 3))
            if add_gravity:
                 m_sub, com_sub = self._compute_subtree_com(j, world_pos, self.limb_data['lengths'], self.limb_data['masses'])
                 
                 if m_sub > 0:
                     # r_com: (F,3)
                     r_com = com_sub - world_pos[:, j, :]
                     f_gravity = m_sub * g_vec 
                     torque_grav = np.cross(r_com, f_gravity)
                     
            t_grav_vecs[:, j, :] = torque_grav
            
            # Net Torque Vector (World Step 1)
            # T_muscle = I*alpha - T_gravity
            torque_net_world = torque_dyn - torque_grav
            
            # --- Transform to Parent Frame ---
            # Users want torque relative to the parent limb (e.g. Chest frame for Shoulder).
            # This makes sense for "muscle effort" directions.
            
            parent_idx = parents[j]
            if parent_idx != -1:
                # Get Parent's Global Rotation
                # global_rots is list of R objects. 
                # But global_rots is for ONE frame? No, we need Vectorized R.
                # Oh, global_rots in loop above was (24,) list of R objects for CURRENT frame (if F=1?)
                # Wait, my vectorized FK I implemented earlier:
                # "global_rots = [None] * 24" inside the loop.
                # But that loop was inside "process_frame".
                
                # ISSUE: The vectorized FK loop computed 'global_rots' as a list of R objects.
                # BUT 'global_rots[parent]' is a Single R object containing (F,) rotations?
                # YES. "global_rots[i] = global_rots[parent] * rot_local" where rot_local is R(F, 4).
                # So global_rots[parent] is a valid R object associated with all F frames.
                
                # Inverse of Parent Rotation
                r_parent_inv = global_rots[parent_idx].inv()
                
                # Transform Vectors
                # torque_net_world is (F, 3). R.apply expects (F, 3).
                t_net_local = r_parent_inv.apply(torque_net_world)
                
                # Also transform components if needed? 
                # User asked for "torque vectors relative to... parent limb".
                # Implies components too? Yes.
                t_dyn_local = r_parent_inv.apply(torque_dyn)
                t_grav_local = r_parent_inv.apply(torque_grav)
                
            else:
                # Root: Parent is World? Or Pelvis Local?
                # "Local coordinate system of the parent limb". Parent of Root is World.
                # So keep World.
                t_net_local = torque_net_world
                t_dyn_local = torque_dyn
                t_grav_local = torque_grav
            
            # --- Passive Limits ---
            # Calculate Passive Torque based on Local Angle (pose_aa)
            # Use t_net_local as reference? No, depends on pose.
            # pose_aa: (F, 24, 3) -> pose_data_aa[:, j, :]
            # Must ensure input was AA. Yes, process_frame converts to AA early on. 'pose_data_aa'.
            
            t_passive_local = np.zeros_like(t_net_local)
            if enable_passive_limits:
                 # Current local pose for this joint
                 curr_pose_aa = pose_data_aa[:, j, :]
                 t_passive_local = self._compute_passive_torque(name, t_net_local, curr_pose_aa)
            
            # Active Active (Net - Passive)
            # If Net is holding head up (positive), and Passive Spring is pushing head UP (positive),
            # Then Active Muscle requirement is Net - Passive.
            t_active_local = t_net_local - t_passive_local
            
            # Store Vector (Local/Parent Frame)
            # We now output ACTIVE torque as the primary 'torques_vec'?
            # Or separate? User asked for "adjust the model". 
            # Ideally 'torques_vec' reflects "Active Muscle Effort".
            torques_vec[:, j, :] = t_active_local
            
            # Store Magnitude (Active)
            torques[:, j] = np.linalg.norm(t_active_local, axis=1)
            
            torques_vec[:, j, :] = t_active_local
            
            # Store Magnitude (Active)
            torques[:, j] = np.linalg.norm(t_active_local, axis=1)
            
            # Effort Normalization
            name = self.joint_names[j]
            max_t = 100.0
            if 'pelvis' in name: max_t = self.max_torques.get('pelvis', 500.0)
            elif 'hip' in name: max_t = self.max_torques.get('hip', 300.0)
            elif 'knee' in name: max_t = self.max_torques.get('knee', 250.0)
            elif 'ankle' in name: max_t = self.max_torques.get('ankle', 150.0)
            elif 'foot' in name: max_t = self.max_torques.get('foot', 30.0)
            elif 'spine' in name: max_t = self.max_torques.get('spine', 400.0)
            elif 'neck' in name: max_t = self.max_torques.get('neck', 30.0)
            elif 'head' in name: max_t = self.max_torques.get('head', 15.0)
            elif 'shoulder' in name: max_t = self.max_torques.get('shoulder', 120.0)
            elif 'elbow' in name: max_t = self.max_torques.get('elbow', 80.0)
            elif 'wrist' in name: max_t = self.max_torques.get('wrist', 20.0)
            elif 'hand' in name: max_t = self.max_torques.get('hand', 10.0)
            
            efforts_dyn[:, j] = np.linalg.norm(torque_dyn, axis=1) / max_t
            efforts_grav[:, j] = np.linalg.norm(torque_grav, axis=1) / max_t
            
            # Net Effort (Combined Vector Magnitude / Max)
            # torque_net_world magnitude is already in torques[:, j]
            # We use that (rotation invariant)
            efforts_net[:, j] = torques[:, j] / max_t

        # --- Output Spike Rejection ---
        # Logic: If Input Spike was detected, FORCE output rejection.
        # Otherwise, check Output Spike Threshold if enabled.
        
        masked_output = False
        
        if input_spike_detected:
            masked_output = True 
            
        if spike_threshold > 0.0:
            if self.prev_efforts is None:
                self.prev_efforts = np.zeros_like(efforts_net)
                
            # Check for Output-based spikes only if Input trigger didn't catch it
            if not masked_output:
                if efforts_net.shape == self.prev_efforts.shape:
                    diff = np.abs(efforts_net - self.prev_efforts)
                    mask = diff > spike_threshold
                    if np.any(mask):
                        efforts_net[mask] = self.prev_efforts[mask]
                        # We don't set masked_output=True globally because we handled it per-joint
                        
        # Global Override for Input Detection
        if masked_output:
            if self.prev_efforts is not None:
                if efforts_net.shape == self.prev_efforts.shape:
                    efforts_net = self.prev_efforts.copy()
            else:
                 efforts_net[:] = 0.0 # Safety
                 
        self.prev_efforts = efforts_net.copy()
            
        output_quats = quats[:, :self.target_joint_count, :]
        
        
        # Calculate Floor Contact Mask for Output (Float)
        # Represents "Pressure/Support"
        y_dim = 2 if input_up_axis == 'Z' else 1
        
        # 1. Base Contact (Height Falloff)
        # 1.0 at floor, 0.0 at tolerance
        dist_to_floor = world_pos[..., y_dim] - floor_height
        dist_to_floor = np.maximum(dist_to_floor, 0.0)
        base_contact = 1.0 - (dist_to_floor / max(floor_tolerance, 1e-4))
        base_contact = np.clip(base_contact, 0.0, 1.0)
        
        # 2. Velocity Guard (Lift Detection + Horizontal Slip)
        if self.prev_world_pos is None:
             self.prev_world_pos = world_pos.copy()
             
        dt = 1.0 / max(self.framerate, 1.0)
        lin_vel = (world_pos - self.prev_world_pos) / dt
        self.prev_world_pos = world_pos.copy()
        
        # Upward Velocity Penalty
        vel_up = lin_vel[..., y_dim]
        v_min_up, v_max_up = 0.05, 0.25 # 5cm/s to 25cm/s
        vel_up_penalty = (vel_up - v_min_up) / (v_max_up - v_min_up)
        vel_up_penalty = np.clip(vel_up_penalty, 0.0, 1.0)
        
        # Horizontal Velocity Penalty (Sliding/Swinging)
        vel_horiz_vec = lin_vel.copy()
        vel_horiz_vec[..., y_dim] = 0.0 # Remove vertical component
        vel_horiz = np.linalg.norm(vel_horiz_vec, axis=2)
        
        v_min_h, v_max_h = 0.05, 0.60 # 5cm/s to 60cm/s (Relaxed for dynamic pivots)
        vel_horiz_penalty = (vel_horiz - v_min_h) / (v_max_h - v_min_h)
        vel_horiz_penalty = np.clip(vel_horiz_penalty, 0.0, 1.0)
        
        velocity_factor = (1.0 - vel_up_penalty) * (1.0 - vel_horiz_penalty)
        
        # 3. CoM Weighting (Center of Gravity Shift)
        # Higher pressure if closer to vertical projection of CoM
        com = self._compute_full_body_com(world_pos) # (F, 3)
        
        # Horizontal Distance
        diff = world_pos - com[:, np.newaxis, :]
        diff[..., y_dim] = 0.0
        dist_sq = np.sum(diff**2, axis=2)
        
        # Gaussian Falloff
        sigma = 0.4
        com_factor = np.exp(-dist_sq / (2 * sigma_factor**2)) if 'sigma_factor' in locals() else np.exp(-dist_sq / (2 * 0.4**2))
        com_factor = 0.3 + 0.7 * com_factor # Min weight 0.3
        
        
        # Combine Base Contact Quality
        contact_mask = base_contact * velocity_factor * com_factor
        
        # --- Mass Distribution Logic ---
        # Instead of binary/float contact, we distribute Total Body Mass
        # based on contact quality and proximity to CoM projection.
        
        total_mass = self.total_mass_kg
        contact_masses = np.zeros((F, 24), dtype=np.float32)
        
        # Iterate over batch (vectorizing this fully is complex due to varying candidates per frame)
        # So we do a loop over F (usually 1).
        
        for f in range(F):
            candidates = [] # (index, quality, dist_sq_to_com)
            
            # 1. Base Joints
            for i in range(24):
                qual = contact_mask[f, i]
                if qual > 0.01:
                    # Dist to CoM (Horizontal)
                    d = world_pos[f, i] - com[f]
                    d[y_dim] = 0.0
                    d_sq = np.sum(d**2)
                    candidates.append({'idx': i, 'qual': qual, 'd_sq': d_sq, 'is_tip': False})
                    
            # 2. Tips (if any)
            if hasattr(self, 'temp_tips') and self.temp_tips:
                if not hasattr(self, 'prev_tips') or self.prev_tips is None:
                     self.prev_tips = {k: v.copy() for k, v in self.temp_tips.items()}
                     
                for idx, tip_pos in self.temp_tips.items():
                     if idx >= 24: continue
                     
                     # Re-calc qual (Height)
                     d_floor = tip_pos[f, y_dim] - floor_height
                     d_floor = np.maximum(d_floor, 0.0)
                     qual = 1.0 - (d_floor / max(floor_tolerance, 1e-4))
                     qual = np.clip(qual, 0.0, 1.0)
                     
                     if qual > 0.01:
                         # Velocity Check
                         prev_tip = None
                         if f > 0:
                             prev_tip = tip_pos[f-1]
                         elif hasattr(self, 'last_tip_positions') and idx in self.last_tip_positions:
                             prev_tip = self.last_tip_positions[idx]
                         else:
                             prev_tip = tip_pos[f] # 0 vel
                             
                         t_vel = (tip_pos[f] - prev_tip)/dt
                         
                         tv_up = t_vel[y_dim]
                         tv_h = np.linalg.norm(t_vel - [0, t_vel[y_dim], 0] if y_dim==1 else [0,0,0]) 
                         
                         tv_up_pen = np.clip((tv_up - 0.05)/(0.25-0.05), 0.0, 1.0)
                         tv_h_pen = np.clip((tv_h - 0.05)/(0.60-0.05), 0.0, 1.0) # Relaxed to 60cm/s
                         
                         qual *= (1.0 - tv_up_pen) * (1.0 - tv_h_pen)
                     
                     if qual > 0.01:
                         # Dist
                         d = tip_pos[f] - com[f]
                         d[y_dim] = 0.0 # Remove vertical component
                         d_sq = np.sum(d**2)
                         # Propagate Tip Quality to Parent Joint Mask for Mass Distribution Logic
                         current_q = contact_mask[f, idx]
                         contact_mask[f, idx] = max(current_q, qual)

                         candidates.append({'idx': idx, 'qual': qual, 'd_sq': d_sq, 'is_tip': True})
            
            if not candidates:
                continue

            # 3. Calculate Weights
            # ... (No change)
            epsilon = 0.05 
            
            weights = []
            for c in candidates:
                w = c['qual'] / (c['d_sq'] + epsilon)
                weights.append(w)
                
            total_w = sum(weights)
            
            if total_w > 0:
                for i, c in enumerate(candidates):
                    mass_share = total_mass * (weights[i] / total_w)
                    target_idx = c['idx']
                    contact_masses[f, target_idx] += mass_share
            
            # --- Refine Foot/Ankle Balance (Post-Pass) ---
            # Pairs: Left (7=Ankle, 10=Foot), Right (8=Ankle, 11=Foot)
            for (idx_a, idx_f) in [(7, 10), (8, 11)]:
                m_a = contact_masses[f, idx_a]
                m_f = contact_masses[f, idx_f]
                total_pair = m_a + m_f
                
                if total_pair > 0.1: # Only if significant contact
                    # --- Graduated Fade Logic ---
                    
                    # 1. Base Weight (Lean / Flat Logic)
                    # We calculate this first as the "Flat" baseline.
                    plane_dims = [d for d in [0, 1, 2] if d != y_dim]
                    p_a = world_pos[f, idx_a, plane_dims]
                    p_f = world_pos[f, idx_f, plane_dims]
                    p_c = com[f, plane_dims]
                    
                    # Need heights for Bias logic later
                    h_a = world_pos[f, idx_a, y_dim]
                    h_f = world_pos[f, idx_f, y_dim]
                    
                    vec_af = p_f - p_a
                    len_sq = np.sum(vec_af**2)
                    
                    w_foot_base = 0.5 
                    
                    if len_sq > 0.0025: # > 5cm horizontal length
                         # Stable Projection
                        vec_ac = p_c - p_a
                        t = np.dot(vec_ac, vec_af) / len_sq
                        w_foot_base = np.clip(t, 0.0, 1.0)
                    else:
                         # Unstable Fallback
                         q_a = contact_mask[f, idx_a]
                         q_f = contact_mask[f, idx_f]
                         if q_a + q_f > 1e-4:
                             w_foot_base = q_f / (q_a + q_f)
                             
                    # 2. Pitch / Height Bias
                    # Determine Pitch
                    foot_pitch = 0.0
                    has_tip = False
                    if hasattr(self, 'temp_tips') and idx_f in self.temp_tips:
                        t_pos = self.temp_tips[idx_f][f]
                        v_ft = t_pos - world_pos[f, idx_f]
                        v_ft_h = np.linalg.norm([v_ft[d] for d in range(3) if d != y_dim])
                        if v_ft_h > 0.01:
                            foot_pitch = np.arctan2(v_ft[y_dim], v_ft_h) * (180.0 / np.pi)
                            has_tip = True
                            
                    # Calculate Biases
                    # Toe Bias (Target 1.0)
                    # Pitch: -15 (0%) -> -35 (100%)
                    # Ankle H: 2cm (0%) -> 7cm (100%)
                    toe_bias = 0.0
                    if has_tip:
                        p_factor = np.clip((foot_pitch - (-15.0))/(-35.0 - (-15.0)), 0.0, 1.0)
                        h_factor = np.clip((h_a - 0.02)/(0.07 - 0.02), 0.0, 1.0)
                        toe_bias = p_factor * h_factor
                        
                    # Heel Bias (Target 0.0)
                    # Pitch: 15 (0%) -> 35 (100%)
                    # Foot H: 2cm (0%) -> 7cm (100%)
                    heel_bias = 0.0
                    if has_tip:
                        p_factor = np.clip((foot_pitch - 15.0)/(35.0 - 15.0), 0.0, 1.0)
                        h_factor = np.clip((h_f - 0.02)/(0.07 - 0.02), 0.0, 1.0)
                        heel_bias = p_factor * h_factor
                        
                    # Blend
                    # If Biases overlap (unlikely), normalize? 
                    # Usually mutually exclusive (Pitch < 0 vs Pitch > 0).
                    w_foot = w_foot_base * (1.0 - toe_bias - heel_bias) + (1.0 * toe_bias) + (0.0 * heel_bias)
                    
                    # 3. Hover Guard
                    # If Pitch is "Flat-ish" (Low Bias), apply Hover Guard?
                    # "Flat-ish" roughly -25 to 25.
                    if (toe_bias < 0.1 and heel_bias < 0.1):
                        if min(h_a, h_f) > floor_height + 0.05:
                            contact_masses[f, idx_f] = 0.0
                            contact_masses[f, idx_a] = 0.0
                        else:
                            contact_masses[f, idx_f] = total_pair * w_foot
                            contact_masses[f, idx_a] = total_pair * (1.0 - w_foot)
                    else:
                        # Bias active (Toe or Heel Reach) -> Allow contact (Height guarded by Bias calc)
                        contact_masses[f, idx_f] = total_pair * w_foot
                        contact_masses[f, idx_a] = total_pair * (1.0 - w_foot)
                    
        # Update contact_mask to be mass
        contact_mask = contact_masses.astype(np.float32)

        # Update History (Store LAST frame tips)
        if hasattr(self, 'temp_tips') and self.temp_tips:
             # temp_tips is {idx: Array(F,3)}
             # We want {idx: Array(3)} (Last Frame)
             self.last_tip_positions = {k: v[-1].copy() for k, v in self.temp_tips.items()}

        # Output dict
        res = {
            'pose': output_quats if return_quats else pose_data_aa[:, :self.target_joint_count, :],
            'trans': trans_data,
            'torques': torques,
            'torques_vec': torques_vec,
            'torques_grav_vec': t_grav_vecs,
            'torques_dyn_vec': t_dyn_vecs,
            'inertias': inertias,
            'efforts_dyn': efforts_dyn,
            'efforts_grav': efforts_grav,
            'efforts_net': efforts_net,
            'positions': world_pos, # Added for floor contact analysis
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
