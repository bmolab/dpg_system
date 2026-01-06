import numpy as np
import pickle
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
try:
    from dpg_system.body_dynamics import BodySegment, EffortEstimator
except ImportError:
    from body_dynamics import BodySegment, EffortEstimator

# --- SMPL Constants ---
SMPL_JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 
    'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 
    'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist'
]

SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 
    3, 4, 5, 6, 7, 8, 
    9, 9, 9, 12, 13, 14, 
    16, 17, 18, 19
]

# --- Mass Ratios (Approximate) ---
# Source: Modified from Plagenhoef et al. (1983), de Leva (1996)
# Adapted for SMPL segments (Pelvis/Spines split)
# Normalized Sum ~ 1.0

# Male
MALE_MASS_RATIOS = {
    'Pelvis': 0.11, 'Spine1': 0.11, 'Spine2': 0.11, 'Spine3': 0.06, 
    'Neck': 0.024, 'Head': 0.066,
    'L_Hip': 0.11, 'R_Hip': 0.11, # Thighs
    'L_Knee': 0.045, 'R_Knee': 0.045, # Shins
    'L_Ankle': 0.013, 'R_Ankle': 0.013, # Feet
    'L_Foot': 0.0, 'R_Foot': 0.0, 
    'L_Collar': 0.018, 'R_Collar': 0.018, 
    'L_Shoulder': 0.026, 'R_Shoulder': 0.026, # Upper Arms
    'L_Elbow': 0.015, 'R_Elbow': 0.015, # Forearms
    'L_Wrist': 0.006, 'R_Wrist': 0.006, # Hands (Approximated)
}

# Female
FEMALE_MASS_RATIOS = {
    'Pelvis': 0.12, 'Spine1': 0.10, 'Spine2': 0.10, 'Spine3': 0.05, 
    'Neck': 0.022, 'Head': 0.055,
    'L_Hip': 0.115, 'R_Hip': 0.115, # Thighs
    'L_Knee': 0.046, 'R_Knee': 0.046, # Shins
    'L_Ankle': 0.011, 'R_Ankle': 0.011, # Feet
    'L_Foot': 0.0, 'R_Foot': 0.0,
    'L_Collar': 0.016, 'R_Collar': 0.016,
    'L_Shoulder': 0.024, 'R_Shoulder': 0.024, # Upper Arms
    'L_Elbow': 0.013, 'R_Elbow': 0.013, # Forearms
    'L_Wrist': 0.005, 'R_Wrist': 0.005, # Hands
}

GENDER_MAP = {'male': MALE_MASS_RATIOS, 'female': FEMALE_MASS_RATIOS, 'neutral': MALE_MASS_RATIOS}

class SMPLDynamicsModel:
    def __init__(self, total_mass=75.0, gender='neutral', betas=None, model_path=None):
        """
        Args:
            total_mass (float): Total body mass in kg.
            gender (str): 'male', 'female', or 'neutral'.
            betas (np.array, optional): (10,) or (16,) shape parameters.
            model_path (str, optional): Path to SMPL model file (.pkl).
        """
        self.total_mass = total_mass
        self.gender = gender.lower() if gender else 'neutral'
        self.segments = {}
        self.root = None
        self.offsets = np.zeros((22, 3))
        
        # Initialize Default Offsets (approx Mean Human)
        # Y-up approximation for T-pose
        self.offsets[1] = [0.07, -0.05, 0.0] # L_Hip (Approx)
        self.offsets[2] = [-0.07, -0.05, 0.0] # R_Hip
        self.offsets[3] = [0.0, 0.1, 0.0] # Spine1
        self.offsets[6] = [0.0, 0.1, 0.0] # Spine2
        self.offsets[9] = [0.0, 0.1, 0.0] # Spine3
        self.offsets[12] = [0.0, 0.05, 0.0] # Neck
        self.offsets[15] = [0.0, 0.1, 0.0] # Head
        # Limbs (Approx lengths)
        self.offsets[4] = [0.0, -0.4, 0.0] # L_Knee (From Hip)
        self.offsets[5] = [0.0, -0.4, 0.0] # R_Knee
        self.offsets[7] = [0.0, -0.4, 0.0] # L_Ankle
        self.offsets[8] = [0.0, -0.4, 0.0] # R_Ankle
        self.offsets[10] = [0.0, 0.0, 0.15] # L_Foot (Forward +Z)
        self.offsets[11] = [0.0, 0.0, 0.15] # R_Foot (Forward +Z)
        self.offsets[13] = [0.1, 0.1, 0.0] # L_Collar
        self.offsets[14] = [-0.1, 0.1, 0.0] # R_Collar
        self.offsets[16] = [0.1, 0.0, 0.0] # L_Shoulder
        self.offsets[17] = [-0.1, 0.0, 0.0] # R_Shoulder
        self.offsets[18] = [0.3, 0.0, 0.0] # L_Elbow (T-Pose Left +X)
        self.offsets[19] = [-0.3, 0.0, 0.0] # R_Elbow (T-Pose Right -X)
        self.offsets[20] = [0.25, 0.0, 0.0] # L_Wrist (Left +X)
        self.offsets[21] = [-0.25, 0.0, 0.0] # R_Wrist (Right -X)

        # 1. Compute Skeleton if model provider
        if model_path and betas is not None:
             computed_offsets = self._compute_skeleton_from_betas(model_path, betas)
             if computed_offsets is not None:
                 self.offsets = computed_offsets

        # Build Segment Hierarchy
        self._build_hierarchy()
        
        # Calibration Offset
        self.calibration_offset = np.eye(3)        # Rotation Matrix
        self.calibration_pos_offset = np.zeros(3)  # Translation Vector
        
        # EMA Smoothing State
        self.ema_pos = None     # Current smoothed Root Position
        self.ema_pose = None    # Current smoothed Pose (Axis-Angle)
        
        # State History for Realtime processing
        self.reset_state()
        
    def reset_state(self):
        """Reset internal state history (velocities/accelerations)."""
        self.history = {
            'pos': [], # List of (22, 3) arrays
            'rot': [], # List of (22, 3, 3) arrays
            'lin_vel': [], # List of (22, 3)
            'ang_vel': []  # List of (22, 3)
        }
        
    def update_frame(self, trans, pose, dt=1.0/60.0, smoothing_sigma=0.0):
        """
        Process a SINGLE frame in real-time.
        
        Args:
            trans: (3,) Root translation
            pose: (22, 3) or (66,) Axis-Angle pose
            dt: float, time since last frame
            smoothing_sigma: float, smoothing factor. 0.0=None.
                             Uses EMA: alpha = 1.0 / (1.0 + sigma)
            
        Returns:
            torques: dict
            metrics: dict
        """
        pose = pose.reshape(22, 3).copy() # Copy to avoid modifying input in-place
        
        # 0. Input Smoothing (EMA)
        if smoothing_sigma > 0.0:
            alpha = 1.0 / (1.0 + smoothing_sigma)
            
            if self.ema_pos is None:
                self.ema_pos = trans.copy()
                self.ema_pose = pose.copy()
            else:
                self.ema_pos = alpha * trans + (1.0 - alpha) * self.ema_pos
                self.ema_pose = alpha * pose + (1.0 - alpha) * self.ema_pose
                
            trans = self.ema_pos
            pose = self.ema_pose
        else:
             # Reset EMA if smoothing disabled to avoid lag when re-enabling?
             # Or just track. Let's update tracker to current even if disabled
             self.ema_pos = trans.copy()
             self.ema_pose = pose.copy()
             
        # 0.5 Apply Calibration Offset
        # Apply Translation Offset
        trans = trans + self.calibration_pos_offset
        
        # Apply to Root Orientation (pose[0])
        # Root Pose is Axis-Angle. Convert to Matrix, Apply, Convert back.
        # R_new = R_cal @ R_root
        if not np.array_equal(self.calibration_offset, np.eye(3)):
            root_rotvec = pose[0]
            root_mat = Rotation.from_rotvec(root_rotvec).as_matrix()
            new_root_mat = self.calibration_offset @ root_mat
            new_root_rotvec = Rotation.from_matrix(new_root_mat).as_rotvec()
            pose[0] = new_root_rotvec
        
        # 1. Forward Kinematics
        # We need batch dimension (1, 3)
        batch_trans = trans.reshape(1, 3)
        batch_pose = pose.reshape(1, 22, 3)
        
        curr_pos, curr_rot = self._batch_forward_kinematics(batch_trans, batch_pose, self.offsets)
        
        curr_pos = curr_pos[0] # (22, 3)
        curr_rot = curr_rot[0] # (22, 3, 3)
        
        # 2. Compute Derivatives (Finite Difference with History)
        # We need at least previous frame for Velocity.
        # We need 2 previous frames for Acceleration (or Prev Velocity).
        
        # Update History
        self.history['pos'].append(curr_pos)
        self.history['rot'].append(curr_rot)
        
        # Keep only necessary history (3 frames for central diff if we had latency, but for realtime we use backward)
        if len(self.history['pos']) > 3:
            self.history['pos'].pop(0)
            self.history['rot'].pop(0)
            
        # Calculate Velocity
        lin_vel = np.zeros((22, 3))
        ang_vel = np.zeros((22, 3))
        
        if len(self.history['pos']) >= 2:
            # v[t] = (p[t] - p[t-1]) / dt
            p_curr = self.history['pos'][-1]
            p_prev = self.history['pos'][-2]
            lin_vel = (p_curr - p_prev) / dt
            
            # Omega
            R_curr = self.history['rot'][-1]
            R_prev = self.history['rot'][-2]
            # R_curr = R_step @ R_prev => R_step = R_curr @ R_prev.T
            # R_prev is (22, 3, 3). We need (22, 3, 3) where each 3x3 is transposed.
            R_prev_T = R_prev.transpose(0, 2, 1)
            R_step = R_curr @ R_prev_T
            rot_vec = Rotation.from_matrix(R_step).as_rotvec()
            ang_vel = rot_vec / dt
        
        self.history['lin_vel'].append(lin_vel)
        self.history['ang_vel'].append(ang_vel)
        if len(self.history['lin_vel']) > 3:
            self.history['lin_vel'].pop(0)
            self.history['ang_vel'].pop(0)
            
        # Calculate Acceleration
        lin_acc = np.zeros((22, 3))
        ang_acc = np.zeros((22, 3))
        
        if len(self.history['lin_vel']) >= 2:
            v_curr = self.history['lin_vel'][-1]
            v_prev = self.history['lin_vel'][-2]
            lin_acc = (v_curr - v_prev) / dt
            
            w_curr = self.history['ang_vel'][-1]
            w_prev = self.history['ang_vel'][-2]
            ang_acc = (w_curr - w_prev) / dt
            
        # 3. Effort Estimator (PASS 1: Floating Base)
        estimator = EffortEstimator(self.root)
        
        state_dict = {}
        for i, name in enumerate(SMPL_JOINT_NAMES):
            state_dict[name] = {
                'position': curr_pos[i],
                'orientation': curr_rot[i],
                'linear_velocity': lin_vel[i],
                'linear_acceleration': lin_acc[i],
                'angular_velocity': ang_vel[i],
                'angular_acceleration': ang_acc[i]
            }
            
        estimator.set_kinematics(state_dict)
        torques_pass1 = estimator.calculate_torques()
        metrics_pass1 = estimator.calculate_whole_body_metrics()
        
        # 4. Floor Contact Handling (PASS 2)
        # Check Net Force at Root (Pelvis) from Pass 1
        # In floating base, Root Force = Total Force required to support body + Accelerate CoM
        # We want to shift this force to the Feet if they are in contact.
        
        root_segment = self.segments['Pelvis']
        total_grf_needed = root_segment.net_force.copy() # The force the "Air" is exerting on Pelvis
        # Wait, net_force on segment is m*a.
        # The recursion calculated f_prox (Force at Proximal Joint). For Root, this is the force holding it up.
        # However, `calculate_torques` returns torques. It computes forces internally but doesn't return root force easily unless we inspect root.
        
        # We need to access the Root's computed force from the estimator's pass.
        # In `_backward_pass`, `f_prox` is computed.
        # Let's inspect `root_segment.net_force`... No, `net_force` attribute is `m(a-g)`.
        # The PROXIMAL FORCE `f_prox` is what we need.
        # Currently `net_force` attr stores `m(a-g)`.
        # The recursive `f_prox` is not stored on the segment in my implementation, it's a return/accumulator.
        # Wait, the `calculate_torques` implementation usually returns `torques`.
        
        # Helper to get Root Force:
        # F_root_needed = Sum(m_i * (a_i - g)) for all segments.
        # This is exactly equal to `f_prox` at root.
        
        F_root_needed = np.zeros(3)
        for name, seg in self.segments.items():
             F_root_needed += seg.mass * (seg.linear_acceleration - estimator.gravity)
             
        # Detect Contacts
        # Candidates: Feet, Knees, Hands, Head, Spine, Pelvis
        # Note: Use 'L_Wrist'/'R_Wrist' for Hands
        contact_candidates = [
            'L_Foot', 'R_Foot', 
            'L_Ankle', 'R_Ankle', 
            'L_Knee', 'R_Knee', 'L_Wrist', 'R_Wrist', 'Head',
            'Pelvis', 'Spine1', 'Spine2', 'Spine3'
        ]
        active_contacts = []
        
        contact_thresh_height = 0.15 # Meters (Approx threshold)
        contact_thresh_vel = 0.2 # m/s
        
        metrics_contacts = {}
        
        for name in contact_candidates:
            if name in SMPL_JOINT_NAMES:
                idx = SMPL_JOINT_NAMES.index(name)
                pos = curr_pos[idx]
                vel = lin_vel[idx]
                
                # print(f"DEBUG: update_frame {name} Z={pos[2]:.4f}") # Debug print added here
                # Check Z Height and Velocity magnitude
                is_contact = (pos[2] < contact_thresh_height) and (np.linalg.norm(vel) < contact_thresh_vel)
                
                metrics_contacts[name] = bool(is_contact)
                if is_contact:
                    active_contacts.append(name)

        # Force Distribution
        # If any contacts, apply distributed F_root_needed as External Force
        
        if len(active_contacts) > 0:
            # We will re-run the estimator.
            
            # Simple Distribution: Split evenly
            force_per_contact = F_root_needed / len(active_contacts)
            
            for name in active_contacts:
                # Find the segment that corresponds to this joint name
                # Note: SMPL_JOINT_NAMES includes 'L_Foot', which maps to self.segments['L_Foot']
                # BUT 'L_Hand' is not in SMPL_JOINT_NAMES? 
                # Wait, SMPL_JOINT_NAMES: ..., 'L_Wrist', 'R_Wrist'. 
                # The 'Hand' usually refers to the Wrist or Finger tips. 
                # Let's map candidates to actual Joint Names.
                
                seg_name = name
                if name == 'L_Hand': seg_name = 'L_Wrist'
                elif name == 'R_Hand': seg_name = 'R_Wrist'
                
                if seg_name in self.segments:
                    seg = self.segments[seg_name]
                    seg.external_force = force_per_contact
                    
                    # Apply Compensating Torque to shift application point from CoM to Joint
                    # By default, external_force assumes application at CoM.
                    # We want it to act at the Joint (Proximal) to simulate bone-stacking.
                    # Torque = r_joint_from_com x Force
                    # r_joint_from_com_local = -seg.center_of_mass
                    r_joint_from_com_world = seg.orientation @ (-seg.center_of_mass)
                    compensating_torque = np.cross(r_joint_from_com_world, force_per_contact)
                    seg.external_torque = compensating_torque
                    # print(f"DEBUG: {name} Force={force_per_contact[2]:.1f} CoM={seg.center_of_mass} r_world={r_joint_from_com_world} CompTorque={compensating_torque}")
                 
            # Re-run Dynamics (Pass 2)
            # Torques will now reflect that the Load is supported by Contacts
            torques = estimator.calculate_torques()
            metrics = estimator.calculate_whole_body_metrics()
            
            # Add floor contact info to metrics
            metrics['contact_state'] = metrics_contacts
            metrics['contact_state']['distributed_force'] = F_root_needed
            metrics['contact_state']['active_count'] = len(active_contacts)
            
            return torques, metrics
            
        else:
            # Flying or Falling -> Pass 1 is correct
            metrics_pass1['contact_state'] = metrics_contacts
            metrics_pass1['contact_state']['distributed_force'] = np.zeros(3)
            metrics_pass1['contact_state']['active_count'] = 0
            
            return torques_pass1, metrics_pass1

    def calibrate_balance(self):
        """
        Calculate calibration offset to align Body CoM vertically with Ankles.
        Assumes the current state (in `self.segments`) is the 'Standing' pose to calibrate.
        """
        # 1. Calculate Body CoM
        total_mass = 0
        com_world = np.zeros(3)
        
        for segment in self.segments.values():
            com_world += segment.mass * segment._cached_p_cm_world
            total_mass += segment.mass
            
        if total_mass == 0: return # Should not happen
        
        com_world /= total_mass
        
        # 2. Calculate Center of Support (CoP)
        # Verify active contacts (Z < threshold)
        contact_points = []
        floor_threshold = 0.15 # Match update_frame threshold
        for seg_name, segment in self.segments.items():
            # print(f"DEBUG: {seg_name} Z={segment.position[2]:.4f}")
            if segment.position[2] < floor_threshold:
                contact_points.append(segment.position)
        
        if len(contact_points) > 0:
             # Align to Mean Contact Point (CoP with simple distribution)
             target_center = np.mean(contact_points, axis=0)
             print(f"Calibration Target: Center of Support (Active count: {len(contact_points)})")
        elif 'L_Ankle' in self.segments and 'R_Ankle' in self.segments:
            # Fallback to Ankles if no contact
            l_pos = self.segments['L_Ankle'].position
            r_pos = self.segments['R_Ankle'].position
            target_center = (l_pos + r_pos) / 2.0
            print("Calibration Target: Mean Ankle Position (Fallback)")
        else:
            print("Calibration Failed: No contacts or Ankles found.")
            return

        # 3. Calculate Offset Vector
        # We want CoM to be vertically aligned with Target Center.
        # Current Offset in Horizontal Plane:
        offset = com_world - target_center
        
        # Vector from Target to CoM
        v_current = com_world - target_center
        # Target vector: Vertical alignment
        v_target = np.array([0, 0, v_current[2]])
        
        # 4. Compute Rotation to align v_current to v_target
        # We create a rotation that aligns vector A to vector B.
        # Axis = cross(A, B)
        # Angle = acos(dot(A, B))
        
        # Normalize
        v_curr_n = v_current / np.linalg.norm(v_current)
        v_targ_n = v_target / np.linalg.norm(v_target)
        
        axis = np.cross(v_curr_n, v_targ_n)
        dot = np.dot(v_curr_n, v_targ_n)
        
        # Check singularities
        if np.linalg.norm(axis) < 1e-6:
            # Already aligned or opposite
            print("Calibration: Already aligned.")
            return
            
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        
        # Create Rotation Matrix
        R_correction = Rotation.from_rotvec(axis * angle).as_matrix()
        
        # Update Calibration Offset
        # We apply this to the Root.
        # Accumulate the new correction on top of the existing one.
        # Order: New Correction is applied to the already-rotated body.
        # But `calibration_offset` is applied to the raw input pose.
        # If R_new = R_corr @ R_current_world, and R_current_world = R_old_cal @ R_raw
        # Then R_new = R_corr @ R_old_cal @ R_raw
        # So new calibration offset is R_corr @ self.calibration_offset
        
        # Update Calibration Rotation
        # self.calibration_offset = R_correction @ self.calibration_offset (Accumulate)
        self.calibration_offset = R_correction @ self.calibration_offset
        
        # Update Calibration Translation
        # We need to shift the root so that the Pivot (CoP) remains stationary.
        # Original: Root -> CoP = v_root_to_cop
        # New: Root' -> CoP = R @ v_root_to_cop ? 
        # Actually: Pivot is fixed point P.
        # Root_new = P + R @ (Root_old - P)
        # Shift = Root_new - Root_old = P + R@(Root-P) - Root
        
        # Uses current root position (trans? no, com_world is based on current state)
        root_pos = self.segments['Pelvis'].position
        
        # Pivot point (target_center)
        P = target_center
        v_root_rel = root_pos - P
        
        root_pos_new = P + R_correction @ v_root_rel
        t_shift = root_pos_new - root_pos
        
        # Accumulate translation offset? 
        # If we previously shifted, 'root_pos' already includes it.
        # So we just add the NEW shift required by this NEW rotation.
        self.calibration_pos_offset += t_shift
        
        # CRITICAL: Reset history to prevent velocity spikes!
        self.reset_state()
        self.ema_pos = None
        self.ema_pose = None
        
        print(f"Calibration Applied: Shifted CoM by {np.degrees(angle):.2f} deg. Pivot Correction: {t_shift}")

    def _compute_skeleton_from_betas(self, model_path, betas):
        """
        Load SMPL model and compute joint locations J.
        Calclate offsets (bone vectors).
        """
        try:
            with open(model_path, 'rb') as f:
                dd = pickle.load(f, encoding='latin1')
                
            # Basic SMPL Linear Blend Shape logic:
            # v_shaped = v_template + shapedirs * betas
            # J = J_regressor * v_shaped
            
            # Need strict shapes
            v_template = dd['v_template'] # (6890, 3)
            shapedirs = dd['shapedirs'] # (6890, 3, 10) typically
            J_regressor = dd['J_regressor'] # (24, 6890) sparse usually
            
            # Handle beta count
            num_betas = shapedirs.shape[-1]
            if len(betas) > num_betas:
                betas = betas[:num_betas]
            elif len(betas) < num_betas:
                pad = np.zeros(num_betas - len(betas))
                betas = np.concatenate([betas, pad])
                
            # Compute v_shaped
            # shapedirs * betas -> (6890, 3)
            delta_v = np.einsum('ijk,k->ij', shapedirs, betas)
            v_shaped = v_template + delta_v
            
            # Compute J
            # J_regressor is often sparse (scipy.sparse.csc_matrix or similar)
            # Check type
            import scipy.sparse
            if scipy.sparse.issparse(J_regressor):
                J = J_regressor.dot(v_shaped) # (24, 3)
            else:
                J = np.dot(J_regressor, v_shaped)
            
            # Compute Offsets (Relative)
            # offset[i] = J[i] - J[parent[i]]
            offsets = np.zeros((22, 3))
            offsets[0] = np.zeros(3) # Root has no offset from itself (or world origin in local frame)
            
            for i in range(1, 22):
                parent = SMPL_PARENTS[i]
                offsets[i] = J[i] - J[parent]
                
            return offsets
            
        except Exception as e:
            print(f"Error computing skeleton from betas: {e}. Using default offsets.")
            return self.offsets # Fallback
        
    def _build_hierarchy(self):
        # Select Mass Ratios
        mass_ratios = GENDER_MAP.get(self.gender, MALE_MASS_RATIOS)
        total_ratio = sum(mass_ratios.values())
        
        # Create all segments first
        created_segments = {}
        
        for i, name in enumerate(SMPL_JOINT_NAMES):
            ratio = mass_ratios.get(name, 0.0)
            mass = (ratio / total_ratio) * self.total_mass
            
            # Simple Inertia approximation
            # Using actual bone length from offsets?
            # Offset is vector from parent to THIS joint.
            # But the segment extends from THIS joint to its CHILDREN.
            # We don't have a single "length" for branched segments (Spine).
            # We'll use the norm of the offset to the FIRST child as approx length, or 0.1 default.
            
            # This segment represents the bone starting at 'name' and going to its children.
            # Find children indices from SMPL_PARENTS
            children_indices = [idx for idx, p in enumerate(SMPL_PARENTS) if p == i]
            
            if len(children_indices) > 0:
                child_idx = children_indices[0] # Pick first child
                length = np.linalg.norm(self.offsets[child_idx])
            else:
                length = 0.1 # Default for end effectors (Foot, Head, Wrist)
            
            inertia = np.eye(3) * (0.1 * mass * length**2) # Simple Thin rod approx
            
            # CoM approximation: Halfway along the "main" child axis?
            if len(children_indices) > 0:
                 child_idx = children_indices[0]
                 com = self.offsets[child_idx] * 0.5 
                 # This offset to child is the LOCAL offset in T-Pose.
                 # BodySegment expects vector from proximal joint. This matches.
            else:
                 com = np.array([0, length*0.5, 0]) # Just along Y local? 
            
            # Determine ref_offset (Reference Offset to Child)
            # This is tricky because BodySegment supports multiple children, but ref_offset is singular?
            # Wait, my BodySegment update added `ref_offset` as a SINGLE property.
            # But the logic loops over children and uses `child.ref_offset`.
            # So `ref_offset` is "Offset from Parent to THIS segment".
            # YES. `ref_offset` belongs to the CHILD (describing its position relative to Parent).
            
            # self.offsets[i] IS the vector from Parent(i) to Joint(i).
            # So `ref_offset` for segment `i` is `self.offsets[i]`.
            
            segment = BodySegment(
                name=name,
                mass=mass,
                length=length, 
                center_of_mass=com, 
                inertia=inertia,
                ref_offset=self.offsets[i]
            )
            created_segments[i] = segment
            self.segments[name] = segment
        
        # Link Children
        for i, parent_idx in enumerate(SMPL_PARENTS):
            if parent_idx == -1:
                self.root = created_segments[i]
            else:
                parent = created_segments[parent_idx]
                child = created_segments[i]
                parent.children.append(child)

    def process_sequence(self, trans, poses, dt=None, smoothing_sigma=2.0):
        """
        Process a full sequence of SMPL poses to estimate dynamics.
        
        Args:
            trans (np.array): (N, 3) Root translation (World).
            poses (np.array): (N, 22, 3) Axis-Angle poses.
            dt (float, optional): Time step. If None, assumes 60Hz (0.016s).
            smoothing_sigma (float): Sigma for Gaussian smoothing of input motion. 
                                     Set 0 to disable. Standard deviation in frames.
                                     Default 2.0 (approx 25ms at 60Hz, 16ms at 120Hz).
        
        Returns:
            dict containing:
            - 'torques': list of dicts (one per frame)
            - 'metrics': list of dicts (one per frame)
            - 'global_positions': (N, 22, 3)
            - 'global_orientations': (N, 22, 3, 3)
        """
        N = trans.shape[0]
        if dt is None:
            dt = 1.0 / 60.0
            
        # Optional: Smooth Inputs (Critical for clean derivatives)
        if smoothing_sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            trans = gaussian_filter1d(trans, sigma=smoothing_sigma, axis=0)
            poses = gaussian_filter1d(poses, sigma=smoothing_sigma, axis=0)

        # 0.5 Apply Calibration Offset
        # Apply Calibration Translation
        trans = trans + self.calibration_pos_offset
        
        # Apply to Root Orientation (poses[:, 0]) which is Axis-Angle.
        if not np.array_equal(self.calibration_offset, np.eye(3)):
            # Convert Root Poses to Matrices
            # poses[:, 0] is (N, 3)
            root_rotvecs = poses[:, 0]
            root_mats = Rotation.from_rotvec(root_rotvecs).as_matrix() # (N, 3, 3)
            
            # Apply R_new = R_cal @ R_root
            # Broadcasting: (3,3) @ (N, 3, 3) -> (N, 3, 3)
            new_root_mats = np.matmul(self.calibration_offset, root_mats)
            
            # Convert back
            new_root_rotvecs = Rotation.from_matrix(new_root_mats).as_rotvec()
            poses[:, 0] = new_root_rotvecs

        # 1. Batch Forward Kinematics (positions/rotations for all frames)
        # Result: (N, 22, 3) and (N, 22, 3, 3)
        global_pos, global_rot = self._batch_forward_kinematics(trans, poses, self.offsets)
        
        # Finite Differences
        global_lin_vel = np.gradient(global_pos, dt, axis=0) # (N, 22, 3)
        global_lin_acc = np.gradient(global_lin_vel, dt, axis=0)
        
        global_ang_vel = self._compute_angular_velocity(global_rot, dt)
        global_ang_acc = np.gradient(global_ang_vel, dt, axis=0)
        
        results = {
            'torques': [],
            'metrics': [],
            'global_positions': global_pos,
            'global_orientations': global_rot,
            'global_linear_velocity': global_lin_vel,
            'global_linear_acceleration': global_lin_acc
        }
        
        # 3. Compute Dynamics per frame
        estimator = EffortEstimator(self.root)
        
        for t in range(N):
            # Build state dict for this frame
            state_dict = {}
            for i, name in enumerate(SMPL_JOINT_NAMES):
                state_dict[name] = {
                    'position': global_pos[t, i],
                    'orientation': global_rot[t, i],
                    'linear_velocity': global_lin_vel[t, i], 
                    'linear_acceleration': global_lin_acc[t, i], 
                    'angular_velocity': global_ang_vel[t, i],
                    'angular_acceleration': global_ang_acc[t, i]
                }
            
            estimator.set_kinematics(state_dict)
            torques = estimator.calculate_torques()
            metrics = estimator.calculate_whole_body_metrics()
            
            results['torques'].append(torques)
            results['metrics'].append(metrics)
            
        return results

    def _batch_forward_kinematics(self, trans, poses, offsets):
        """
        Args:
            trans: (N, 3)
            poses: (N, 22, 3) (Axis-Angle)
            offsets: (22, 3) (Relative bone vectors)
        returns:
            global_pos: (N, 22, 3)
            global_rot: (N, 22, 3, 3) matrix
        """
        N = trans.shape[0]
        J = 22
        
        global_pos = np.zeros((N, J, 3))
        global_rot = np.zeros((N, J, 3, 3))
        
        # Convert all poses to Matrices first
        # Flatten: (N*22, 3)
        flat_poses = poses.reshape(-1, 3)
        flat_mats = Rotation.from_rotvec(flat_poses).as_matrix()
        local_mats = flat_mats.reshape(N, J, 3, 3)
        
        # Iterate via hierarchy
        # Root (0)
        # Root Orientation (Pelvis) is usually Global in SMPL
        global_rot[:, 0] = local_mats[:, 0]
        global_pos[:, 0] = trans
        
        for i in range(1, J):
            parent = SMPL_PARENTS[i]
            # Global Rot = Parent Global Rot @ Local Rot
            global_rot[:, i] = np.matmul(global_rot[:, parent], local_mats[:, i])
            
            # Global Pos = Parent Global Pos + Parent Global Rot @ Offset
            # Offset is constant (22, 3) -> (1, 3)
            # We need to rotate offset by parent rotation for N frames.
            # parent_rot: (N, 3, 3)
            # offset: (3,)
            # rot_offset: (N, 3)
            rot_offset = np.einsum('nij,j->ni', global_rot[:, parent], offsets[i])
            global_pos[:, i] = global_pos[:, parent] + rot_offset
            
        return global_pos, global_rot

    def _compute_angular_velocity(self, global_rot, dt):
        N, J, _, _ = global_rot.shape
        omega = np.zeros((N, J, 3))
        
        # Forward difference orientation
        R_curr = global_rot[:-1]
        R_next = global_rot[1:]
        
        # R_next = R_step @ R_curr => R_step = R_next @ R_curr.T
        R_step = np.matmul(R_next, R_curr.transpose(0, 1, 3, 2))
        
        # Convert to rotvec
        flat_step = R_step.reshape(-1, 3, 3)
        step_vec = Rotation.from_matrix(flat_step).as_rotvec()
        step_vec = step_vec.reshape(N-1, J, 3)
        
        # Assign velocities
        omega[:-1] = step_vec / dt
        omega[-1] = omega[-2] # Repeat last
        
        return omega
    
def load_amass_sequence(npz_path):
    """
    Load AMASS sequence from .npz file.
    Returns:
        trans: (N, 3)
        poses: (N, 22, 3)
        dt: float (1.0 / mocap_framerate)
        betas: (16,)
        gender: str
    """
    data = np.load(npz_path)
    trans = data['trans']
    poses = data['poses'][:, :66].reshape(-1, 22, 3)
    
    # Framerate
    if 'mocap_framerate' in data:
        framerate = float(data['mocap_framerate'])
        dt = 1.0 / framerate
    else:
        dt = 1.0 / 60.0 # Default
        
    # Betas
    betas = data['betas'] if 'betas' in data else np.zeros(16)
    
    # Gender (AMASS files usually denote gender in metadata or filename, but npz usually has 'gender' field)
    gender = str(data['gender']) if 'gender' in data else 'neutral'
    
    return trans, poses, dt, betas, gender

