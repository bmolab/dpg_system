
import numpy as np
from scipy.spatial.transform import Rotation

class BodySegment:
    def __init__(self, name, mass, length, center_of_mass, inertia, children=None, ref_offset=None):
        """
        Initialize a body segment.

        Args:
            name (str): Name of the segment.
            mass (float): Mass of the segment in kg.
            length (float): Length of the segment in meters (proximal to distal joint).
            center_of_mass (array-like): vector relative to proximal joint (3,).
            inertia (array-like): Inertia tensor (3x3).
            children (list, optional): List of child BodySegment objects. Defaults to None.
            ref_offset (array-like): Vector from Parent Joint to This Joint in Parent's rest frame.
        """
        self.name = name
        self.mass = mass
        self.length = length
        self.center_of_mass = np.array(center_of_mass)
        self.inertia = np.array(inertia)
        self.children = children if children else []
        self.ref_offset = np.array(ref_offset) if ref_offset is not None else np.array([0, length, 0])
        
        # State variables (will be updated during kinematic pass)
        self.position = np.zeros(3) # Position of proximal joint
        self.orientation = np.eye(3) # Orientation of segment
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.linear_acceleration = np.zeros(3)
        self.angular_acceleration = np.zeros(3)
        
        # Computed forces and torques (at proximal joint)
        self.net_force = np.zeros(3)
        self.net_torque = np.zeros(3)
        self._cached_p_cm_world = np.zeros(3)
        self.net_torque = np.zeros(3)
        self.external_force = np.zeros(3) # Force acting ON the segment at Center of Mass (e.g., GRF)
        self.external_torque = np.zeros(3)
        self._cached_p_cm_world = np.zeros(3)
        self._cached_v_cm_world = np.zeros(3)

class EffortEstimator:
    def __init__(self, root_segment, gravity=9.81):
        """
        Initialize the effort estimator.

        Args:
            root_segment (BodySegment): The root of the kinematic tree.
            gravity (float): Magnitude of gravity.
        """
        self.root = root_segment
        self.gravity = np.array([0, 0, -gravity])

    def set_kinematics(self, segment_states):
        """
        Updates the kinematic state of the body.
        
        Args:
           segment_states (dict): Dictionary mapping segment names to their state dicts.
                                  State dict should contain:
                                  - 'position': np.array (3,)
                                  - 'orientation': np.array (3,3) OR (4,) quaternion OR (3,) axis-angle
                                  - 'linear_velocity': np.array (3,)
                                  - 'angular_velocity': np.array (3,)
                                  - 'angular_acceleration': np.array (3,)
                                  - 'linear_acceleration': np.array (3,) (of the CM or joint)
        
        Note: For a full physics engine, we propagates kinematics from root to leaves. 
        Here, we assume the user provides the full kinematic state for each segment 
        (e.g., derived from motion capture data via finite differences).
        """
        # In a real RNEA, we propagate kinematics. 
        # Since the user has "kinematic motion capture data", we assume we can compute
        # world-frame velocities and accelerations for each segment beforehand or here.
        
        # For this simplified version, let's assume `segment_states` provides 
        # the essential world-frame kinematic derivatives for the Center of Mass (CoM).
        
        self._populate_state_recursive(self.root, segment_states)

    def _ensure_orientation_matrix(self, orientation_input):
        """
        Convert various orientation formats to a 3x3 Rotation Matrix.
        Supports:
        - Rotation Matrix (3, 3)
        - Quaternion (4,) [x, y, z, w]
        - Rotation Vector/Axis-Angle (3,)
        """
        orientation_input = np.array(orientation_input)
        if orientation_input.shape == (3, 3):
            return orientation_input
        elif orientation_input.shape == (4,):
            # Assume scalar-last [x, y, z, w]
            return Rotation.from_quat(orientation_input).as_matrix()
        elif orientation_input.shape == (3,):
            # Assume rotation vector
            return Rotation.from_rotvec(orientation_input).as_matrix()
        else:
            raise ValueError(f"Unsupported orientation shape: {orientation_input.shape}")

    def _populate_state_recursive(self, segment, segment_states):
        if segment.name in segment_states:
            state = segment_states[segment.name]
            segment.position = state.get('position', np.zeros(3))
            
            # Flexible Orientation Handling
            raw_orientation = state.get('orientation', np.eye(3))
            segment.orientation = self._ensure_orientation_matrix(raw_orientation)
            
            segment.linear_velocity = state.get('linear_velocity', np.zeros(3)) # Velocity of Proximal Joint
            segment.angular_velocity = state.get('angular_velocity', np.zeros(3))
            segment.angular_acceleration = state.get('angular_acceleration', np.zeros(3))
            segment.linear_acceleration = state.get('linear_acceleration', np.zeros(3)) # Acceleration of CoM
            
            # Reset External Forces (User must set them manually per frame if needed, or we assume zero reset)
            # Actually, let's reset them here to ensure clean slate if not using 2-pass
            segment.external_force = np.zeros(3) 
            segment.external_torque = np.zeros(3)
        
        for child in segment.children:
            self._populate_state_recursive(child, segment_states)

    def calculate_torques(self):
        """
        Compute joint torques using Backward Recursive Newton-Euler.
        Returns a dictionary mapping segment names to the torque at their proximal joints.
        """
        torques = {}
        self._backward_pass(self.root, torques)
        return torques

    def _backward_pass(self, segment, torques_dict):
        # 1. Compute Net Force/Torque required for the motion of THIS segment alone
        # F_net = m * a_cm
        # Tau_net = I * alpha + omega x (I * omega)
        
        F_net = segment.mass * (segment.linear_acceleration - self.gravity) # Gravity is an acceleration
        
        # Euler's equation for rotation
        I_world = segment.orientation @ segment.inertia @ segment.orientation.T
        omega = segment.angular_velocity
        alpha = segment.angular_acceleration
        
        Tau_net = I_world @ alpha + np.cross(omega, I_world @ omega)
        
        # 2. Add forces/torques from children (Reaction forces)
        for child in segment.children:
            self._backward_pass(child, torques_dict)
            
            # Force exerted BY child ON parent at the child's proximal joint
            # This is -Force exerted BY parent ON child
            # But the recursive relations usually sum the forces acting ON the segment.
            # Forces on Segment = Force_parent + Sum(-Force_child) + F_external + Gravity
            # F_parent = F_net + Sum(F_child) - F_external - Gravity
            # Note: My F_net calculation above: m(a-g). This already includes -Gravity term effectively (m*a = F_ext + mg -> F_ext = m(a-g)).
            
            # The Equation of Motion:
            # F_total_acc = m * a_cm
            # Forces providing this:
            # F_proximal + F_external (GRF) + m*g - Sum(F_child_reaction) = m * a_cm
            
            # Therefore:
            # F_proximal = m * a_cm - m*g + Sum(F_child_reaction) - F_external
            # F_proximal = F_inertial_net + Sum(F_child_reaction) - F_external
            
            # RNEA Standard:
            # f_i = f_child + F_inertial - f_ext
            
            # Let's stick to: f_i = F_i + sum(f_{i+1})
            # where f_i is force at proximal joint, F_i is net inertial force
            
            F_net += child.net_force
            
            # Torque at proximal joint:
            # tau_i = Tau_i + sum(tau_{i+1}) + (r_i,cm x F_i) + sum(r_{i, i+1} x f_{i+1})
            # Wait, standard RNEA formulation (referencing Featherstone or similar):
            # f_i = I_i*a_i + v_i x (I_i*v_i) (spatial notation)
            # here we have decoupled linear and angular.
            
            # Torque balance at Segment Center of Mass:
            # Sum(Moments) = I * alpha + w x Iw
            # Moments = Tau_proximal - Sum(Tau_distal) + r_prox_cm x F_prox - Sum(r_dist_cm x F_dist)
            
            # We want Tau_proximal.
            # Tau_proximal = Tau_net + Sum(Tau_child_proximal) 
            #                - (vector from CoM to Proximal Joint) x F_proximal ??? NO
            
            # Easier Approach: Balance moments around the Proximal Joint.
            # M_prox = d(H)/dt + v_prox x P  ... effectively
            
            # Let's use the recursive relation:
            # f_joint = F_inertial + sum(f_child_joint)
            # n_joint = N_inertial + sum(n_child_joint) + (c_i x F_inertial) + sum(r_child_joint x f_child_joint)
            # where c_i is vector from Proximal Joint to CoM
            # and r_child_joint is vector from Proximal Joint to Child Joint
            
            reaction_force_child = child.net_force
            reaction_torque_child = child.net_torque
            
            # Vector from Segment Prox joint to Child Prox joint
            # Use the defined skeletal offset `ref_offset` of the child.
            # ref_offset is vector in Parent(segment) frame.
            # Transform to World Frame using current segment orientation.
            r_child_joint = segment.orientation @ child.ref_offset
            
            # Legacy Note: Previously we assumed [0, length, 0], which ignored X/Z offsets.
            # dist_to_child is just this vector.
            dist_to_child = r_child_joint
            
            Tau_net += reaction_torque_child + np.cross(dist_to_child, reaction_force_child)

        # 3. Add moment due to the inertial force at CoM acting at a distance from Proximal Joint
        # n_joint = ... + (c_i x F_inertial)
        # F_inertial includes gravity effect if we treated g as acc.
        # F_net above was m(a-g).
        
        c_i = segment.orientation @ segment.center_of_mass # Vector prom Proximal Joint to CoM in World Frame
        F_inertial = segment.mass * (segment.linear_acceleration - self.gravity) # Total force required for this segment
        
        # The torque at the joint must support this inertial force (moment arm c_i)
        # plus pure inertial torque Tau_net (I alpha...) 
        # plus the children.
        
        # Wait, I accumulated children into F_net and Tau_net already?
        # Re-calc cleanly:
        
        # Force required at proximal joint to support THIS segment and CHILDREN
        # f_prox = F_inertial_this + Sum(f_child_prox)
        # f_prox = F_inertial_this + Sum(f_child_prox) - F_external
        f_prox = F_inertial.copy() - segment.external_force
        for child in segment.children:
            f_prox += child.net_force
        
        # Moment required at proximal joint
        # n_prox = N_inertial_this + (c_i x F_inertial_this) 
        #          + Sum(n_child_prox + r_child_prox x f_child_prox)
        
        N_inertial_this = I_world @ alpha + np.cross(omega, I_world @ omega)
        
        # n_prox = ... - n_external - (c_i x F_external)
        # Assuming F_external acts at CoM.
        # If F_external acts at a specific point P_contact, we need r_contact x F_external.
        # For now, let's assume F_external is at CoM (simplification for single segment GRF).
        # Actually for Foot, GRF is at CoP. But we can assume the user puts the equivalent torque in external_torque.
        
        n_prox = N_inertial_this.copy() - segment.external_torque
        
        # Moment due to inertial force (acting at CoM)
        n_prox += np.cross(c_i, F_inertial)
        
        # Moment due to External Force (acting at CoM) -> Vector from Prox to CoM is c_i
        # Force exerted BY external world ON segment.
        # Moment at Proximal Joint due to F_external at CoM:
        # M_ext = c_i x F_external
        # Only if we are checking equilibrium.
        
        # Eq: n_prox + c_i x F_ext + n_ext - sum(child terms) = I alpha + w x Iw
        # n_prox = (I alpha...) - n_ext - c_i x F_ext + sum(child terms...) - c_i x (m(a-g))?
        # NO.
        
        # Consistent RNEA:
        # n_i = N_i + sum(n_i+1) + c_i x F_i + sum(r_i,i+1 x f_i+1) ... wait.
        
        # Let's trust the Force balance:
        # F_prox = F_inertial + sum(F_child) - F_ext
        
        # Moment Balance around Proximal Joint P:
        # Sum(M_around_P) = Rate of Change of Angular Momentum around P
        # M_prox + M_ext + (c_i x (m*g)) - Sum(...) = ...
        
        # Let's stick to the subtraction logic:
        # We need to provide the torque to Overcome Inertia AND Overcome External Forces.
        # If Ground pushes UP, we need LESS torque to hold it up?
        # Yes. F_ext helps.
        
        # So minus sign is correct.
        
        # Moment arm for F_external assuming it's at CoM:
        n_prox -= np.cross(c_i, segment.external_force)
        
        for child in segment.children:
            # Vector from Segment Proximal Joint to Child Proximal Joint
            # Ideally this offset is a property of the link.
            # Used to be: r_child = segment.orientation @ np.array([0, segment.length, 0])
            
            # New Method: Use ref_offset (Vector from Parent to Child in Rest Frame)
            # This rotates with the parent segment.
            r_child = segment.orientation @ child.ref_offset

            n_prox += child.net_torque + np.cross(r_child, child.net_force)

        # Store results
        segment.net_force = f_prox
        segment.net_torque = n_prox
        torques_dict[segment.name] = n_prox

    def calculate_whole_body_metrics(self):
        """
        Calculate whole-body metrics: CoM, Linear Momentum, Angular Momentum.
        Returns:
            dict with keys:
            - 'total_mass'
            - 'com_position' (world frame)
            - 'com_velocity' (world frame)
            - 'linear_momentum'
            - 'angular_momentum' (about CoM)
        """
        metrics = {
            'total_mass': 0.0,
            'com_position': np.zeros(3),
            'com_velocity': np.zeros(3),
            'linear_momentum': np.zeros(3),
            'angular_momentum': np.zeros(3)
        }
        
        segments = []
        self._collect_segments(self.root, segments)
        
        total_mass = 0.0
        weighted_pos = np.zeros(3)
        weighted_vel = np.zeros(3)
        
        # 1. Calculate Whole Body CoM and Linear Momentum
        for seg in segments:
            total_mass += seg.mass
            # Center of mass in world frame
            # seg.center_of_mass is relative to proximal joint. 
            # We need absolute CoM position.
            # Assuming seg.position is Proximal Joint Position (World).
            # And seg.orientation is World Orientation.
            
            p_cm_world = seg.position + seg.orientation @ seg.center_of_mass
            
            # Velocity of CoM in world frame
            # v_cm = v_prox + omega x r_prox_cm
            # Wait, our `set_kinematics` assumes we know kinematics.
            # But `BodySegment` fields `linear_velocity` usually refers to... joint?
            # Let's check `set_kinematics`. 
            # Ah, I didn't populate `linear_velocity` in `_populate_state_recursive` explicitly for joint or CoM?
            # `linear_acceleration` was used.
            # Let's assume user provides `linear_velocity` in state dict for the PROXIMAL JOINT.
            # Need to update _populate_state_recursive to accept/store linear_velocity if available.
            
            # Since I am just extending, I will calculate v_cm assuming `linear_velocity` is Proximal Joint Velocity.
            # If it's missing, default to 0.
            
            v_prox = seg.linear_velocity
            omega = seg.angular_velocity
            r_prox_cm = seg.orientation @ seg.center_of_mass
            v_cm_world = v_prox + np.cross(omega, r_prox_cm)
            
            weighted_pos += seg.mass * p_cm_world
            weighted_vel += seg.mass * v_cm_world
            
            # Store calculated world CoM properties on segment for step 2
            seg._cached_p_cm_world = p_cm_world
            seg._cached_v_cm_world = v_cm_world

        if total_mass > 0:
            com_pos = weighted_pos / total_mass
            com_vel = weighted_vel / total_mass
        else:
            com_pos = np.zeros(3)
            com_vel = np.zeros(3)
            
        metrics['total_mass'] = total_mass
        metrics['com_position'] = com_pos
        metrics['com_velocity'] = com_vel
        metrics['linear_momentum'] = total_mass * com_vel
        
        # 2. Calculate Angular Momentum about Whole Body CoM
        # H_G = Sum( I_i * omega_i + m_i * (p_i - P_G) x (v_i - V_G) )
        
        H_G = np.zeros(3)
        
        for seg in segments:
            p_i = seg._cached_p_cm_world
            v_i = seg._cached_v_cm_world
            
            I_world = seg.orientation @ seg.inertia @ seg.orientation.T
            
            H_local = I_world @ seg.angular_velocity # Spin
            H_orbital = np.cross(p_i - com_pos, v_i - com_vel) * seg.mass
            
            H_G += H_local + H_orbital
            
        metrics['angular_momentum'] = H_G
        
        return metrics

    def _collect_segments(self, segment, list_):
        list_.append(segment)
        for child in segment.children:
            self._collect_segments(child, list_)

