"""
Consensus Contact Detection System

Multi-factor approach combining:
1. Sensory evidence (height, velocity)
2. Structural analysis (what contacts are required by the pose?)
3. Dynamic analysis (is the body falling, accelerating?)
4. Torque plausibility (do the resulting torques make physical sense?)

These factors feed into an iterative consensus loop to produce
physically consistent contact probabilities.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────
# SMPL joint indices
# ─────────────────────────────────────────────────────────────────────
JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
]

# Parent indices for kinematic tree  (index → parent, -1 = root)
PARENTS = [
    -1,  0,  0,  0,  1,  2,
     3,  4,  5,  6,  7,  8,
     9,  9,  9, 12, 13, 14,
    16, 17, 18, 19, 20, 21
]

# Virtual joints appended after 24
VIRTUAL_JOINTS = {
    24: ('left_toe',  10),  # child of left_foot
    25: ('right_toe', 11),  # child of right_foot
    26: ('left_finger', 22),
    27: ('right_finger', 23),
    28: ('left_heel', 7),
    29: ('right_heel', 8),
}

# All joints that could plausibly make contact with the ground.
# Excludes head (15), neck (12), collar (13,14), spine3 (9) — unlikely contacts.
# Includes virtual toe/heel joints.
CONTACT_CANDIDATES = {
    0,                      # pelvis (sitting)
    1, 2,                   # hips (lying)
    3, 6,                   # spine1, spine2 (lying on back)
    4, 5,                   # knees (kneeling)
    7, 8,                   # ankles
    10, 11,                 # feet
    16, 17,                 # shoulders (lying on side)
    18, 19,                 # elbows (propped up)
    20, 21,                 # wrists
    22, 23,                 # hands
    24, 25,                 # toes (virtual)
    28, 29,                 # heels (virtual)
}

# Segment mass mapping (joint index → which limb_masses key)
SEGMENT_MASS_MAP = {
    0: 'pelvis',
    1: 'upper_leg', 2: 'upper_leg',
    3: 'spine', 6: 'spine',
    4: 'lower_leg', 5: 'lower_leg',
    7: 'foot', 8: 'foot',
    9: 'spine',
    10: 'foot', 11: 'foot',   # foot tip area (small fraction)
    12: 'head',                # neck
    13: 'upper_arm', 14: 'upper_arm',  # collar
    15: 'head',
    16: 'upper_arm', 17: 'upper_arm',
    18: 'lower_arm', 19: 'lower_arm',
    20: 'hand', 21: 'hand',
    22: 'hand', 23: 'hand',
}


@dataclass
class ConsensusOptions:
    """Configuration for consensus contact detection."""
    floor_height: float = 0.0
    
    # Sensory prior parameters
    height_sigma: float = 0.05       # Height Gaussian σ (meters)
    height_ceiling: float = 0.30     # Above this, p_height ≈ 0 (default for extremities)
    velocity_scale: float = 5.0      # Exponential decay rate for upward velocity
    
    # Per-joint height overrides (joint_index → (ceiling, sigma))
    # Pelvis: seated height can be 0.4-0.6m, so needs a much higher ceiling
    # Knees: kneeling puts them at ~0.15-0.35m
    # Elbows: propping up puts them at ~0.2-0.4m
    pelvis_height_ceiling: float = 0.60
    pelvis_height_sigma: float = 0.15
    knee_height_ceiling: float = 0.40
    knee_height_sigma: float = 0.04   # Tight: knee needs to be very close to contact
    elbow_height_ceiling: float = 0.40
    elbow_height_sigma: float = 0.08
    
    # Structural analysis
    structural_weight: float = 0.5   # How much structural analysis influences result
    
    # Dynamic analysis  
    dynamic_weight: float = 0.3      # How much dynamics influences result
    fall_threshold: float = 1.0      # CoM downward acceleration threshold (m/s²)
    
    # Torque plausibility
    plausibility_weight: float = 0.2
    torque_excess_penalty: float = 0.1  # Per N·m of excess torque
    
    # Consensus
    max_iterations: int = 2
    convergence_threshold: float = 0.01
    
    # Smoothing
    temporal_alpha: float = 0.2      # EMA smoothing (20% new, 80% old)
    
    # Y-axis index (for height)
    up_axis: int = 1  # Y-up


class SensoryPrior:
    """
    Computes per-joint contact probability from direct sensory evidence:
    height above floor, vertical velocity, chain geometry, and horizontal velocity.
    
    Enhancements over simple height gating:
    - Chain suppression: a knee above its ankle can't be a contact point
    - Horizontal velocity: a fast-moving swinging foot shouldn't trigger contact
    """
    
    # Which child joint(s) must be BELOW a parent for the parent to be a valid contact.
    # Format: parent_joint → (list of child joints, height_margin_meters)
    # The margin accounts for anatomical offset between the joint center
    # and the actual contact surface. Parent is only suppressed if it's
    # MORE THAN margin above the lowest child.
    CHAIN_CHILDREN = {
        0:  ([1, 2],       0.15),  # pelvis: buttocks extend ~15cm below joint center
        1:  ([4],          0.02),  # L_hip → L_knee (lying detection)
        2:  ([5],          0.02),  # R_hip → R_knee
        4:  ([7, 10, 28],  0.02),  # L_knee: kneecap close to joint center
        5:  ([8, 11, 29],  0.02),  # R_knee: kneecap close to joint center
        7:  ([10, 28],     0.08),  # L_ankle: heel is the real contact; ~8cm offset
        8:  ([11, 29],     0.08),  # R_ankle: heel is the real contact; ~8cm offset
        16: ([18],         0.02),  # L_shoulder → L_elbow
        17: ([19],         0.02),  # R_shoulder → R_elbow
        18: ([20, 22],     0.05),  # L_elbow: olecranon ~5cm offset
        19: ([21, 23],     0.05),  # R_elbow: olecranon ~5cm offset
        20: ([22],         0.03),  # L_wrist: small offset
        21: ([23],         0.03),  # R_wrist: small offset
    }
    
    def __init__(self, num_joints: int = 30):
        self.num_joints = num_joints
        self._prev_heights = None
        self._prev_positions = None  # (J, 3) for horizontal velocity
        self._smooth_velocity = None
        self._smooth_horiz_vel = None  # (J,) horizontal speed
        self._lift_state = None  # Per-joint: True if lifted
        self._lift_frames = None  # How many frames in current state
    
    def reset(self):
        self._prev_heights = None
        self._prev_positions = None
        self._smooth_velocity = None
        self._smooth_horiz_vel = None
        self._lift_state = np.zeros(self.num_joints, dtype=bool)
        self._lift_frames = np.zeros(self.num_joints, dtype=int)
    
    def compute(self, world_pos: np.ndarray, dt: float, 
                options: ConsensusOptions) -> np.ndarray:
        """
        Compute sensory contact probability for each joint.
        
        Args:
            world_pos: (J, 3) joint positions in world space
            dt: time step
            options: ConsensusOptions
            
        Returns:
            p_sensory: (J,) probability per joint based on height + velocity
        """
        J = world_pos.shape[0]
        up = options.up_axis
        floor = options.floor_height
        
        heights = world_pos[:, up] - floor
        
        # ── Per-joint height ceilings and sigmas ──
        ceilings = np.full(J, options.height_ceiling)
        sigmas = np.full(J, options.height_sigma)
        
        # Pelvis (j=0)
        if J > 0:
            ceilings[0] = options.pelvis_height_ceiling
            sigmas[0] = options.pelvis_height_sigma
        
        # Knees (j=4, 5)
        for j in [4, 5]:
            if j < J:
                ceilings[j] = options.knee_height_ceiling
                sigmas[j] = options.knee_height_sigma
        
        # Elbows (j=18, 19)
        for j in [18, 19]:
            if j < J:
                ceilings[j] = options.elbow_height_ceiling
                sigmas[j] = options.elbow_height_sigma
        
        # ── Height probability ──
        h_clamped = np.clip(heights, 0, ceilings)
        p_height = np.exp(-0.5 * (h_clamped / sigmas) ** 2)
        p_height[heights > ceilings] = 0.0
        # Below-floor joints are very likely in contact, but cap at 0.95
        # rather than 1.0 — floor position is a soft prior, not ground truth.
        # This allows velocity/dynamics signals to override height evidence.
        p_height[heights < 0] = 0.95
        
        # ── Horizontal plane dims ──
        if up == 1:
            plane_dims = [0, 2]
        elif up == 2:
            plane_dims = [0, 1]
        else:
            plane_dims = [1, 2]
        
        # ── Velocity estimation (vertical + horizontal) ──
        if self._prev_heights is None:
            self._prev_heights = heights.copy()
            self._prev_positions = world_pos.copy()
            self._smooth_velocity = np.zeros(J)
            self._smooth_horiz_vel = np.zeros(J)
            self._lift_state = np.zeros(J, dtype=bool)
            self._lift_frames = np.zeros(J, dtype=int)
        
        raw_velocity = (heights - self._prev_heights) / max(dt, 1e-6)
        vel_alpha = 0.3
        self._smooth_velocity = (self._smooth_velocity * (1 - vel_alpha) + 
                                  raw_velocity * vel_alpha)
        
        # Horizontal velocity (speed in the ground plane)
        horiz_disp = world_pos[:, plane_dims] - self._prev_positions[:min(J, self._prev_positions.shape[0]), :][:, plane_dims]
        raw_horiz_speed = np.linalg.norm(horiz_disp, axis=-1) / max(dt, 1e-6)
        horiz_alpha = 0.3
        self._smooth_horiz_vel = (self._smooth_horiz_vel * (1 - horiz_alpha) + 
                                   raw_horiz_speed * horiz_alpha)
        
        self._prev_heights = heights.copy()
        self._prev_positions = world_pos.copy()
        
        vy = self._smooth_velocity
        vh = self._smooth_horiz_vel
        
        # ── Lift-off / landing state machine ──
        p_lift = np.ones(J)
        
        # Height-based lift-state escape: a joint near the floor cannot
        # be in a "lifted" state regardless of velocity history.
        # This prevents the lift-state from getting stuck during rapid
        # hop sequences where the foot returns to floor but vy never
        # settles for 3 consecutive frames.
        LIFT_ESCAPE_HEIGHT = 0.08  # 8cm — if below this, force clear lift state
        
        for j in range(J):
            # Height escape: near-floor joints can't be lifted
            if self._lift_state[j] and heights[j] < LIFT_ESCAPE_HEIGHT:
                self._lift_state[j] = False
                self._lift_frames[j] = 0
            
            if self._lift_state[j]:
                if vy[j] < -0.05 and heights[j] < ceilings[j]:
                    self._lift_frames[j] += 1
                    if self._lift_frames[j] > 3:
                        self._lift_state[j] = False
                        self._lift_frames[j] = 0
                else:
                    self._lift_frames[j] = 0
                    p_lift[j] = 0.05
            else:
                if vy[j] > 0.1 and heights[j] > -0.05:
                    self._lift_frames[j] += 1
                    if self._lift_frames[j] > 2:
                        self._lift_state[j] = True
                        self._lift_frames[j] = 0
                else:
                    self._lift_frames[j] = 0
                
                # Stronger velocity suppression for upward-moving joints.
                # Even small upward velocity reduces contact likelihood.
                if vy[j] > 0:
                    # Boosted scale: vy=0.1 → 0.45, vy=0.2 → 0.20, vy=0.3 → 0.09
                    p_lift[j] = np.exp(-8.0 * vy[j])
        
        # ── Chain suppression ──
        # Two mechanisms work together:
        # (A) Margin-based: suppress parent if it's above child beyond the
        #     anatomical margin (angle + height). Handles upright poses.
        # (B) Child-priority: when ANY child is closer to the floor than the
        #     parent, attenuate the parent based on the height difference.
        #     Handles crouching poses where the foot/toes are the real
        #     contact but the knee is also near the floor.
        p_chain = np.ones(J)
        
        CHILD_PRIORITY_SCALE = 20.0  # per-meter; 0.05m diff → 63% suppression
        
        for parent, (children, margin) in self.CHAIN_CHILDREN.items():
            if parent >= J:
                continue
            
            valid_children = [c for c in children if c < J]
            if not valid_children:
                continue
            
            parent_h = heights[parent]
            min_child_h = min(heights[c] for c in valid_children)
            
            # ── (A) Margin-based directional suppression ──
            h_diff_margin = parent_h - min_child_h - margin
            
            if h_diff_margin > 0:
                best_child = valid_children[np.argmin([heights[c] for c in valid_children])]
                child_pos = world_pos[best_child]
                parent_pos = world_pos[parent]
                
                to_child = child_pos - parent_pos
                to_child_len = np.linalg.norm(to_child)
                
                if to_child_len > 0.01:
                    down = np.zeros(3)
                    down[up] = -1.0
                    cos_angle = np.dot(to_child / to_child_len, down)
                    
                    if cos_angle > 0:
                        ramp = max(0.10, margin)
                        suppression = min(0.95, cos_angle * min(1.0, h_diff_margin / ramp))
                        p_chain[parent] *= (1.0 - suppression)
            
            # ── (B) Child-priority attenuation ──
            # If any child is closer to the floor, the parent is less likely
            # to be the contact point. Uses excess height beyond the margin.
            # Only applies when parent is within contact range (below ceiling).
            h_excess = parent_h - min_child_h - margin  # excess beyond margin
            
            if h_excess > 0.005 and parent_h < ceilings[parent]:
                # Exponential suppression based on how much the parent
                # exceeds the margin-adjusted child height.
                # With margin=0.02 (knee), 0.07m raw → 0.05m excess → 63%
                # With margin=0.15 (pelvis), 0.12m raw → 0m excess → 0%
                attenuation = 1.0 - np.exp(-CHILD_PRIORITY_SCALE * h_excess)
                p_chain[parent] *= (1.0 - attenuation)
        
        # ── Horizontal velocity suppression ──
        # A joint moving quickly horizontally is likely swinging, not in contact.
        # Grounded joints have near-zero horizontal velocity.
        # Scale: 0.5 m/s → mild suppression, 1.5+ m/s → strong suppression
        HORIZ_VEL_THRESHOLD = 0.3   # m/s below this, no penalty
        HORIZ_VEL_SCALE = 3.0       # exponential decay rate
        
        p_horiz = np.ones(J)
        for j in range(J):
            excess_vel = max(0, vh[j] - HORIZ_VEL_THRESHOLD)
            if excess_vel > 0:
                p_horiz[j] = np.exp(-HORIZ_VEL_SCALE * excess_vel)
        
        p_sensory = p_height * p_lift * p_chain * p_horiz
        return p_sensory


class StructuralAnalyzer:
    """
    Determines which contacts are structurally required to support the body.
    
    Uses the simplified chain approach:
    1. For a given set of candidate contacts, partition the body into
       "support islands" — segments between contact points
    2. Each contact bears the mass of the segments it supports,
       weighted by lever arms
    """
    
    def __init__(self, parents: list, segment_masses: np.ndarray, 
                 num_joints: int = 24):
        """
        Args:
            parents: list of parent indices (-1 for root)
            segment_masses: (24,) mass of each segment in kg
        """
        self.parents = parents
        self.segment_masses = segment_masses
        self.num_joints = num_joints
        
        # Pre-compute children for each joint
        self.children = {j: [] for j in range(num_joints)}
        for j in range(num_joints):
            p = parents[j]
            if p >= 0:
                self.children[p].append(j)
    
    def compute(self, world_pos: np.ndarray, contact_candidates: set,
                p_sensory: np.ndarray, options: ConsensusOptions) -> np.ndarray:
        """
        Compute structural load necessity for each joint.
        
        Strategy:
        1. Find likely contact joints (p_sensory > threshold)
        2. Compute the total gravitational torque about CoM
        3. Determine how much load each contact must bear for equilibrium
        
        Args:
            world_pos: (J, 3) joint positions
            contact_candidates: set of joint indices eligible for contact
            p_sensory: (J,) sensory probabilities (to filter candidates)
            options: ConsensusOptions
            
        Returns:
            p_structural: (J,) structural load necessity (0 to 1, sums to ~1)
        """
        J = min(world_pos.shape[0], self.num_joints)
        up = options.up_axis
        
        # Identify active contacts: candidates with some sensory evidence
        SENSORY_THRESHOLD = 0.05
        active_contacts = []
        for j in contact_candidates:
            if j < J and p_sensory[j] > SENSORY_THRESHOLD:
                active_contacts.append(j)
        
        if len(active_contacts) == 0:
            # No contacts detected — return uniform
            result = np.zeros(world_pos.shape[0])
            return result
        
        # ── Compute body CoM ──
        total_mass = 0.0
        com = np.zeros(3)
        for j in range(min(J, 24)):
            m = self.segment_masses[j] if j < len(self.segment_masses) else 0
            if m > 0:
                com += world_pos[j] * m
                total_mass += m
        if total_mass > 0:
            com /= total_mass
        
        # ── Determine horizontal plane dimensions ──
        # For Y-up: horizontal = [0, 2] (X, Z)
        if up == 1:
            plane_dims = [0, 2]
        elif up == 2:
            plane_dims = [0, 1]
        else:
            plane_dims = [1, 2]
        
        com_hz = com[plane_dims]
        
        # ── Compute load shares via inverse-distance weighting ──
        # More sophisticated than simple Gaussian: uses the structural logic
        # of how much gravitational torque each contact must counteract.
        #
        # For N contacts, solve the static equilibrium:
        #   Sum(F_i) = M*g  (total force = total weight)
        #   Sum(F_i * r_i) = 0  (net torque about CoM = 0)
        #
        # In 2D horizontal plane, this gives us 3 equations for N unknowns.
        # For N=2: fully determined (lever arm ratio)
        # For N=3: fully determined (barycentric coordinates)
        # For N>3: underdetermined, use least-squares

        contact_positions_hz = np.array([world_pos[j][plane_dims] for j in active_contacts])
        
        load_shares = self._solve_static_equilibrium(
            com_hz, contact_positions_hz, total_mass
        )
        
        # Map back to full joint array
        result = np.zeros(world_pos.shape[0])
        for i, j in enumerate(active_contacts):
            result[j] = max(0, load_shares[i])
        
        # Normalize to sum to 1
        total = np.sum(result)
        if total > 1e-6:
            result /= total
        
        return result
    
    def _solve_static_equilibrium(self, com_hz: np.ndarray, 
                                   contact_pts: np.ndarray,
                                   total_mass: float) -> np.ndarray:
        """
        Solve for force distribution at contact points for static equilibrium.
        
        Uses least-norm solution for underdetermined systems, 
        exact solution for determined systems (2-3 contacts),
        and least-squares for overdetermined.
        
        Args:
            com_hz: (2,) CoM position in horizontal plane
            contact_pts: (N, 2) contact positions in horizontal plane
            total_mass: total body mass
            
        Returns:
            forces: (N,) vertical force at each contact (in units of body weight)
        """
        N = len(contact_pts)
        
        if N == 0:
            return np.array([])
        
        if N == 1:
            return np.array([1.0])
        
        # Build the equilibrium matrix:
        # Row 0: Sum(F_i) = 1  (forces sum to total weight, normalized to 1)
        # Row 1: Sum(F_i * (x_i - com_x)) = 0  (torque about X)
        # Row 2: Sum(F_i * (z_i - com_z)) = 0  (torque about Z)
        A = np.zeros((3, N))
        b = np.array([1.0, 0.0, 0.0])
        
        for i in range(N):
            A[0, i] = 1.0
            A[1, i] = contact_pts[i, 0] - com_hz[0]
            A[2, i] = contact_pts[i, 1] - com_hz[1]
        
        if N == 2:
            # 2 contacts: use moment equation to solve
            # F1 * d1 = F2 * d2, F1 + F2 = 1
            d1 = np.linalg.norm(contact_pts[0] - com_hz)
            d2 = np.linalg.norm(contact_pts[1] - com_hz)
            total_d = d1 + d2
            if total_d > 1e-6:
                # Inverse distance: closer contact bears more
                return np.array([d2 / total_d, d1 / total_d])
            else:
                return np.array([0.5, 0.5])
        
        elif N == 3:
            # 3 contacts: exactly determined (barycentric coordinates)
            # A is 3x3, solve directly
            try:
                forces = np.linalg.solve(A, b)
                return forces
            except np.linalg.LinAlgError:
                # Singular (collinear contacts) — fall back to inverse distance
                return self._inverse_distance_weights(com_hz, contact_pts)
        
        else:
            # N > 3: underdetermined — use minimum-norm solution
            # This gives the force distribution with smallest total force magnitude
            # subject to equilibrium constraints
            try:
                # Least-norm: F = A^T (A A^T)^{-1} b
                AAT = A @ A.T
                forces = A.T @ np.linalg.solve(AAT, b)
                return forces
            except np.linalg.LinAlgError:
                return self._inverse_distance_weights(com_hz, contact_pts)
    
    def _inverse_distance_weights(self, com_hz: np.ndarray, 
                                   contact_pts: np.ndarray) -> np.ndarray:
        """Fallback: inverse distance weighting."""
        distances = np.array([max(0.01, np.linalg.norm(p - com_hz)) 
                             for p in contact_pts])
        inv_d = 1.0 / distances
        return inv_d / np.sum(inv_d)
    
    def compute_chain_loads(self, world_pos: np.ndarray, 
                            active_contacts: list,
                            options: ConsensusOptions) -> Dict[int, float]:
        """
        Alternative structural analysis using chain partitioning.
        
        For each pair of adjacent contact points on the kinematic tree,
        compute the mass of segments between them and distribute based
        on lever arms.
        
        This is the "support island" approach described in the design doc.
        
        Returns:
            Dict mapping joint index → load fraction
        """
        up = options.up_axis
        if up == 1:
            plane_dims = [0, 2]
        else:
            plane_dims = [0, 1]
            
        # For each joint, find the nearest contact point(s) on the kinematic path
        # toward root and toward leaves
        loads = {j: 0.0 for j in active_contacts}
        
        for j in range(min(world_pos.shape[0], 24)):
            m = self.segment_masses[j] if j < len(self.segment_masses) else 0
            if m <= 0:
                continue
            
            # Find the nearest contact going up (toward root)
            contact_above = self._find_contact_toward_root(j, active_contacts)
            # Find the nearest contact(s) going down (toward leaves)  
            contacts_below = self._find_contacts_toward_leaves(j, active_contacts)
            
            # All contacts that could support this segment
            supporting_contacts = []
            if contact_above is not None:
                supporting_contacts.append(contact_above)
            supporting_contacts.extend(contacts_below)
            
            if len(supporting_contacts) == 0:
                # No contacts found — this mass is unsupported (maybe falling)
                continue
            
            if len(supporting_contacts) == 1:
                # Only one contact — it bears all the mass
                loads[supporting_contacts[0]] += m
            else:
                # Multiple contacts — distribute by inverse lever arm
                seg_pos = world_pos[j][plane_dims]
                distances = []
                for c in supporting_contacts:
                    d = max(0.01, np.linalg.norm(
                        world_pos[c][plane_dims] - seg_pos))
                    distances.append(d)
                total_d = sum(distances)
                for i, c in enumerate(supporting_contacts):
                    # Inverse: closer contact bears more
                    loads[c] += m * (1 - distances[i] / total_d) / (len(supporting_contacts) - 1) if len(supporting_contacts) > 1 else m
        
        # Normalize
        total = sum(loads.values())
        if total > 0:
            loads = {j: v / total for j, v in loads.items()}
        
        return loads
    
    def _find_contact_toward_root(self, joint: int, 
                                   contacts: list) -> Optional[int]:
        """Walk up the kinematic tree to find nearest contact."""
        current = joint
        while current >= 0:
            if current in contacts and current != joint:
                return current
            if current == joint and current in contacts:
                return current  # Joint itself is a contact
            current = self.parents[current] if current < len(self.parents) else -1
        return None
    
    def _find_contacts_toward_leaves(self, joint: int,
                                      contacts: list) -> list:
        """Find contacts in the subtree below this joint."""
        result = []
        stack = list(self.children.get(joint, []))
        while stack:
            j = stack.pop()
            if j in contacts:
                result.append(j)
                # Don't go deeper past a contact point
            else:
                stack.extend(self.children.get(j, []))
        return result


class DynamicAnalyzer:
    """
    Adjusts contact probabilities based on body dynamics.
    
    If the body is falling, contacts should be suppressed.
    If the body is decelerating toward the ground, contacts should be boosted.
    """
    
    def __init__(self, num_joints: int = 30):
        self.num_joints = num_joints
        self._prev_com = None
        self._prev_com_vel = None
        self._smooth_com_vel = None
        self._smooth_com_acc = None
    
    def reset(self):
        self._prev_com = None
        self._prev_com_vel = None
        self._smooth_com_vel = None
        self._smooth_com_acc = None
    
    def compute(self, com: np.ndarray, world_pos: np.ndarray,
                dt: float, options: ConsensusOptions) -> np.ndarray:
        """
        Compute dynamic modifiers for contact probabilities.
        
        Args:
            com: (3,) center of mass position
            world_pos: (J, 3) joint positions
            dt: time step
            options: ConsensusOptions
            
        Returns:
            p_dynamic: (J,) multiplier per joint (typically 0.5-1.5)
                       <1 = suppressed, >1 = boosted, 1 = neutral
        """
        J = world_pos.shape[0]
        up = options.up_axis
        
        # ── CoM velocity and acceleration ──
        if self._prev_com is None:
            self._prev_com = com.copy()
            self._smooth_com_vel = np.zeros(3)
            self._smooth_com_acc = np.zeros(3)
            return np.ones(J)
        
        com_vel = (com - self._prev_com) / max(dt, 1e-6)
        
        if self._prev_com_vel is None:
            self._prev_com_vel = com_vel.copy()
            self._smooth_com_vel = com_vel.copy()
            self._prev_com = com.copy()
            return np.ones(J)
        
        com_acc = (com_vel - self._prev_com_vel) / max(dt, 1e-6)
        
        # Smooth
        alpha = 0.3
        self._smooth_com_vel = self._smooth_com_vel * (1 - alpha) + com_vel * alpha
        self._smooth_com_acc = self._smooth_com_acc * (1 - alpha) + com_acc * alpha
        
        self._prev_com_vel = com_vel.copy()
        self._prev_com = com.copy()
        
        # ── Global dynamic state ──
        com_vy = self._smooth_com_vel[up]
        com_ay = self._smooth_com_acc[up]
        
        # Is the body falling? (CoM accelerating downward)
        # Suppress all contacts if in freefall
        # Note: with gravity present, stationary body has ay ≈ 0 (balanced)
        # Falling body has ay ≈ -9.81
        falling_factor = 1.0
        if com_ay < -options.fall_threshold:
            # Body is accelerating downward — reducing contact
            falling_factor = max(0.3, 1.0 + com_ay / 9.81)
        
        # Is the CoM moving upward? (jumping)
        jumping_factor = 1.0
        if com_vy > 0.3:
            # Significant upward velocity — contacts releasing
            jumping_factor = max(0.2, np.exp(-2.0 * com_vy))
        
        # ── Horizontal CoM movement → load shift ──
        # If CoM is accelerating horizontally, the contacts on the 
        # "push-off" side bear more load
        if up == 1:
            plane_dims = [0, 2]
        else:
            plane_dims = [0, 1]
        
        com_hz = com[plane_dims]
        acc_hz = self._smooth_com_acc[plane_dims]
        acc_mag = np.linalg.norm(acc_hz)
        
        p_dynamic = np.ones(J)
        
        for j in range(J):
            # Base: global modifiers
            p_dynamic[j] *= falling_factor * jumping_factor
            
            # Per-joint: if CoM is accelerating, joints opposite to 
            # acceleration direction get boosted (they're pushing)
            if acc_mag > 1.0:
                joint_hz = world_pos[j][plane_dims]
                # Vector from CoM to joint
                to_joint = joint_hz - com_hz
                to_joint_norm = np.linalg.norm(to_joint)
                if to_joint_norm > 0.01:
                    # Dot with acceleration: negative = opposite to accel = pushing
                    cos_angle = np.dot(to_joint, -acc_hz) / (to_joint_norm * acc_mag)
                    # Boost joints opposing acceleration
                    boost = 1.0 + 0.3 * max(0, cos_angle) * min(1.0, acc_mag / 5.0)
                    p_dynamic[j] *= boost
        
        return p_dynamic


class TorquePlausibility:
    """
    Checks whether a contact hypothesis produces physically plausible torques.
    
    Uses previous-frame torques as a baseline, and checks whether the
    contact forces implied by the hypothesis would create excessive 
    joint torques.
    """
    
    def __init__(self, max_torques: np.ndarray = None, num_joints: int = 24):
        """
        Args:
            max_torques: (24, 3) maximum torque per joint per axis
        """
        self.max_torques = max_torques
        self.num_joints = num_joints
        self._prev_implausibility = None
    
    def reset(self):
        self._prev_implausibility = None
    
    def compute(self, contact_probs: np.ndarray, world_pos: np.ndarray,
                prev_torques: np.ndarray, options: ConsensusOptions) -> np.ndarray:
        """
        Compute plausibility correction for each joint's contact probability.
        
        If the previous frame's torques were excessive at certain joints,
        this suggests the contact assignment may be wrong.
        
        Args:
            contact_probs: (J,) current contact probability hypothesis
            world_pos: (J, 3) joint positions
            prev_torques: (24, 3) torque vectors from previous frame, or None
            options: ConsensusOptions
            
        Returns:
            p_plausible: (J,) multiplier per joint (≤1, where 1=plausible)
        """
        J = contact_probs.shape[0]
        p_plausible = np.ones(J)
        
        if prev_torques is None or self.max_torques is None:
            return p_plausible
        
        # For each joint, check if previous torque was excessive
        num_torque_joints = min(J, prev_torques.shape[0])
        for j in range(num_torque_joints):
            torque_mag = np.linalg.norm(prev_torques[j])
            max_mag = np.linalg.norm(self.max_torques[j])
            
            if max_mag < 1e-6:
                continue
            
            ratio = torque_mag / max_mag
            
            if ratio > 1.0:
                # Excessive torque — this joint is over-stressed
                # This could mean:
                # 1. A contact below this joint is missing (need to add support)
                # 2. A false contact above is forcing load through this joint
                
                excess = ratio - 1.0
                penalty = min(0.8, excess * options.torque_excess_penalty)
                
                # Penalize contacts near this over-stressed joint
                # The idea: if a joint is over-torqued, the current contact
                # assignment around it might be wrong
                p_plausible[j] *= (1.0 - penalty * 0.5)
        
        return p_plausible
    
    def suggest_missing_contacts(self, prev_torques: np.ndarray,
                                  world_pos: np.ndarray,
                                  current_contacts: np.ndarray,
                                  options: ConsensusOptions) -> Dict[int, float]:
        """
        Identify joints that should probably be in contact based on
        torque implausibility.
        
        If a joint has excessive torque and a child joint is near the floor,
        adding that child as a contact would reduce the parent's torque.
        
        Returns:
            Dict of joint_idx → suggested contact boost
        """
        suggestions = {}
        
        if prev_torques is None or self.max_torques is None:
            return suggestions
        
        up = options.up_axis
        floor = options.floor_height
        
        for j in range(min(world_pos.shape[0], prev_torques.shape[0])):
            torque_mag = np.linalg.norm(prev_torques[j])
            max_mag = np.linalg.norm(self.max_torques[j])
            
            if max_mag > 1e-6 and torque_mag > max_mag * 1.5:
                # Very over-stressed — look for nearby supports
                height = world_pos[j, up] - floor
                if height < 0.3 and current_contacts[j] < 0.3:
                    # This joint is low and not a current contact — suggest it
                    excess_ratio = torque_mag / max_mag
                    suggestions[j] = min(0.5, (excess_ratio - 1.0) * 0.2)
        
        return suggestions


class ContactConsensus:
    """
    Main orchestrator for consensus contact detection.
    
    Combines sensory, structural, dynamic, and plausibility evidence
    into a single iterative consensus process.
    """
    
    def __init__(self, parents: list = None, segment_masses: np.ndarray = None,
                 max_torques: np.ndarray = None, num_joints: int = 30):
        """
        Args:
            parents: kinematic tree parent indices
            segment_masses: (24,) per-segment masses
            max_torques: (24, 3) per-joint max torque
            num_joints: total joints including virtual
        """
        if parents is None:
            parents = PARENTS
        if segment_masses is None:
            segment_masses = np.ones(24) * 3.0  # ~72kg / 24 joints default
        
        self.num_joints = num_joints
        
        # Initialize sub-modules
        self.sensory = SensoryPrior(num_joints)
        self.structural = StructuralAnalyzer(parents, segment_masses, 
                                              num_joints=min(24, num_joints))
        self.dynamic = DynamicAnalyzer(num_joints)
        self.plausibility = TorquePlausibility(max_torques, num_joints=24)
        
        # State
        self._smooth_contacts = None
        self._prev_torques = None
        self.options = ConsensusOptions()
    
    def reset(self):
        """Reset all internal state."""
        self.sensory.reset()
        self.dynamic.reset()
        self.plausibility.reset()
        self._smooth_contacts = None
        self._prev_torques = None
    
    def set_prev_torques(self, torques: np.ndarray):
        """Update with torques from the most recent frame."""
        self._prev_torques = torques.copy() if torques is not None else None
    
    def compute_contacts(self, world_pos: np.ndarray, com: np.ndarray,
                          dt: float, options: ConsensusOptions = None
                          ) -> np.ndarray:
        """
        Compute consensus contact probabilities.
        
        Args:
            world_pos: (J, 3) joint positions in world space
            com: (3,) center of mass position
            dt: time step in seconds
            options: ConsensusOptions (uses stored defaults if None)
            
        Returns:
            contact_probs: (J,) consensus contact probabilities
        """
        if options is None:
            options = self.options
        
        J = world_pos.shape[0]
        
        # Only consider plausible contact candidates
        candidates = {j for j in CONTACT_CANDIDATES if j < J}
        
        # ══════════════════════════════════════════════════
        # Stage 1: Sensory Prior
        # ══════════════════════════════════════════════════
        p_sensory = self.sensory.compute(world_pos, dt, options)
        
        # ══════════════════════════════════════════════════
        # Stage 2: Structural Analysis
        # ══════════════════════════════════════════════════
        p_structural = self.structural.compute(
            world_pos, candidates, p_sensory, options)
        
        # ══════════════════════════════════════════════════
        # Stage 3: Dynamic Analysis
        # ══════════════════════════════════════════════════
        p_dynamic = self.dynamic.compute(com, world_pos, dt, options)
        
        # ══════════════════════════════════════════════════
        # Stage 4: Combine Evidence
        # ══════════════════════════════════════════════════
        # Product-of-experts with structural boosting
        p_combined = np.zeros(J)
        
        for j in range(J):
            if j not in candidates:
                p_combined[j] = 0.0
                continue
            
            # Base: sensory evidence
            p = p_sensory[j]
            
            # Structural: boost joints that are structurally necessary
            # If structural says this joint should bear 30% of load but
            # sensory says low probability, structural can boost it
            if p_structural[j] > 0.01:
                # Structural evidence suggests this is a support point
                # Blend: sensory weighted by (1 - structural_weight),
                # structural by structural_weight
                w_st = options.structural_weight
                p = p * (1 - w_st) + p_structural[j] * w_st
            
            # Dynamic modulation
            p *= p_dynamic[j]
            
            p_combined[j] = np.clip(p, 0, 1)
        
        # ══════════════════════════════════════════════════
        # Stage 5: Torque Plausibility (from previous frame)
        # ══════════════════════════════════════════════════
        p_plausible = self.plausibility.compute(
            p_combined, world_pos, self._prev_torques, options)
        
        # Apply plausibility corrections
        p_final = p_combined * p_plausible
        
        # Check for missing contacts suggested by torque analysis
        suggestions = self.plausibility.suggest_missing_contacts(
            self._prev_torques, world_pos, p_final, options)
        
        for j, boost in suggestions.items():
            if j < J:
                p_final[j] = max(p_final[j], boost)
        
        # ══════════════════════════════════════════════════
        # Stage 6: Temporal Smoothing
        # ══════════════════════════════════════════════════
        if self._smooth_contacts is None:
            self._smooth_contacts = p_final.copy()
        else:
            alpha = options.temporal_alpha
            self._smooth_contacts = (self._smooth_contacts * (1 - alpha) + 
                                      p_final * alpha)
        
        return self._smooth_contacts.copy()
