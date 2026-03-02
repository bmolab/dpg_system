"""
Dynamic Structural Frame Evaluator

Evaluates contact hypotheses against whole-body dynamic equilibrium.
Given a proposed set of contact joints and the body's dynamic state,
solves Newton-Euler to determine what force each contact must provide.

This is a read-only analysis tool — it does not modify contact decisions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple


@dataclass
class EvalResult:
    """Result of a dynamic frame evaluation."""
    
    # Per-contact required force in kg (positive = pushing up)
    per_contact_force: Dict[int, float] = field(default_factory=dict)
    
    # Per-contact necessity classification
    # 'necessary' = F > threshold, 'marginal' = 0 < F < threshold, 'unnecessary' = F <= 0
    necessity: Dict[int, str] = field(default_factory=dict)
    
    # Total required GRF vector (3,) in world space
    f_required: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Approximate ZMP position in horizontal plane (2,)
    zmp_approx: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    # Support polygon: contact positions in horizontal plane [(2,), ...]
    support_polygon: List[np.ndarray] = field(default_factory=list)
    
    # Residual unbalanced force if contacts can't fully balance (3,)
    residual: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Contacts that were pruned as unnecessary during refinement
    pruned_contacts: Set[int] = field(default_factory=set)
    
    # Contacts suggested as potentially missing
    suggested_contacts: Set[int] = field(default_factory=set)
    
    # Per-contact force as array (J,) for heatmap visualization
    force_array: np.ndarray = field(default_factory=lambda: np.zeros(0))
    
    # Per-contact necessity as array (J,): 1.0=necessary, 0.5=marginal, 0.0=unnecessary
    necessity_array: np.ndarray = field(default_factory=lambda: np.zeros(0))


class DynamicFrameEvaluator:
    """Evaluate contact hypotheses against whole-body dynamic equilibrium.
    
    The body with multiple ground contacts is a structural frame in motion.
    Given a proposed set of contact points, physics tells us:
    - Whether each contact is necessary (the frame needs it for support)
    - Whether each contact is unnecessary (the frame would be fine without it)
    - Whether a contact is missing (the frame can't balance without more support)
    """
    
    # Thresholds for necessity classification
    NECESSARY_THRESHOLD_KG = 2.0     # Above this = clearly necessary
    MARGINAL_THRESHOLD_KG = 0.5      # Below this = unnecessary
    NEAR_FLOOR_THRESHOLD_M = 0.15    # Joint must be within this height to be a candidate
    AUGMENT_RESIDUAL_THRESHOLD = 5.0 # kg-equivalent residual to trigger augment search
    
    def __init__(self, total_mass: float, segment_masses: np.ndarray,
                 num_joints: int = 30):
        """
        Args:
            total_mass: Total body mass in kg
            segment_masses: (24,) per-segment mass in kg
            num_joints: Total joint count (24 SMPL + virtual)
        """
        self.total_mass = total_mass
        self.segment_masses = segment_masses
        self.num_joints = num_joints
    
    @staticmethod
    def compute_effective_surface_distances(joint_surface_extents, global_rots,
                                             floor_normal, num_joints=24):
        """Compute per-joint distance from joint center to contact surface.
        
        Transforms T-pose surface extents into the current pose using global
        rotations, then finds how far the surface extends toward the floor.
        
        Args:
            joint_surface_extents: (24, 3, 2) min/max vertex offsets per axis in T-pose
            global_rots: (24, 3, 3) global rotation matrices for current pose
            floor_normal: (3,) upward floor normal (e.g., [0,1,0] for Y-up)
            num_joints: Number of joints to process
            
        Returns:
            surface_distances: (J,) distance from joint center to contact surface
                               in the floor-ward direction (always positive)
        """
        n_ext = min(num_joints, joint_surface_extents.shape[0])
        surface_distances = np.full(num_joints, 0.03)  # Default 3cm
        
        for j in range(n_ext):
            # Transform floor normal into joint's local frame
            if j < global_rots.shape[0]:
                # R_global rotates local → world, so R_global^T rotates world → local
                floor_in_local = global_rots[j].T @ floor_normal
            else:
                floor_in_local = floor_normal
            
            # The contact surface is in the direction OPPOSITE to floor normal
            # (floor normal points up, contact surface is on the downward side)
            contact_dir = -floor_in_local  # direction toward floor in local frame
            
            # For each axis, determine how much the surface extends in contact_dir
            # extents[:, 0] = min (negative direction), extents[:, 1] = max (positive)
            extent = 0.0
            for axis in range(3):
                if contact_dir[axis] > 0:
                    # Surface extends in positive direction → use max extent
                    extent += contact_dir[axis] * joint_surface_extents[j, axis, 1]
                else:
                    # Surface extends in negative direction → use min extent (negative)
                    extent += contact_dir[axis] * joint_surface_extents[j, axis, 0]
            
            # extent is now the signed distance from joint to surface in contact_dir
            # (should be positive = surface is below joint)
            surface_distances[j] = max(0.01, abs(extent))
        
        return surface_distances
    
    def evaluate(self, contact_joints: Set[int],
                 joint_positions: np.ndarray,
                 com: np.ndarray,
                 com_acc: np.ndarray,
                 floor_height: float,
                 up_axis: int = 1) -> EvalResult:
        """Evaluate a proposed contact set against dynamic equilibrium.
        
        Args:
            contact_joints: Set of joint indices proposed as contacts
            joint_positions: (J, 3) joint positions in world space
            com: (3,) center of mass position
            com_acc: (3,) center of mass acceleration (m/s²)
            floor_height: Floor height in world coordinates
            up_axis: Vertical axis index (1=Y-up, 2=Z-up)
            
        Returns:
            EvalResult with per-contact forces, necessity, ZMP, etc.
        """
        J = joint_positions.shape[0]
        g_mag = 9.81
        
        # Horizontal plane dimensions
        if up_axis == 1:
            plane_dims = [0, 2]
        else:
            plane_dims = [0, 1]
        
        # --- Step 1: Compute required GRF (full 3D) ---
        g_vec = np.zeros(3)
        g_vec[up_axis] = -g_mag
        f_required = self.total_mass * (com_acc - g_vec)  # (3,)
        f_required_up = f_required[up_axis]
        
        # --- Step 2: Compute approximate ZMP ---
        com_hz = com[plane_dims]
        h_com = com[up_axis] - floor_height
        a_hz = com_acc[plane_dims]
        a_vert = com_acc[up_axis] + g_mag  # effective vertical acceleration
        
        if a_vert > 2.0 and h_com > 0.01:
            # Body is being supported — ZMP is meaningful
            zmp_displacement = (h_com / a_vert) * a_hz
            # Clamp displacement to reasonable range (2m from CoM projection)
            disp_mag = np.linalg.norm(zmp_displacement)
            if disp_mag > 2.0:
                zmp_displacement *= 2.0 / disp_mag
            zmp_hz = com_hz - zmp_displacement
        else:
            # Freefall or very low CoM — ZMP is undefined, use CoM projection
            zmp_hz = com_hz.copy()
        
        # --- Step 3: Filter to valid contacts ---
        active_contacts = sorted(j for j in contact_joints if j < J)
        
        if not active_contacts:
            result = self._empty_result(J, f_required, zmp_hz)
            return result
        
        # Get contact positions in horizontal plane
        contact_positions_hz = np.array([
            joint_positions[j][plane_dims] for j in active_contacts
        ])
        
        # Build support polygon (convex hull of contact positions)
        support_polygon = [pos.copy() for pos in contact_positions_hz]
        
        # --- Step 4: Solve dynamic equilibrium ---
        # Total vertical force needed (in units of kg, not Newtons)
        total_force_kg = max(0, f_required_up / g_mag)
        
        if total_force_kg < 0.1:
            # Essentially no support needed (freefall or near-freefall)
            per_contact_force = {j: 0.0 for j in active_contacts}
        else:
            # Solve for force distribution using ZMP as the balance point
            force_fractions = self._solve_dynamic_equilibrium(
                zmp_hz, contact_positions_hz, active_contacts
            )
            per_contact_force = {
                j: force_fractions[i] * total_force_kg
                for i, j in enumerate(active_contacts)
            }
        
        # --- Step 5: Classify necessity ---
        necessity = {}
        for j, force in per_contact_force.items():
            if force > self.NECESSARY_THRESHOLD_KG:
                necessity[j] = 'necessary'
            elif force > self.MARGINAL_THRESHOLD_KG:
                necessity[j] = 'marginal'
            else:
                necessity[j] = 'unnecessary'
        
        # --- Step 6: Compute residual ---
        # Check if the contacts can actually balance the required force/moment
        residual = self._compute_residual(
            zmp_hz, contact_positions_hz, per_contact_force,
            active_contacts, total_force_kg
        )
        
        # --- Build output arrays ---
        force_array = np.zeros(J)
        necessity_array = np.zeros(J)
        for j, force in per_contact_force.items():
            if j < J:
                force_array[j] = max(0, force)
        for j, nec in necessity.items():
            if j < J:
                if nec == 'necessary':
                    necessity_array[j] = 1.0
                elif nec == 'marginal':
                    necessity_array[j] = 0.5
                else:
                    necessity_array[j] = 0.0
        
        return EvalResult(
            per_contact_force=per_contact_force,
            necessity=necessity,
            f_required=f_required,
            zmp_approx=zmp_hz,
            support_polygon=support_polygon,
            residual=residual,
            pruned_contacts=set(),
            suggested_contacts=set(),
            force_array=force_array,
            necessity_array=necessity_array,
        )
    
    def evaluate_and_refine(self, candidate_joints: Set[int],
                            joint_positions: np.ndarray,
                            com: np.ndarray,
                            com_acc: np.ndarray,
                            floor_height: float,
                            up_axis: int = 1,
                            max_iterations: int = 3,
                            all_joint_heights: Optional[np.ndarray] = None,
                            surface_distances: Optional[np.ndarray] = None,
                            excluded_joints: Optional[set] = None,
                            consensus_probs: Optional[np.ndarray] = None,
                            joint_velocities: Optional[np.ndarray] = None,
                            prev_active_contacts: Optional[Set[int]] = None,
                            prev_seed: Optional[int] = None,
                            ) -> EvalResult:
        """Greedy iterative contact selection.
        
        Builds the active contact set one joint at a time:
        1. Seed with highest-probability candidate
        2. Evaluate plausibility (max force, residual)
        3. If not plausible, recruit the best additional candidate
        4. Repeat until plausible or candidates exhausted
        
        Args:
            candidate_joints: Pool of possible contact joint indices
            joint_positions: (J, 3) joint positions
            com: (3,) center of mass
            com_acc: (3,) CoM acceleration
            floor_height: Floor height
            up_axis: Vertical axis (1=Y, 2=Z)
            max_iterations: Max recruitment rounds
            all_joint_heights: Optional (J,) heights for augment search
            surface_distances: Optional per-joint surface offsets
            consensus_probs: Optional (J,) consensus probability per joint
            joint_velocities: Optional (J,) horizontal velocity magnitude per joint
            prev_active_contacts: Optional set of contacts active in previous frame
            prev_seed: Optional joint index that was the seed in previous frame
        """
        J = joint_positions.shape[0]
        plane_dims = [0, 2] if up_axis == 1 else [0, 1]
        g_mag = 9.81
        
        if not candidate_joints:
            return self._empty_result(J, np.zeros(3), np.zeros(2))
        
        # --- Contact group merging ---
        # Foot/heel and hand/wrist are treated as single candidates.
        # The lowest member represents the group during greedy selection.
        # After force distribution, we split within groups by relative height.
        CONTACT_GROUPS = [
            (10, 28),  # L_foot, L_heel
            (11, 29),  # R_foot, R_heel
            (20, 22),  # L_wrist, L_hand
            (21, 23),  # R_wrist, R_hand
        ]
        HEIGHT_SCALE = 0.06  # Height diff for full weight shift (covers ±6cm range)
        
        # Build group map: member → (representative, partner)
        group_rep = {}   # member_j → representative_j
        group_members = {}  # representative_j → [member_j, ...]
        merged_candidates = set(candidate_joints)
        
        for j_a, j_b in CONTACT_GROUPS:
            a_in = j_a in candidate_joints
            b_in = j_b in candidate_joints
            if not (a_in or b_in):
                continue
            # Use the lower joint as representative
            h_a = joint_positions[j_a, up_axis] if j_a < J else 999
            h_b = joint_positions[j_b, up_axis] if j_b < J else 999
            if h_a <= h_b:
                rep, other = j_a, j_b
            else:
                rep, other = j_b, j_a
            
            group_rep[j_a] = rep
            group_rep[j_b] = rep
            group_members[rep] = [j_a, j_b]
            
            # Remove non-representative from candidates
            merged_candidates.discard(other)
            merged_candidates.add(rep)
        
        # Use merged candidates for the greedy process
        candidate_joints = merged_candidates
        
        # --- Step 1: Score candidates for initial ranking ---
        # Combines consensus probability with temporal hysteresis boost.
        # Hysteresis only applies if consensus still mildly supports contact —
        # prevents feedback loop where a joint recruited once by noisy ZMP
        # stays locked in via hysteresis despite consensus = 0.
        HYSTERESIS_BOOST = 0.40  # Boost for contacts from previous frame
        HYSTERESIS_MIN_CONSENSUS = 0.05  # Consensus must be at least this for boost
        
        candidate_scores = {}
        for j in candidate_joints:
            score = 0.0
            cons_prob = 0.0
            # Consensus probability — use max of group members
            if consensus_probs is not None:
                members = group_members.get(j, [j])
                cons_prob = max(
                    float(consensus_probs[m]) if m < len(consensus_probs) else 0
                    for m in members
                )
                score = cons_prob
            # Temporal hysteresis — check any group member was active
            if prev_active_contacts is not None and cons_prob >= HYSTERESIS_MIN_CONSENSUS:
                members = group_members.get(j, [j])
                if any(m in prev_active_contacts for m in members):
                    score += HYSTERESIS_BOOST
            candidate_scores[j] = score
        
        # Sort candidates by score (descending)
        ranked = sorted(candidate_scores.keys(), key=lambda j: candidate_scores[j], reverse=True)
        
        # --- Step 2: Seed the active set ---
        # For simple cases (1-2 prev contacts), use single seed (greedy works well).
        # For complex cases (3+ prev contacts), seed with all previously-active
        # contacts that are still candidates — this prevents greedy path instability.
        prev_active_reps = set()
        if prev_active_contacts:
            for j in prev_active_contacts:
                rep = group_rep.get(j, j)
                if rep in candidate_joints:
                    prev_active_reps.add(rep)
        
        if len(prev_active_reps) >= 3:
            # Multi-contact stability: seed with previous active set
            active_set = set(prev_active_reps)
            seed = ranked[0] if ranked else None
            remaining = [j for j in ranked if j not in active_set]
        else:
            # Normal single-seed greedy
            effective_seed = prev_seed
            if effective_seed is not None and effective_seed in group_rep:
                effective_seed = group_rep[effective_seed]
            if effective_seed is not None and effective_seed in candidate_joints:
                seed = effective_seed
            else:
                seed = ranked[0]
            active_set = {seed}
            remaining = [j for j in ranked if j != seed]
        
        # --- Step 3-5: Iterative recruitment ---
        MAX_FORCE_THRESHOLD = 1.5 * self.total_mass  # kg
        PREV_CONTACT_BOOST = 5.0  # Temporal advantage: previous contacts score 5x
        IMPROVEMENT_THRESHOLD = 0.25  # Keep recruiting if adding a contact reduces max force by >25%
        MIN_RECRUIT_SCORE = 0.15  # Minimum score to recruit during improvement phase
        
        prev_max_force = float('inf')
        last_added_candidate = None  # Track what was just added
        
        for iteration in range(max_iterations + len(remaining)):
            result = self.evaluate(
                active_set, joint_positions, com, com_acc,
                floor_height, up_axis
            )
            
            # Check plausibility
            max_force = max(result.per_contact_force.values()) if result.per_contact_force else 0
            residual_mag = np.linalg.norm(result.residual)
            
            # Phase 1: Must-recruit (forces too high or residual too large)
            must_recruit = (max_force > MAX_FORCE_THRESHOLD or 
                           residual_mag > self.AUGMENT_RESIDUAL_THRESHOLD)
            
            if not must_recruit and not remaining:
                break
            
            if not must_recruit:
                # Phase 2: Improvement check
                if iteration > 0:
                    improvement = (prev_max_force - max_force) / (prev_max_force + 1e-6)
                    was_prev_active = (prev_active_contacts is not None and 
                                       last_added_candidate is not None and
                                       last_added_candidate in prev_active_contacts)
                    last_consensus = candidate_scores.get(last_added_candidate, 0) if last_added_candidate is not None else 0
                    # Only relax if the joint isn't moving horizontally
                    last_vel = 0.0
                    if joint_velocities is not None and last_added_candidate is not None and last_added_candidate < len(joint_velocities):
                        last_vel = float(joint_velocities[last_added_candidate])
                    if was_prev_active and last_consensus > 0.3 and last_vel < 0.3:
                        thresh = 0.02
                    else:
                        thresh = IMPROVEMENT_THRESHOLD
                    if improvement < thresh:
                        break
            
            prev_max_force = max_force
            
            if not remaining:
                break
            
            # --- Find where support is needed ---
            # The residual indicates the unbalanced force/moment direction.
            # We want a candidate in the direction that reduces the residual.
            # Use ZMP-to-CoM direction as the "need" direction.
            com_hz = com[plane_dims]
            zmp_hz = result.zmp_approx
            need_direction = zmp_hz - com_hz  # Direction ZMP is displaced from CoM
            need_mag = np.linalg.norm(need_direction)
            if need_mag > 1e-6:
                need_direction = need_direction / need_mag
            else:
                need_direction = np.zeros(2)
            
            # --- Score remaining candidates for recruitment ---
            best_score = -np.inf
            best_candidate = None
            
            for j in remaining:
                # Spatial relevance: how well does this candidate address the need?
                # Vector from CoM to candidate position (horizontal)
                cand_hz = joint_positions[j][plane_dims]
                cand_dir = cand_hz - com_hz
                cand_dist = np.linalg.norm(cand_dir)
                if cand_dist > 1e-6:
                    cand_dir_norm = cand_dir / cand_dist
                else:
                    cand_dir_norm = np.zeros(2)
                
                # Dot product with need direction (1.0 = perfect, -1.0 = opposite)
                spatial = np.dot(cand_dir_norm, need_direction)
                # Shift to [0, 1] range: (-1,1) → (0,1)
                spatial = max(0.0, (spatial + 1.0) / 2.0)
                
                # Velocity penalty: moving joints are unlikely real contacts
                # Squared for stronger rejection — 1m/s → 0.053, 0.5m/s → 0.138
                vel_penalty = 1.0
                if joint_velocities is not None and j < len(joint_velocities):
                    v_hz = float(joint_velocities[j])
                    vp = 1.0 / (1.0 + v_hz / 0.3)  # 0.3 m/s reference
                    vel_penalty = vp * vp  # Squared for stronger effect
                
                # Consensus as mild tiebreaker
                prob_factor = 0.5 + 0.5 * candidate_scores.get(j, 0)
                
                # Temporal advantage: previous contacts get a boost
                # so the greedy process naturally preserves continuity
                temporal_boost = 1.0
                if prev_active_contacts is not None and j in prev_active_contacts:
                    temporal_boost = PREV_CONTACT_BOOST
                
                recruit_score = spatial * vel_penalty * prob_factor * temporal_boost
                
                if recruit_score > best_score:
                    best_score = recruit_score
                    best_candidate = j
            
            if best_candidate is not None:
                # During improvement phase, require minimum quality
                # But allow previously-active contacts through — they shouldn't
                # need to re-prove spatial relevance every frame
                if not must_recruit and best_score < MIN_RECRUIT_SCORE:
                    best_was_prev = (prev_active_contacts is not None and
                                    best_candidate in prev_active_contacts)
                    # Only persist if the joint isn't moving horizontally
                    best_vel = 0.0
                    if joint_velocities is not None and best_candidate < len(joint_velocities):
                        best_vel = float(joint_velocities[best_candidate])
                    if not (best_was_prev and best_vel < 0.3):
                        break  # Best candidate isn't good enough — stop
                active_set.add(best_candidate)
                remaining.remove(best_candidate)
                last_added_candidate = best_candidate
            else:
                break
        
        # Final evaluation with the constructed set
        result = self.evaluate(
            active_set, joint_positions, com, com_acc,
            floor_height, up_axis
        )
        result.pruned_contacts = candidate_joints - active_set
        result.seed = seed  # Store which joint was the seed
        
        # --- Group force splitting ---
        # Split representative's force between group members.
        # Rule 1: If a member is off the floor, it gets zero — all force
        #         goes to the grounded member. A joint off the ground
        #         cannot bear contact force.
        # Rule 2: If both are on the floor, use ZMP proximity (lever rule)
        #         to determine the split. The member closer to the ZMP
        #         bears more force (like a seesaw).
        FLOOR_CONTACT_MARGIN = 0.02  # 2cm above floor → still in contact
        for rep_j, members in group_members.items():
            rep_force = result.per_contact_force.get(rep_j, 0)
            if rep_force <= 0 or len(members) != 2:
                continue
            j_a, j_b = members
            
            # --- Floor contact check ---
            h_a = joint_positions[j_a, up_axis] - floor_height if j_a < J else 1.0
            h_b = joint_positions[j_b, up_axis] - floor_height if j_b < J else 1.0
            a_grounded = h_a <= FLOOR_CONTACT_MARGIN
            b_grounded = h_b <= FLOOR_CONTACT_MARGIN
            
            if not a_grounded and not b_grounded:
                # Neither on floor — keep force on representative
                continue
            elif a_grounded and not b_grounded:
                # Only j_a on floor
                frac_a = 1.0
            elif b_grounded and not a_grounded:
                # Only j_b on floor
                frac_a = 0.0
            else:
                # Both grounded — use ZMP lever rule
                zmp_hz = result.zmp_approx  # 2D horizontal ZMP
                pos_a_hz = joint_positions[j_a][plane_dims] if j_a < J else np.zeros(2)
                pos_b_hz = joint_positions[j_b][plane_dims] if j_b < J else np.zeros(2)
                
                d_a = np.linalg.norm(pos_a_hz - zmp_hz)
                d_b = np.linalg.norm(pos_b_hz - zmp_hz)
                d_total = d_a + d_b
                
                if d_total > 0.01:
                    frac_a = d_b / d_total  # closer to ZMP → more force
                else:
                    frac_a = 0.5
            
            result.per_contact_force[j_a] = rep_force * frac_a
            result.per_contact_force[j_b] = rep_force * (1.0 - frac_a)
            # Remove representative entry if it's not one of the members
            if rep_j not in (j_a, j_b):
                del result.per_contact_force[rep_j]
        
        # Sync force_array with per_contact_force after group splitting
        for j, f in result.per_contact_force.items():
            if j < len(result.force_array):
                result.force_array[j] = f
        
        # --- Augmentation: check for missing contacts ---
        residual_mag = np.linalg.norm(result.residual)
        if residual_mag > self.AUGMENT_RESIDUAL_THRESHOLD and all_joint_heights is not None:
            suggested = self._find_missing_contacts(
                active_set, joint_positions, com, com_acc,
                floor_height, up_axis, all_joint_heights, result,
                surface_distances=surface_distances,
                excluded_joints=excluded_joints
            )
            result.suggested_contacts = suggested
        
        return result
    
    # --- Contact spread radii ---
    # Each contact has a physical spread (patch size), not a point.
    # When two contacts are close, their patches may overlap with the
    # ZMP, making lever-arm physics unreliable. We blend toward equal
    # distribution when the effective lever arm (inter-contact distance
    # minus combined spread) is small relative to the spread.
    CONTACT_SPREAD_RADIUS = {
        10: 0.08, 11: 0.08,  # L_foot, R_foot
        28: 0.08, 29: 0.08,  # L_heel, R_heel
        20: 0.05, 21: 0.05,  # L_wrist, R_wrist
        22: 0.05, 23: 0.05,  # L_hand, R_hand
        4: 0.05, 5: 0.05,    # L_knee, R_knee
    }
    DEFAULT_SPREAD_RADIUS = 0.04

    def _solve_dynamic_equilibrium(self, balance_point_hz: np.ndarray,
                                    contact_pts_hz: np.ndarray,
                                    contact_joint_ids: list = None) -> np.ndarray:
        """Solve for vertical force distribution at contacts.
        
        Uses the ZMP (or CoM projection) as the balance point.
        Forces must sum to 1 (normalized) and produce zero moment about
        the balance point.
        
        Args:
            balance_point_hz: (2,) the point about which moments must balance
            contact_pts_hz: (N, 2) contact positions in horizontal plane
            contact_joint_ids: (N,) joint IDs for spread radius lookup
            
        Returns:
            fractions: (N,) force fraction at each contact (sum ≈ 1)
                       May contain negative values (= contact pulling)
        """
        N = len(contact_pts_hz)
        
        if N == 0:
            return np.array([])
        
        if N == 1:
            return np.array([1.0])
        
        # Build equilibrium matrix:
        # Row 0: Σ F_i = 1
        # Row 1: Σ F_i × (x_i - bp_x) = 0
        # Row 2: Σ F_i × (z_i - bp_z) = 0
        A = np.zeros((3, N))
        b = np.array([1.0, 0.0, 0.0])
        
        for i in range(N):
            A[0, i] = 1.0
            A[1, i] = contact_pts_hz[i, 0] - balance_point_hz[0]
            A[2, i] = contact_pts_hz[i, 1] - balance_point_hz[1]
        
        if N == 2:
            # Use proper moment equation instead of simple inverse distance
            # F1 × r1 = F2 × r2, F1 + F2 = 1
            # where r1, r2 are signed distances along the line connecting contacts
            d = contact_pts_hz[1] - contact_pts_hz[0]
            d_len = np.linalg.norm(d)
            if d_len < 1e-6:
                return np.array([0.5, 0.5])
            
            # Project balance point onto the line between contacts
            t = np.dot(balance_point_hz - contact_pts_hz[0], d) / (d_len * d_len)
            # t=0 → all force on contact 0, t=1 → all force on contact 1
            # F0 = 1-t, F1 = t (lever arm ratio)
            f0 = 1.0 - t
            f1 = t
            forces = np.array([f0, f1])
        
        elif N == 3:
            try:
                forces = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                forces = self._inverse_distance_weights(balance_point_hz, contact_pts_hz)
        
        else:
            # N > 3: minimum-norm solution
            try:
                AAT = A @ A.T
                forces = A.T @ np.linalg.solve(AAT, b)
            except np.linalg.LinAlgError:
                forces = self._inverse_distance_weights(balance_point_hz, contact_pts_hz)
    
        # --- Clamp fractions to prevent force amplification ---
        # When ZMP is outside the support polygon, the unconstrained
        # solution has fractions >> 1 and << 0, causing enormous force
        # on one contact. Clamp negatives to 0 and renormalize.
        # Track which contacts were clamped — they should not receive
        # force from spread blending (ZMP says they can't help).
        was_clamped = forces < -1e-6  # True for contacts that need "pulling"
        forces = np.maximum(forces, 0.0)
        fsum = np.sum(forces)
        if fsum > 1e-10:
            forces /= fsum
        else:
            forces = np.ones(N) / N
        
        # --- Contact spread regularization ---
        # Real contacts are patches, not points. When contacts are close,
        # the lever-arm between their patch edges is small, making force
        # distribution sensitive to ZMP noise. Blend toward equal
        # distribution ONLY among contacts that got positive solver force.
        # Clamped contacts (solver gave negative force) stay at zero —
        # blending should not resurrect them.
        if N >= 2:
            # Get spread radii for each contact
            spreads = np.array([
                self.CONTACT_SPREAD_RADIUS.get(
                    contact_joint_ids[i] if contact_joint_ids else -1,
                    self.DEFAULT_SPREAD_RADIUS
                )
                for i in range(N)
            ])
            
            # Only consider positive-force contact pairs for blending
            positive_mask = ~was_clamped
            n_positive = np.sum(positive_mask)
            
            if n_positive >= 2:
                # Use minimum pairwise effective lever arm among positive contacts
                min_physics_trust = 1.0
                for i in range(N):
                    if not positive_mask[i]:
                        continue
                    for j2 in range(i + 1, N):
                        if not positive_mask[j2]:
                            continue
                        dist = np.linalg.norm(contact_pts_hz[i] - contact_pts_hz[j2])
                        combined_spread = spreads[i] + spreads[j2]
                        effective_lever = max(0.0, dist - combined_spread)
                        avg_spread = 0.5 * combined_spread
                        if avg_spread > 1e-6:
                            trust = 1.0 - np.exp(-0.5 * (effective_lever / avg_spread) ** 2)
                        else:
                            trust = 1.0
                        min_physics_trust = min(min_physics_trust, trust)
                
                # Blend only among positive-force contacts
                equal_share = np.where(positive_mask, 1.0 / n_positive, 0.0)
                forces = min_physics_trust * forces + (1.0 - min_physics_trust) * equal_share
        
        return forces
    
    def _inverse_distance_weights(self, point_hz: np.ndarray,
                                   contact_pts_hz: np.ndarray) -> np.ndarray:
        """Fallback: inverse distance weighting."""
        distances = np.array([
            max(0.01, np.linalg.norm(p - point_hz))
            for p in contact_pts_hz
        ])
        inv_d = 1.0 / distances
        return inv_d / np.sum(inv_d)
    
    def _compute_residual(self, balance_point_hz: np.ndarray,
                          contact_pts_hz: np.ndarray,
                          per_contact_force: Dict[int, float],
                          active_contacts: List[int],
                          total_force_kg: float) -> np.ndarray:
        """Compute unbalanced force/moment residual.
        
        Returns (3,): [force_residual, moment_x_residual, moment_z_residual]
        All in kg or kg·m.
        """
        if not active_contacts:
            return np.array([total_force_kg, 0.0, 0.0])
        
        # Force residual
        total_applied = sum(max(0, f) for f in per_contact_force.values())
        force_residual = total_force_kg - total_applied
        
        # Moment residual about balance point
        moment_residual = np.zeros(2)
        for i, j in enumerate(active_contacts):
            f = max(0, per_contact_force.get(j, 0))
            r = contact_pts_hz[i] - balance_point_hz
            moment_residual += f * r
        
        return np.array([force_residual, moment_residual[0], moment_residual[1]])
    
    def _find_missing_contacts(self, current_contacts: Set[int],
                                joint_positions: np.ndarray,
                                com: np.ndarray,
                                com_acc: np.ndarray,
                                floor_height: float,
                                up_axis: int,
                                joint_heights: np.ndarray,
                                current_result: EvalResult,
                                surface_distances: Optional[np.ndarray] = None,
                                excluded_joints: Optional[set] = None
                                ) -> Set[int]:
        """Look for near-floor joints that could reduce the residual.
        
        Uses per-joint surface distances when available, otherwise falls
        back to global NEAR_FLOOR_THRESHOLD_M.
        """
        suggested = set()
        
        # Find joints near the floor but not in current contacts
        for j in range(min(len(joint_heights), self.num_joints)):
            if j in current_contacts:
                continue
            if excluded_joints and j in excluded_joints:
                continue
            
            # Per-joint surface distance: accounts for body padding
            if surface_distances is not None and j < len(surface_distances):
                tolerance = surface_distances[j] + 0.05  # surface offset + 5cm margin
            else:
                tolerance = self.NEAR_FLOOR_THRESHOLD_M
            
            if joint_heights[j] > tolerance:
                continue
            if joint_heights[j] < -0.05:  # Below floor — likely noise
                continue
            
            # Test: would adding this joint improve the solution?
            test_contacts = current_contacts | {j}
            test_result = self.evaluate(
                test_contacts, joint_positions, com, com_acc,
                floor_height, up_axis
            )
            
            test_residual = np.linalg.norm(test_result.residual)
            current_residual = np.linalg.norm(current_result.residual)
            
            if test_residual < current_residual * 0.7:  # 30% improvement
                suggested.add(j)
        
        return suggested
    
    def _empty_result(self, J: int, f_required: np.ndarray,
                      zmp_hz: np.ndarray) -> EvalResult:
        """Create an empty result when there are no contacts."""
        return EvalResult(
            per_contact_force={},
            necessity={},
            f_required=f_required,
            zmp_approx=zmp_hz,
            support_polygon=[],
            residual=np.array([f_required[1] / 9.81 if len(f_required) > 1 else 0, 0, 0]),
            pruned_contacts=set(),
            suggested_contacts=set(),
            force_array=np.zeros(J),
            necessity_array=np.zeros(J),
        )
