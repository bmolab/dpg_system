"""
Unified Contact Detection System

Multi-evidence consensus approach combining:
1. Sensory: height, chain suppression, horizontal speed
2. Kinematic: integrated Δh, deceleration, velocity trend (window-based)
3. Dynamic: CoM acceleration, freefall, XCoM/AM corrections
4. Directional: CoM→foot approach/depart, alt-support
5. Equilibrium: per-contact necessity, residual magnitude
6. Plausibility: liftoff composite from all above

Evidence streams feed into a state machine with hysteresis,
then force distribution uses XCoM-based ball-heel lever rule.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List
from collections import deque


# ─────────────────────────────────────────────────────────────────────
# SMPL joint constants
# ─────────────────────────────────────────────────────────────────────
FOOT_GROUPS = {
    'LF': [10, 28],   # L_foot, L_heel
    'RF': [11, 29],   # R_foot, R_heel
}
HAND_GROUPS = {
    'LH': [20, 22],   # L_wrist, L_hand
    'RH': [21, 23],   # R_wrist, R_hand
}
HEEL_MAP = {'LF': 28, 'RF': 29}
BALL_MAP = {'LF': 10, 'RF': 11}

PARENTS = [
    -1,  0,  0,  0,  1,  2,
     3,  4,  5,  6,  7,  8,
     9,  9,  9, 12, 13, 14,
    16, 17, 18, 19, 20, 21
]

CHAIN_CHILDREN = {
    4:  ([7, 10, 28], 0.02),
    5:  ([8, 11, 29], 0.02),
    7:  ([10, 28],    0.08),
    8:  ([11, 29],    0.08),
    18: ([20, 22],    0.05),
    19: ([21, 23],    0.05),
}


@dataclass
class UnifiedContactOptions:
    """Configuration for unified contact detection.
    
    Three-tier architecture:
      Gate streams (sensory, kinematic, dynamic): multiplicative, can veto.
      Support streams (directional, equilibrium, plausibility): weighted
        geometric mean, contributive but cannot individually veto.
      Equilibrium softening: when equilibrium necessity is high, gate
        veto power is reduced (noisy position data gets less trust).
    """
    # Iteration
    max_iterations: int = 1

    # Per-stream enables (gates and supports)
    enable_sensory: bool = True       # GATE
    enable_kinematic: bool = True     # GATE
    enable_dynamic: bool = True       # GATE
    enable_directional: bool = True   # SUPPORT
    enable_equilibrium: bool = True   # SUPPORT + gate softener
    enable_plausibility: bool = True  # SUPPORT

    # Support stream weights (for weighted geometric mean)
    weight_directional: float = 0.6
    weight_equilibrium: float = 1.0
    weight_plausibility: float = 0.8

    # Gate softening: how much equilibrium necessity can reduce gate veto
    # 0.0 = no softening (gates have absolute veto)
    # 0.5 = equilibrium necessity=1.0 raises gate floor to 0.5
    # 1.0 = equilibrium can fully override gates
    gate_softening: float = 1.0

    # Sensory
    height_sigma: float = 0.05
    height_ceiling: float = 0.30
    horiz_vel_threshold: float = 0.3
    horiz_vel_scale: float = 6.0

    # Kinematic
    kinematic_window: int = 10
    dh_rise_scale: float = 0.05
    dh_max_penalty: float = 5.0
    hspeed_scale: float = 0.3

    # State machine
    thresh_on: float = 0.45
    thresh_off: float = 0.30
    frames_on: int = 1
    frames_off: int = 3
    height_gate: float = 0.35

    # Foot model
    split_alpha: float = 0.3

    # Axis
    up_axis: int = 1


@dataclass
class UnifiedContactResult:
    """Output of unified contact detection."""
    contact_state: Dict[str, bool] = field(default_factory=dict)
    group_force: Dict[str, float] = field(default_factory=dict)
    pressure_array: np.ndarray = field(default_factory=lambda: np.zeros(30))
    per_evidence: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# Evidence Streams
# ─────────────────────────────────────────────────────────────────────

class SensoryEvidence:
    """Height-based contact probability with chain suppression."""

    def __init__(self, num_joints=30):
        self.num_joints = num_joints

    def compute(self, pos, floor_height, opts):
        """Returns per-group probability dict {gname: 0..1}."""
        up = opts.up_axis
        heights = pos[:, up] - floor_height

        all_groups = {}
        all_groups.update(FOOT_GROUPS)
        all_groups.update(HAND_GROUPS)

        result = {}
        for gname, joints in all_groups.items():
            valid = [j for j in joints if j < len(heights)]
            if not valid:
                result[gname] = 0.0
                continue

            # Height probability (Gaussian decay)
            min_h = min(heights[j] for j in valid)
            if min_h > opts.height_ceiling:
                result[gname] = 0.0
            elif min_h < 0:
                result[gname] = 0.95
            else:
                h_clamped = min(min_h, opts.height_ceiling)
                result[gname] = float(np.exp(-0.5 * (h_clamped / opts.height_sigma) ** 2))

            # Chain suppression
            for gj in valid:
                if gj in CHAIN_CHILDREN:
                    children, margin = CHAIN_CHILDREN[gj]
                    child_valid = [c for c in children if c < len(heights)]
                    if child_valid:
                        min_child = min(heights[c] for c in child_valid)
                        excess = heights[gj] - min_child - margin
                        if excess > 0.005:
                            suppression = 1.0 - np.exp(-20.0 * excess)
                            result[gname] *= (1.0 - suppression)

        return result


class KinematicEvidence:
    """Window-based movement analysis: Δh, deceleration, horizontal speed."""

    def __init__(self):
        self._pos_history = {}  # gname → deque of (y, x, z)

    def compute(self, pos, floor_height, dt, opts):
        """Returns per-group multiplier dict {gname: 0..1+}. >1 never, <1 = penalize."""
        up = opts.up_axis
        plane = [0, 2] if up == 1 else [0, 1]
        window = opts.kinematic_window

        all_groups = {}
        all_groups.update(FOOT_GROUPS)
        all_groups.update(HAND_GROUPS)

        # Update histories
        for gname, joints in all_groups.items():
            valid = [j for j in joints if j < pos.shape[0]]
            if not valid:
                continue
            rep_j = min(valid, key=lambda j: pos[j, up])
            p = pos[rep_j].copy()
            if gname not in self._pos_history:
                self._pos_history[gname] = deque(maxlen=window)
            self._pos_history[gname].append(p)

        result = {}
        for gname in all_groups:
            hist = self._pos_history.get(gname)
            if hist is None or len(hist) < 3:
                result[gname] = 1.0
                continue

            oldest, newest = hist[0], hist[-1]
            dh = newest[up] - oldest[up]

            dt_window = len(hist) / max(1.0 / max(dt, 1e-6), 1.0)
            dx = newest[plane[0]] - oldest[plane[0]]
            dz = newest[plane[1]] - oldest[plane[1]]
            hspeed = np.sqrt(dx**2 + dz**2) / max(dt_window, 0.001)

            # Vertical velocity (m/s, positive = rising)
            vy = dh / max(dt_window, 0.001)

            # Rising penalty: applied when foot moves upward over window
            if dh > opts.dh_rise_scale:
                rise_p = 1.0 + (opts.dh_max_penalty - 1.0) * min(
                    1.0, (dh - opts.dh_rise_scale) / (3.0 * opts.dh_rise_scale))
            else:
                rise_p = 1.0

            # Upward velocity penalty: even slow sustained rise should suppress
            # vy > 0.01 m/s ≈ 1mm/frame at 100fps triggers mild penalty
            if vy > 0.01:
                vy_factor = 1.0 + min(4.0, (vy / 0.05) ** 1.5)
                rise_p = max(rise_p, vy_factor)

            # Horizontal speed penalty (sqrt-scaled)
            if hspeed > opts.hspeed_scale:
                ratio = (hspeed - opts.hspeed_scale) / opts.hspeed_scale
                hspeed_p = 1.0 + np.sqrt(ratio) * 1.5
            else:
                hspeed_p = 1.0

            # Deceleration detection (velocity trend within window)
            mid = len(hist) // 2
            if mid > 0 and mid < len(hist) - 1:
                v_early = (hist[mid][up] - hist[0][up]) / max(mid * dt, 1e-6)
                v_late = (hist[-1][up] - hist[mid][up]) / max((len(hist) - mid) * dt, 1e-6)
                decel = v_early - v_late  # positive when slowing descent
                # Deceleration while descending = landing signal
                if v_early < -0.05 and decel > 0.1:
                    decel_bonus = max(0.5, 1.0 - decel * 2.0)
                    rise_p *= decel_bonus

            # Movement factor: higher = harder to accept
            factor = rise_p * hspeed_p
            # Convert to multiplier: 1/factor (so higher factor = lower probability)
            result[gname] = 1.0 / max(factor, 1.0)

        return result


class DynamicEvidence:
    """CoM acceleration analysis: freefall, jumping, support fraction."""

    def __init__(self):
        self._prev_com = None
        self._prev_vel = None

    def compute(self, com, com_vel, com_acc, total_mass, dt, opts):
        """Returns per-group multiplier dict. <1 = freefall suppression."""
        up = opts.up_axis
        g_mag = 9.81

        # Support fraction
        g_vec = np.zeros(3)
        g_vec[up] = -g_mag
        f_required = total_mass * (com_acc - g_vec)
        f_support_up = f_required[up]
        support_frac = float(np.clip(f_support_up / (total_mass * g_mag), 0.0, 1.5))

        # Freefall / jumping
        com_vy = com_vel[up] if com_vel is not None else 0.0

        factor = 1.0
        if support_frac < 0.3:
            factor *= max(0.1, support_frac / 0.3)
        if com_vy > 0.3:
            factor *= max(0.2, np.exp(-2.0 * com_vy))

        # Return uniform factor for all groups
        all_groups = list(FOOT_GROUPS.keys()) + list(HAND_GROUPS.keys())
        return {g: factor for g in all_groups}


class DirectionalEvidence:
    """CoM→foot direction and alt-support analysis."""

    def compute(self, com, com_vel, pos, group_states, floor_height, dt, opts):
        """Returns per-group multiplier dict."""
        up = opts.up_axis
        plane = [0, 2] if up == 1 else [0, 1]
        com_hz = com[plane]
        com_vel_hz = com_vel[plane] if com_vel is not None else np.zeros(2)

        result = {}
        for gname, joints in FOOT_GROUPS.items():
            valid = [j for j in joints if j < pos.shape[0]]
            if not valid:
                result[gname] = 1.0
                continue

            # Representative foot position
            rep_j = min(valid, key=lambda j: pos[j, up])
            foot_hz = pos[rep_j][plane]

            # CoM direction signal
            com_to_foot = foot_hz - com_hz
            dist = np.linalg.norm(com_to_foot)
            if dist > 0.01:
                approach_speed = np.dot(com_vel_hz, com_to_foot / dist)
                dir_signal = float(np.clip(approach_speed / 0.3, -1.0, 1.0))
            else:
                dir_signal = 0.0

            # Approaching foot → boost contact (multiplier > 1 not used, keep at 1)
            # Departing foot → reduce contact probability
            if dir_signal < 0:
                factor = 1.0 + 0.3 * abs(dir_signal)  # departing: mild suppression
            else:
                factor = 1.0  # approaching: neutral (sensory handles this)

            # Alt-support: is the OTHER foot grounded?
            other = 'RF' if gname == 'LF' else 'LF'
            other_on = group_states.get(other, False)
            other_joints = FOOT_GROUPS.get(other, [])
            other_valid = [j for j in other_joints if j < pos.shape[0]]
            if other_valid:
                other_h = min(pos[j, up] - floor_height for j in other_valid)
            else:
                other_h = 999.0

            # If other foot is solidly grounded, liftoff of this foot is more plausible
            if other_on and other_h < 0.05:
                factor *= 1.0  # neutral — alt_support used by plausibility
            result[gname] = 1.0 / max(factor, 1.0)

        # Hand groups: neutral
        for gname in HAND_GROUPS:
            result[gname] = 1.0

        return result


class EquilibriumEvidence:
    """Wraps DynamicFrameEvaluator to report necessity and residual."""

    def __init__(self, evaluator=None):
        self._evaluator = evaluator

    def compute(self, contact_probs, pos, com, com_acc, floor_height, opts):
        """Returns per-group multiplier based on equilibrium necessity."""
        if self._evaluator is None:
            all_groups = list(FOOT_GROUPS.keys()) + list(HAND_GROUPS.keys())
            return {g: 1.0 for g in all_groups}

        up = opts.up_axis

        # Threshold probabilities to binary contacts
        active = set()
        for gname, joints in {**FOOT_GROUPS, **HAND_GROUPS}.items():
            if contact_probs.get(gname, 0) > 0.3:
                for j in joints:
                    if j < pos.shape[0]:
                        h = pos[j, up] - floor_height
                        if h < opts.height_gate:
                            active.add(j)

        result_eval = self._evaluator.evaluate(
            active, pos, com, com_acc, floor_height, up)

        # Convert per-joint forces to per-group necessity multipliers
        result = {}
        for gname, joints in {**FOOT_GROUPS, **HAND_GROUPS}.items():
            total_force = sum(
                result_eval.per_contact_force.get(j, 0) for j in joints
                if j < pos.shape[0])
            # High force → necessary → boost probability
            if total_force > 5.0:
                result[gname] = 1.0
            elif total_force > 1.0:
                result[gname] = 0.5 + 0.5 * (total_force / 5.0)
            else:
                result[gname] = 0.3 + 0.2 * max(0, total_force)

        return result


class PlausibilityEvidence:
    """Liftoff plausibility composite from kinematic+dynamic+directional."""

    def compute(self, e_kinematic, e_dynamic, e_directional, e_equilibrium,
                group_states, opts):
        """Returns per-group multiplier. Low value = liftoff plausible."""
        result = {}
        for gname in list(FOOT_GROUPS.keys()) + list(HAND_GROUPS.keys()):
            if not group_states.get(gname, False):
                result[gname] = 1.0
                continue

            # Combine signals: if kinematic says rising + equilibrium says
            # low force + directional says departing → strong liftoff evidence
            kin = e_kinematic.get(gname, 1.0)
            dyn = e_dynamic.get(gname, 1.0)
            eq = e_equilibrium.get(gname, 1.0)

            # If foot is kinematically rising (kin < 1) AND
            # equilibrium says it's not needed (eq < 0.5):
            # plausibility is the combined evidence
            if kin < 0.8 and eq < 0.5:
                result[gname] = max(0.1, kin * eq * 4.0)
            else:
                result[gname] = 1.0

        return result


# ─────────────────────────────────────────────────────────────────────
# State Machine
# ─────────────────────────────────────────────────────────────────────

class ContactStateMachine:
    """Hysteresis-based ON/OFF state machine for contact groups."""

    def __init__(self):
        self._state = {}      # gname → bool
        self._frames = {}     # gname → frames in transition

    def update(self, landing_probs, liftoff_probs, group_heights, opts):
        """Update state machine with asymmetric probability tracks.
        
        Args:
            landing_probs: {gname: prob} computed with evidence_floor (OR-like, for OFF→ON)
            liftoff_probs: {gname: prob} computed with near-zero floor (AND-like, for ON→OFF)
            group_heights: {gname: min_height}
            opts: UnifiedContactOptions
        """
        for gname in landing_probs:
            prev = self._state.get(gname, False)
            frames = self._frames.get(gname, 0)
            min_h = group_heights.get(gname, 999.0)

            if prev:  # Currently ON — use liftoff_probs (strict, any stream can veto)
                prob = liftoff_probs.get(gname, 0.0)
                if prob < opts.thresh_off or min_h > opts.height_gate:
                    frames += 1
                    if frames >= opts.frames_off:
                        self._state[gname] = False
                        self._frames[gname] = 0
                    else:
                        self._frames[gname] = frames
                else:
                    self._frames[gname] = 0
            else:  # Currently OFF — use landing_probs (forgiving, one stream can override)
                prob = landing_probs.get(gname, 0.0)
                if prob > opts.thresh_on and min_h < opts.height_gate:
                    frames += 1
                    if frames >= opts.frames_on:
                        self._state[gname] = True
                        self._frames[gname] = 0
                    else:
                        self._frames[gname] = frames
                else:
                    self._frames[gname] = 0

        return dict(self._state)


# ─────────────────────────────────────────────────────────────────────
# Foot Pressure Model
# ─────────────────────────────────────────────────────────────────────

class FootPressureModel:
    """XCoM-based ball-heel force distribution with EMA smoothing."""

    def __init__(self):
        self._split_weights = {}  # gname → {j: weight}

    def distribute(self, contact_state, pos, xcom_hz, total_mass,
                   floor_height, opts):
        """Distribute force within foot groups using XCoM lever rule.

        Returns pressure array (J,) in kg.
        """
        up = opts.up_axis
        plane = [0, 2] if up == 1 else [0, 1]
        J = pos.shape[0]
        pressure = np.zeros(J)

        # Count active foot groups for force sharing
        active_feet = [g for g in FOOT_GROUPS if contact_state.get(g, False)]
        if not active_feet:
            return pressure

        # Group-level force split based on XCoM proximity
        group_forces = {}
        if len(active_feet) == 1:
            group_forces[active_feet[0]] = total_mass
        else:
            # XCoM-based lever rule between foot group centroids
            centroids = {}
            for gname in active_feet:
                joints = FOOT_GROUPS[gname]
                valid = [j for j in joints if j < J]
                if valid:
                    centroids[gname] = np.mean([pos[j][plane] for j in valid], axis=0)

            if len(centroids) >= 2:
                # Inverse distance to XCoM
                inv_dists = {}
                for g, c in centroids.items():
                    d = max(0.01, np.linalg.norm(c - xcom_hz))
                    inv_dists[g] = 1.0 / d
                total_inv = sum(inv_dists.values())
                for g in active_feet:
                    group_forces[g] = total_mass * inv_dists.get(g, 0) / max(total_inv, 1e-6)
            else:
                for g in active_feet:
                    group_forces[g] = total_mass / len(active_feet)

        # Within-group ball-heel split using XCoM lever rule + EMA
        for gname in active_feet:
            gforce = group_forces.get(gname, 0)
            ball_j = BALL_MAP[gname]
            heel_j = HEEL_MAP[gname]
            members = [ball_j, heel_j]
            valid = [j for j in members if j < J]

            if len(valid) < 2 or gforce <= 0:
                for j in valid:
                    pressure[j] = gforce / max(len(valid), 1)
                continue

            # XCoM lever rule: project XCoM onto ball-heel line
            ball_hz = pos[ball_j][plane]
            heel_hz = pos[heel_j][plane]
            foot_vec = ball_hz - heel_hz
            foot_len = np.linalg.norm(foot_vec)

            if foot_len > 0.01:
                t = np.dot(xcom_hz - heel_hz, foot_vec) / (foot_len ** 2)
                t = float(np.clip(t, 0.0, 1.0))
                raw_weights = {ball_j: t, heel_j: 1.0 - t}
            else:
                raw_weights = {ball_j: 0.5, heel_j: 0.5}

            # EMA smoothing — new joints ramp from 0
            prev = self._split_weights.get(gname, {})
            smooth = {}
            alpha = opts.split_alpha
            for j in valid:
                pw = prev.get(j, 0.0)
                rw = raw_weights.get(j, 0.5)
                smooth[j] = alpha * rw + (1.0 - alpha) * pw

            # Normalize
            w_sum = sum(smooth.values())
            if w_sum > 0:
                for j in smooth:
                    smooth[j] /= w_sum

            self._split_weights[gname] = smooth

            for j in valid:
                pressure[j] = gforce * smooth.get(j, 0.5)

        # Also add hand group pressure (simple equal split)
        active_hands = [g for g in HAND_GROUPS if contact_state.get(g, False)]
        for gname in active_hands:
            joints = HAND_GROUPS[gname]
            valid = [j for j in joints if j < J]
            force_per = 5.0 / max(len(valid), 1)  # nominal hand pressure
            for j in valid:
                pressure[j] = force_per

        return pressure


# ─────────────────────────────────────────────────────────────────────
# Main Detector
# ─────────────────────────────────────────────────────────────────────

class UnifiedContactDetector:
    """Unified multi-evidence contact detection with consensus."""

    def __init__(self, framerate=60.0, total_mass_kg=75.0, evaluator=None):
        self.framerate = framerate
        self.total_mass = total_mass_kg

        self.sensory = SensoryEvidence()
        self.kinematic = KinematicEvidence()
        self.dynamic = DynamicEvidence()
        self.directional = DirectionalEvidence()
        self.equilibrium = EquilibriumEvidence(evaluator)
        self.plausibility = PlausibilityEvidence()
        self.state_machine = ContactStateMachine()
        self.foot_model = FootPressureModel()

    def process_frame(self, pos, com, com_vel, com_acc,
                      floor_height, dt, opts=None):
        """Main entry point.

        Args:
            pos: (J, 3) joint positions
            com: (3,) center of mass
            com_vel: (3,) CoM velocity
            com_acc: (3,) CoM acceleration
            floor_height: float
            dt: float time step
            opts: UnifiedContactOptions

        Returns:
            UnifiedContactResult
        """
        if opts is None:
            opts = UnifiedContactOptions()

        up = opts.up_axis
        plane = [0, 2] if up == 1 else [0, 1]
        all_group_names = list(FOOT_GROUPS.keys()) + list(HAND_GROUPS.keys())
        NEUTRAL = {g: 1.0 for g in all_group_names}

        # --- Compute evidence streams ---
        e_sensory = (self.sensory.compute(pos, floor_height, opts)
                     if opts.enable_sensory else dict(NEUTRAL))

        e_kinematic = (self.kinematic.compute(pos, floor_height, dt, opts)
                       if opts.enable_kinematic else dict(NEUTRAL))

        e_dynamic = (self.dynamic.compute(com, com_vel, com_acc,
                                          self.total_mass, dt, opts)
                     if opts.enable_dynamic else dict(NEUTRAL))

        group_states = dict(self.state_machine._state)
        e_directional = (self.directional.compute(
            com, com_vel, pos, group_states, floor_height, dt, opts)
            if opts.enable_directional else dict(NEUTRAL))

        # ─── Three-tier combination ─────────────────────────────────
        #
        # Tier 1: GATE product (sensory × kinematic × dynamic)
        #   Physical constraints — each can independently veto.
        #
        # Tier 2: Equilibrium SOFTENING
        #   When equilibrium necessity is high, gate floor is raised
        #   so noisy position data can't fully veto a necessary contact.
        #
        # Tier 3: SUPPORT geometric mean (directional, equilibrium, plausibility)
        #   Contributive evidence — weighted geometric mean, no single
        #   support can veto the result.
        #
        # combined = softened_gate × support_geo_mean
        # ─────────────────────────────────────────────────────────────

        # --- Tier 1: Raw gate product ---
        raw_gates = {}
        for g in all_group_names:
            gate = 1.0
            if opts.enable_sensory:
                gate *= e_sensory.get(g, 1.0)
            if opts.enable_kinematic:
                gate *= e_kinematic.get(g, 1.0)
            if opts.enable_dynamic:
                gate *= e_dynamic.get(g, 1.0)
            raw_gates[g] = gate

        # Initial contact probs for equilibrium bootstrap (gate-only)
        contact_probs = dict(raw_gates)

        # --- Consensus iteration ---
        e_eq = dict(NEUTRAL)
        e_plaus = dict(NEUTRAL)

        for _iter in range(opts.max_iterations):
            e_eq = (self.equilibrium.compute(
                contact_probs, pos, com, com_acc, floor_height, opts)
                if opts.enable_equilibrium else dict(NEUTRAL))

            e_plaus = (self.plausibility.compute(
                e_kinematic, e_dynamic, e_directional, e_eq,
                group_states, opts)
                if opts.enable_plausibility else dict(NEUTRAL))

            # --- Tier 2: Equilibrium gate softening ---
            # eq_necessity is 0..1 indicating how much equilibrium needs this foot.
            # Higher necessity → higher gate floor → less veto from noisy data.
            # BUT: softening is also gated by the sensory signal — if sensory
            # says the foot is physically impossible (very high), equilibrium
            # can't override regardless of necessity. This prevents the
            # bootstrap loop where equilibrium asserts necessity during descent.
            softened_gates = {}
            target = opts.thresh_on * 1.1  # aim slightly above threshold
            for g in all_group_names:
                raw = raw_gates[g]
                eq_val = e_eq.get(g, 0.5)
                # Map equilibrium evidence to necessity (0=not needed, 1=essential)
                necessity = max(0.0, (eq_val - 0.3) / 0.7)  # normalize to 0..1
                # Gate the softening by sensory plausibility:
                # sensory > 0.005 = foot is at least physically near the floor
                sensory_val = e_sensory.get(g, 0.0) if opts.enable_sensory else 1.0
                # Scale softening: sensory < 0.005 → no softening,
                # sensory 0.005..0.2 → partial, sensory > 0.2 → full
                sensory_gate = float(np.clip((sensory_val - 0.005) / 0.195, 0.0, 1.0))
                gate_floor = necessity * sensory_gate * opts.gate_softening * target
                softened_gates[g] = max(raw, gate_floor)

            # --- Tier 3: Support geometric mean ---
            support_evidences = [e_directional, e_eq, e_plaus]
            support_weights = [
                opts.weight_directional if opts.enable_directional else 0,
                opts.weight_equilibrium if opts.enable_equilibrium else 0,
                opts.weight_plausibility if opts.enable_plausibility else 0,
            ]

            def _support_geo_mean(group):
                """Weighted geometric mean of support streams."""
                log_sum = 0.0
                w_total = 0.0
                for ev_dict, w in zip(support_evidences, support_weights):
                    if w <= 0:
                        continue
                    val = max(1e-6, ev_dict.get(group, 1.0))
                    log_sum += w * np.log(val)
                    w_total += w
                if w_total > 0:
                    return float(np.exp(log_sum / w_total))
                return 1.0

            # Landing probs: softened gates × support geo mean
            for g in all_group_names:
                contact_probs[g] = softened_gates[g] * _support_geo_mean(g)

        # --- Liftoff probs: gates with partial equilibrium softening ---
        # For groups currently ON, if equilibrium says the contact is
        # necessary, blend some of the softened gate into the liftoff
        # probability. This prevents the asymmetry between landing
        # (softened) and liftoff (raw) from creating limit cycles where
        # a firmly planted foot oscillates because its raw gate sits
        # just below thresh_off.
        #
        # For groups currently OFF or not equilibrium-necessary, the
        # liftoff probability uses raw gates (strict), preserving clean
        # liftoff detection for genuine foot raises.
        NEUTRAL_THRESH = 0.95

        def _active_support_geo_mean(group):
            """Geo mean of only support streams deviating from neutral."""
            log_sum = 0.0
            w_total = 0.0
            for ev_dict, w in zip(support_evidences, support_weights):
                if w <= 0:
                    continue
                val = ev_dict.get(group, 1.0)
                if val >= NEUTRAL_THRESH:
                    continue  # Skip neutral supports
                val = max(1e-6, val)
                log_sum += w * np.log(val)
                w_total += w
            if w_total > 0:
                return float(np.exp(log_sum / w_total))
            return 1.0  # All supports neutral → no penalty

        liftoff_probs = {}
        for g in all_group_names:
            raw_liftoff = raw_gates[g] * _active_support_geo_mean(g)
            
            # Partial softening for currently-ON, equilibrium-necessary contacts
            if group_states.get(g, False):
                eq_val = e_eq.get(g, 0.5)
                necessity = max(0.0, (eq_val - 0.3) / 0.7)
                if necessity > 0 and opts.gate_softening > 0:
                    # Blend toward the softened gate value
                    softened_liftoff = softened_gates[g] * _active_support_geo_mean(g)
                    blend = necessity * opts.gate_softening * 0.5  # half-strength
                    raw_liftoff = raw_liftoff * (1.0 - blend) + softened_liftoff * blend
            
            liftoff_probs[g] = raw_liftoff

        # --- Group heights for state machine ---
        all_groups = {**FOOT_GROUPS, **HAND_GROUPS}
        group_heights = {}
        for gname, joints in all_groups.items():
            valid = [j for j in joints if j < pos.shape[0]]
            if valid:
                group_heights[gname] = min(pos[j, up] - floor_height for j in valid)
            else:
                group_heights[gname] = 999.0

        # --- State machine (asymmetric: landing=softened, liftoff=strict) ---
        contact_state = self.state_machine.update(
            contact_probs, liftoff_probs, group_heights, opts)

        # --- XCoM-based force distribution ---
        com_h = com[up] - floor_height
        g_mag = 9.81
        xcom_scale = np.sqrt(max(com_h, 0.01) / g_mag)
        xcom = com.copy()
        if com_vel is not None:
            xcom += com_vel * xcom_scale
        xcom_hz = xcom[plane]

        pressure = self.foot_model.distribute(
            contact_state, pos, xcom_hz, self.total_mass,
            floor_height, opts)

        # --- Build result ---
        per_evidence = {}
        for g in all_group_names:
            per_evidence[g] = {
                'sensory': e_sensory.get(g, 1.0),
                'kinematic': e_kinematic.get(g, 1.0),
                'dynamic': e_dynamic.get(g, 1.0),
                'directional': e_directional.get(g, 1.0),
                'equilibrium': e_eq.get(g, 1.0),
                'plausibility': e_plaus.get(g, 1.0),
                'raw_gate': raw_gates.get(g, 1.0),
                'softened_gate': softened_gates.get(g, 1.0),
                'combined': contact_probs.get(g, 0.0),
                'liftoff': liftoff_probs.get(g, 0.0),
            }

        return UnifiedContactResult(
            contact_state=contact_state,
            group_force={g: float(sum(pressure[j] for j in all_groups[g]
                                      if j < len(pressure)))
                         for g in all_group_names},
            pressure_array=pressure,
            per_evidence=per_evidence,
        )

