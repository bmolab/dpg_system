"""
Log-Odds Continuous Contact Estimator

Replaces binary state-machine contact detection with a continuous
intensity model based on additive log-odds accumulation.

Key properties:
  - No stream can veto: all evidence is additive (log-odds space)
  - No state machine: continuous intensity via sigmoid
  - Temporal memory: accumulated evidence decays but persists,
    providing natural hysteresis without hard thresholds
  - Every stream's contribution is a signed number, observable
    and independently togglable

Architecture:
  Each frame, for each contact group:
    1. Accumulated log-odds decays toward 0 (neutral)
    2. Each evidence stream computes a signed log-odds increment
    3. Increments are summed and added to accumulated state
    4. Sigmoid maps accumulated state to intensity [0, 1]
"""

import json
import os
import numpy as np
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, List
from collections import deque


# ─────────────────────────────────────────────────────────────────────
# Joint group constants (shared with contact_unified.py)
# ─────────────────────────────────────────────────────────────────────
FOOT_GROUPS = {
    'LF': [10, 28],   # L_foot (ball), L_heel
    'RF': [11, 29],   # R_foot (ball), R_heel
}
HAND_GROUPS = {
    'LH': [20, 22],   # L_wrist, L_hand
    'RH': [21, 23],   # R_wrist, R_hand
}
BODY_GROUPS = {
    'LK': [4],    # L_Knee
    'RK': [5],    # R_Knee
    'LH2': [1],   # L_Hip
    'RH2': [2],   # R_Hip
    'LE': [18],   # L_Elbow
    'RE': [19],   # R_Elbow
    'HD': [15],   # Head
    'PV': [0],    # Pelvis
}
PRIMARY_GROUPS = {**FOOT_GROUPS, **HAND_GROUPS}
ALL_GROUPS = {**PRIMARY_GROUPS, **BODY_GROUPS}
HEEL_MAP = {'LF': 28, 'RF': 29}
BALL_MAP = {'LF': 10, 'RF': 11}
# Push-off joint pairs: (secondary, primary)
# During push-off, the secondary joint rises relative to the primary.
# Feet: ankle rises relative to ball; Hands: wrist rises relative to hand
PUSHOFF_JOINTS = {
    'LF': (7, 10),   # L_Ankle, L_Ball
    'RF': (8, 11),   # R_Ankle, R_Ball
    'LH': (20, 22),  # L_Wrist, L_Hand
    'RH': (21, 23),  # R_Wrist, R_Hand
}

def get_active_groups(opts):
    """Return the contact groups to evaluate based on options."""
    if getattr(opts, 'enable_body_contacts', False):
        return ALL_GROUPS
    return PRIMARY_GROUPS


# ─────────────────────────────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────────────────────────────

@dataclass
class LogOddsContactOptions:
    """Configuration for log-odds contact estimator."""

    up_axis: int = 1

    # Temporal accumulator
    decay_rate: float = 0.90      # Per-frame decay toward neutral (0.9 = 10% decay)
    initial_logodds: float = 0.0  # 0 = 50% prior
    max_logodds: float = 6.0      # Clamp magnitude to prevent over-certainty
    sigmoid_temperature: float = 2.0  # Shallower sigmoid → more gradual contact
                                       # T=1.0: binary (0.05-0.95 over 6 LO)
                                       # T=2.0: gradual (0.05-0.95 over 12 LO)

    # Stream enables (3-stream architecture)
    enable_height: bool = True
    enable_kinematic: bool = True      # Unified kinematic stream (approach angle + touchdown + settled)
    enable_structural: bool = True     # Frame evaluator structural necessity
    enable_divergence: bool = True     # Foot-CoM relative velocity divergence
    # Legacy enables (kept for backward compatibility / A/B testing)
    enable_vertical_kinematic: bool = False  # Old unified vertical-only stream
    enable_hspeed: bool = False              # Old separate horizontal speed stream
    enable_equilibrium: bool = False         # Old ZMP proximity stream
    enable_velocity: bool = False
    enable_trajectory: bool = False
    enable_touchdown: bool = False

    # Stream weights (multiplier on each stream's raw log-odds)
    weight_height: float = 1.0
    weight_kinematic: float = 0.8      # Tuned: needs strong negative for swing
                                        # suppression; positive side tamed to avoid
                                        # pre-deceleration contact (divergence handles
                                        # planted/swing discrimination now)
    weight_structural: float = 1.0     # Structural necessity stream
    weight_divergence: float = 1.0     # Foot-CoM divergence stream
    # Legacy weights (for backward compatibility)
    weight_vertical_kinematic: float = 1.0
    weight_hspeed: float = 1.0
    weight_equilibrium: float = 1.0
    weight_velocity: float = 1.0
    weight_trajectory: float = 1.0
    weight_touchdown: float = 1.0

    # Height stream parameters
    height_contact_zone: float = 0.02    # Below this: strong positive
    height_ambiguous_lo: float = 0.05    # Transition zone lower bound
    height_ambiguous_hi: float = 0.10    # Transition zone upper bound
    height_clear_zone: float = 0.20      # Above this: strong negative
    height_contact_logodds: float = 1.0  # Log-odds when clearly on floor
    height_clear_logodds: float = -2.0   # Log-odds when clearly above floor

    # Velocity stream parameters
    vel_ascend_fast: float = 0.15        # m/s: above this → strong negative
    vel_ascend_slow: float = 0.03        # m/s: above this → moderate negative
    vel_descend_decel_zone: float = 0.08 # height (m) below which deceleration matters
    vel_ascend_fast_logodds: float = -2.0
    vel_ascend_slow_logodds: float = -0.5
    vel_decel_logodds: float = 0.8       # deceleration near floor
    vel_stationary_logodds: float = 0.2  # near-zero vy + low height

    # Trajectory stream parameters
    traj_window: int = 5                 # frames of history (shorter = faster reaction)
    traj_stable_low_logodds: float = 0.5 # consistently low + stable
    traj_rising_logodds: float = -1.5    # sustained upward trend
    traj_falling_logodds: float = 0.3    # sustained downward trend
    traj_height_thresh: float = 0.08     # below this = "low"
    traj_stable_std_thresh: float = 0.01 # std below this = "stable"
    traj_trend_thresh: float = 0.002     # slope per frame threshold
    traj_hz_travel_logodds: float = -2.0 # consistent horizontal travel
    traj_hz_travel_thresh: float = 0.05  # min horizontal distance over window (m)

    # Horizontal speed stream parameters (legacy, used only when enable_hspeed=True)
    hspeed_fast: float = 0.3             # m/s: above this → strong negative
    hspeed_slow: float = 0.08            # m/s: below this → neutral/positive
    hspeed_fast_logodds: float = -2.5    # swinging leg
    hspeed_planted_logodds: float = 0.1  # stationary + low = planted

    # Equilibrium stream parameters (legacy, used only when enable_equilibrium=True)
    eq_gravity: float = 9.81
    eq_freefall_thresh: float = 0.7
    eq_necessity_logodds: float = 1.0
    eq_unnecessary_logodds: float = -0.3
    eq_assertion_logodds: float = 2.0

    # Pressure mapping
    intensity_deadzone: float = 0.05     # Below this intensity → zero pressure

    # Body contacts (knees, elbows, head, pelvis)
    enable_body_contacts: bool = False   # OFF by default — preserves hand/foot quality
    body_decay_rate: float = 0.95        # Slower decay for body contacts (stickier)

    # Foot model
    split_alpha: float = 0.3             # EMA for ball-heel split

    # Valving
    enable_valving: bool = False         # Cross-stream contextual adjustment

    # Touchdown sub-signal parameters (dual-EMA crossover, used in KinematicStream)
    td_alpha_fast: float = 0.5           # Fast EMA: responds in ~2 frames
    td_alpha_slow: float = 0.15          # Slow EMA: responds in ~7 frames
    td_descent_gate: float = 0.1         # Minimum abs(slow_ema) to fire (m/s descent)
    td_height_gate: float = 0.12         # Only fire when below this height (m)
    td_descent_scale: float = 1.0        # Normalize abs(slow_ema) to strength 0-1
    td_landing_logodds: float = 1.5      # Log-odds burst on touchdown crossing
    td_settling_logodds: float = 0.3     # Weaker positive during post-landing settling

    # VerticalKinematic stream parameters (legacy, used only when enable_vertical_kinematic=True)
    vk_phase_decay: float = 0.85
    vk_dead_zone: float = 0.3
    vk_max_positive_logodds: float = 1.5
    vk_max_negative_logodds: float = -2.0

    # KinematicStream parameters (unified: approach angle + touchdown + settled)
    kin_phase_decay: float = 0.85        # Per-frame decay toward 0 (uncertain)
    kin_dead_zone: float = 0.3           # Phase magnitude below this → output 0
    kin_max_positive_logodds: float = 1.5 # Output at phase = +1.0 (grounded)
    kin_max_negative_logodds: float = -2.0 # Output at phase = -1.0 (liftoff)
    # Approach angle sub-signal
    kin_angle_shallow_deg: float = 20.0  # Below this = sweeping (when not planted)
    kin_angle_steep_deg: float = 50.0    # Above this = landing/liftoff
    kin_angle_speed_lo: float = 0.1      # Speed below this → angle confidence = 0
    kin_angle_speed_hi: float = 0.5      # Speed above this → angle confidence = 1
    kin_angle_sweep_push: float = -0.15  # Airborne push for shallow angle (not planted)
    kin_angle_land_push: float = 0.10    # Grounded push for steep descending (not planted)
    kin_angle_liftoff_push: float = -0.12 # Airborne push for steep ascending (planted)
    kin_angle_planted_lift_push: float = -0.12  # Airborne push for steep while planted

    # Never-settled sub-signal (detects foot that never reached floor during approach)
    kin_settle_approach_zone: float = 0.08   # Height below which we track approach epoch
    kin_settle_vy_thresh: float = 0.05       # |vy| below this counts as 'vertically still'
    kin_settle_frames: int = 15              # consecutive low-vy frames needed to be 'settled'
    kin_settle_never_push: float = -0.15     # Airborne push when foot never settled
    kin_settle_speed_gate: float = 0.18      # Only apply when speed < this (low-speed gap filler)
    kin_settle_rise_reset: float = 0.012     # Rise above min_h to trigger epoch reset

    # Apex crossover impulse — counterpart of the landing crossover in
    # td_impulse.  Fires a single-frame negative impulse when fast_ema
    # crosses slow_ema downward (rising momentum decaying), the slow_ema
    # is still positive (foot was rising), and the foot is in the
    # decision-relevant height band.  Symmetric to the landing impulse:
    # vy approaches 0 from the negative side → +impulse (contact indicator);
    # vy approaches 0 from the positive side → −impulse (anti-contact).
    #
    # td_apex_scale is intentionally smaller than td_descent_scale because
    # rise velocities are gravity-limited from below (driven by muscles)
    # while fall velocities are gravity-driven from above — landings
    # naturally arrive at much higher speeds than apex peaks reach, so
    # they shouldn't share the same strength normalisation.
    td_apex_logodds: float = -0.4            # Magnitude at full strength
    td_apex_height_gate: float = 0.18        # m; slightly more permissive than landing
    td_apex_rise_gate: float = 0.05          # m/s; minimum slow_ema to qualify as "was rising"
    td_apex_scale: float = 0.2               # m/s; rise speed at which strength saturates

    # Structural stream parameters (frame evaluator wrapper)
    struct_gravity: float = 9.81
    struct_force_strong: float = 10.0    # kg: above this → strong positive
    struct_force_moderate: float = 3.0   # kg: above this → moderate positive
    struct_force_mild: float = 0.5       # kg: above this → mild positive
    struct_necessary_logodds: float = 1.5
    struct_moderate_logodds: float = 0.8
    struct_mild_logodds: float = 0.3
    struct_unnecessary_logodds: float = -0.8
    struct_pulling_logodds: float = -1.5
    struct_freefall_logodds: float = -2.0
    struct_freefall_thresh: float = 0.15  # support fraction below this = freefall
    struct_candidate_height: float = 0.15 # joints within this height are candidates
    # Push-off evidence: scales continuously with excess support fraction
    struct_pushoff_logodds: float = 2.0   # max positive for near-floor groups
    struct_pushoff_height: float = 0.04   # only apply to groups with joints below this
    # Inverted pendulum: CoM falling → trailing foot is pivot,
    # leading foot is proven NOT fully supporting (pendulum still swinging)
    struct_pendulum_vy_thresh: float = -0.03  # m/s: below this = CoM is falling
    struct_pendulum_leading_logodds: float = -0.3  # negative for leading (not yet full support)
    struct_pendulum_min_separation: float = 0.15   # feet must be this far apart for directionality
    # Per-frame ZMP/share noise can flip the lever-rule split many times
    # within a single stance. Apply an EMA on per-group forces before the
    # share decision to suppress that noise. alpha=1.0 disables (no
    # smoothing); alpha=0.2 ≈ 5-frame effective window.
    struct_force_ema_alpha: float = 1.0

    # Effort-relief biomechanical prior: when a trunk joint's gravity
    # torque exceeds the threshold, hand candidates on the lean side are
    # promoted as soft positive evidence even if the FE's ZMP solution
    # gave them no load. Default OFF — preserves prior behavior.
    fe_relief_enable: bool = False
    fe_relief_strain_threshold: float = 25.0   # N·m at spine joints
    struct_relief_logodds: float = 0.3         # per-frame log-odds added when promoted

    # Divergence stream parameters
    div_min_com_speed: float = 0.1    # CoM speed below this → neutral (m/s)
    div_alignment_scale: float = 1.5  # alignment magnitude for full strength
    div_max_logodds: float = 1.5      # max evidence magnitude
    div_ground_gate_scale: float = 0.015  # pro-contact gate: 5mm→72%, 1cm→51%, 2cm→26%
    # EMA on the alignment signal before log-odds mapping.  Smoothing the
    # raw foot-CoM divergence damps the snap from anti-contact (swing) to
    # pro-contact (stance) when the foot stops racing ahead of the CoM but
    # is still in late swing — the foot reaches its limit and is unlikely
    # to be a contact moment, so delaying the switch is non-destructive.
    # 1.0 = no smoothing (raw passthrough); smaller = heavier lag.
    div_alignment_ema_alpha: float = 0.30

    # Pressure model: intra-group rotation-rate suppression
    rise_suppression_scale: float = 0.001  # m/frame differential rise for full suppression


@dataclass
class LogOddsContactResult:
    """Output of one frame of log-odds contact estimation."""
    # Per-group results
    intensity: Dict[str, float] = field(default_factory=dict)
    log_odds_state: Dict[str, float] = field(default_factory=dict)
    per_stream: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Per-group context from streams (for valving & diagnostics)
    # e.g. stream_context['LF'] = {'decel': 0.8, 'vy': -0.5, 'height_m': 0.01, ...}
    stream_context: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Joint-level output
    pressure_array: np.ndarray = field(default_factory=lambda: np.zeros(30))

    # For downstream compatibility (binary contact based on intensity threshold)
    contact_state: Dict[str, bool] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# Evidence Streams
# ─────────────────────────────────────────────────────────────────────

class HeightStream:
    """Height-based evidence: distance from floor → log-odds increment.

    Uses a piecewise mapping rather than Gaussian to avoid the fragile
    threshold regime that caused oscillation in the unified method.

    Zones:
      h < contact_zone (2cm):  strong positive  (+2.0)
      h ∈ [contact, ambig_lo]: interpolated positive → weak positive
      h ∈ [ambig_lo, ambig_hi]: interpolated weak positive → neutral
      h ∈ [ambig_hi, clear]:   interpolated neutral → negative
      h > clear_zone (20cm):   strong negative  (-3.0)
    """

    def compute(self, pos, floor_height, opts, surface_dists=None):
        """Compute per-group log-odds increment from height.

        Args:
            pos: (J, 3) joint positions
            floor_height: float
            opts: LogOddsContactOptions
            surface_dists: Optional (J,) per-joint distance from joint
                center to mesh surface in floor-ward direction

        Returns:
            Dict[group_name, float]: log-odds increment per group
        """
        up = opts.up_axis
        heights = pos[:, up] - floor_height
        # Surface-correct using hybrid distances: max extents for primary
        # joints (foot/hand), min distances (skin thickness) for body joints.
        # The caller provides the appropriate hybrid surface_dists array.
        if surface_dists is not None:
            n = min(len(heights), len(surface_dists))
            heights = heights.copy()  # don't mutate input
            heights[:n] -= surface_dists[:n]

        result = {}
        for gname, joints in get_active_groups(opts).items():
            valid = [j for j in joints if j < len(heights)]
            if not valid:
                result[gname] = 0.0
                continue

            # Use minimum height of group members
            min_h = min(heights[j] for j in valid)

            # Below floor → strong contact
            if min_h < 0:
                result[gname] = opts.height_contact_logodds
                continue

            # Piecewise interpolation
            cz = opts.height_contact_zone
            alo = opts.height_ambiguous_lo
            ahi = opts.height_ambiguous_hi
            clr = opts.height_clear_zone
            lo_pos = opts.height_contact_logodds
            lo_neg = opts.height_clear_logodds

            if min_h <= cz:
                # Contact zone: strong positive
                result[gname] = lo_pos
            elif min_h <= alo:
                # Interpolate: strong positive → weak positive
                t = (min_h - cz) / (alo - cz)
                result[gname] = lo_pos * (1.0 - t) + 0.5 * t
            elif min_h <= ahi:
                # Interpolate: weak positive → neutral
                t = (min_h - alo) / (ahi - alo)
                result[gname] = 0.5 * (1.0 - t) + 0.0 * t
            elif min_h <= clr:
                # Interpolate: neutral → negative
                t = (min_h - ahi) / (clr - ahi)
                result[gname] = 0.0 * (1.0 - t) + lo_neg * 0.5 * t
            else:
                # Clear zone: strong negative
                result[gname] = lo_neg

        return result


class VerticalKinematicStream:
    """Unified vertical kinematic evidence via soft phase integration.

    Fuses three sub-signals that measure the same physical process
    (the foot's contact lifecycle) into one continuous phase variable:

        phase: -1.0 (airborne/liftoff) <-- 0.0 (uncertain) --> +1.0 (grounded)

    Sub-signals:
      1. Trajectory: epoch-based slope prediction (approach/departure)
      2. Touchdown: dual-EMA crossover impulse (impact/liftoff moment)
      3. Velocity state: vy magnitude (settled/ascending)

    The phase decays toward 0 each frame. Output log-odds is mapped
    through a dead zone: phase near 0 produces no evidence, allowing
    height/hspeed/equilibrium to drive the decision.
    """

    # Trajectory epoch detection thresholds (in m/s, converted per-frame)
    HEIGHT_VEL_BREAK_MPS = 0.25

    # Trajectory confidence ramp
    MIN_CONFIDENCE_FRAMES = 2
    FULL_CONFIDENCE_FRAMES = 7

    def __init__(self):
        # Per-group soft phase
        self._phase = {}           # gname -> float in [-1, 1]

        # Trajectory sub-signal state
        self._h_history = {}       # gname -> list of heights (post-epoch)
        self._traj_prev_h = {}     # gname -> previous frame height
        self._traj_prev_dh = {}    # gname -> previous height velocity

        # Touchdown sub-signal state (dual-EMA on vy)
        self._td_prev_h = {}       # gname -> previous height
        self._td_fast_ema = {}     # gname -> fast EMA of vy
        self._td_slow_ema = {}     # gname -> slow EMA of vy
        self._td_prev_delta = {}   # gname -> previous (fast - slow)

        # Velocity sub-signal state
        self._vel_prev_h = {}      # gname -> previous height
        self._vel_smooth_vy = {}   # gname -> EMA-smoothed vy

    def compute(self, pos, floor_height, dt, opts):
        """Compute per-group log-odds increment from unified vertical kinematics.

        Returns:
            Tuple[Dict[str, float], Dict[str, dict]]:
                logodds per group, context per group (phase + sub-signal diagnostics)
        """
        up = opts.up_axis
        heights = pos[:, up] - floor_height
        result = {}
        context = {}

        for gname, joints in get_active_groups(opts).items():
            valid = [j for j in joints if j < len(heights)]
            if not valid:
                result[gname] = 0.0
                context[gname] = {'phase': 0.0, 'traj_push': 0.0,
                                  'td_impulse': 0.0, 'vel_hold': 0.0}
                continue

            rep_j = min(valid, key=lambda j: heights[j])
            h = heights[rep_j]

            # --- Initialize on first frame ---
            if gname not in self._phase:
                self._phase[gname] = 0.0
                self._h_history[gname] = [h]
                self._traj_prev_h[gname] = h
                self._traj_prev_dh[gname] = 0.0
                self._td_prev_h[gname] = h
                self._td_fast_ema[gname] = 0.0
                self._td_slow_ema[gname] = 0.0
                self._td_prev_delta[gname] = 0.0
                self._vel_prev_h[gname] = h
                self._vel_smooth_vy[gname] = 0.0
                result[gname] = 0.0
                context[gname] = {'phase': 0.0, 'traj_push': 0.0,
                                  'td_impulse': 0.0, 'vel_hold': 0.0}
                continue

            # =============================================================
            # SUB-SIGNAL 1: Trajectory (epoch-based slope prediction)
            # =============================================================
            dh = h - self._traj_prev_h[gname]
            prev_dh = self._traj_prev_dh[gname]

            # Break detection: height velocity sign reversal
            h_break = self.HEIGHT_VEL_BREAK_MPS * dt
            is_break = False
            if (dh * prev_dh < 0
                    and (abs(dh) > h_break or abs(prev_dh) > h_break)):
                is_break = True
            if abs(dh) > h_break and abs(prev_dh) < h_break * 0.3:
                is_break = True

            if is_break:
                self._h_history[gname] = [h]
            else:
                self._h_history[gname].append(h)
                if len(self._h_history[gname]) > 20:
                    self._h_history[gname] = self._h_history[gname][-20:]

            self._traj_prev_h[gname] = h
            self._traj_prev_dh[gname] = dh

            # Compute trajectory push from slope
            eff_h = self._h_history[gname]
            nn = len(eff_h)
            traj_push = 0.0

            if nn >= 3:
                arr = np.array(eff_h)
                x = np.arange(nn, dtype=float)
                slope = float(np.polyfit(x, arr, 1)[0])
                mean_h = float(np.mean(arr))
                std_h = float(np.std(arr))

                # Confidence ramp
                if nn <= self.MIN_CONFIDENCE_FRAMES:
                    confidence = 0.0
                else:
                    confidence = min(1.0, (nn - self.MIN_CONFIDENCE_FRAMES) /
                                     (self.FULL_CONFIDENCE_FRAMES - self.MIN_CONFIDENCE_FRAMES))

                if (mean_h < opts.traj_height_thresh
                        and std_h < opts.traj_stable_std_thresh):
                    # Stable and low: very mild positive.
                    # Divergence now handles planted confirmation;
                    # kinematic just nudges to avoid fighting it.
                    traj_push = 0.01 * confidence
                elif slope < -opts.traj_trend_thresh:
                    # Falling: minimal approach arming — divergence and
                    # height will confirm; kinematic shouldn't preempt
                    # before deceleration actually occurs.
                    scale = min(1.0, -slope / (opts.traj_trend_thresh * 5))
                    traj_push = 0.01 * scale * confidence
                elif slope > opts.traj_trend_thresh:
                    # Rising: push toward airborne (departing)
                    scale = min(1.0, slope / (opts.traj_trend_thresh * 5))
                    traj_push = -0.05 * scale * confidence

            # =============================================================
            # SUB-SIGNAL 2: Touchdown (dual-EMA crossover impulse)
            # =============================================================
            raw_vy_td = (h - self._td_prev_h[gname]) / max(dt, 1e-6)
            self._td_prev_h[gname] = h

            alpha_fast = opts.td_alpha_fast
            alpha_slow = opts.td_alpha_slow
            fast = alpha_fast * raw_vy_td + (1.0 - alpha_fast) * self._td_fast_ema[gname]
            slow = alpha_slow * raw_vy_td + (1.0 - alpha_slow) * self._td_slow_ema[gname]
            self._td_fast_ema[gname] = fast
            self._td_slow_ema[gname] = slow

            delta = fast - slow
            prev_delta = self._td_prev_delta[gname]
            self._td_prev_delta[gname] = delta

            td_impulse = 0.0
            descent_speed = abs(slow)

            # Landing crossover: prev_delta < 0, delta >= 0, was genuinely falling
            if (prev_delta < 0 and delta >= 0
                    and descent_speed > opts.td_descent_gate
                    and h < opts.td_height_gate):
                strength = min(1.0, descent_speed / opts.td_descent_scale)
                td_impulse = 0.5 * strength  # strong push toward grounded

            # Liftoff crossover: fast drops below slow while near floor
            # and foot was in grounded phase
            elif (prev_delta >= 0 and delta < 0
                  and abs(slow) < 0.15  # was nearly stationary
                  and h < opts.td_height_gate
                  and self._phase.get(gname, 0.0) > 0.3):
                td_impulse = -0.3  # push toward airborne

            # =============================================================
            # SUB-SIGNAL 3: Velocity state (settled vs ascending)
            # =============================================================
            raw_vy_vel = (h - self._vel_prev_h[gname]) / max(dt, 1e-6)
            self._vel_prev_h[gname] = h

            prev_smooth = self._vel_smooth_vy.get(gname, 0.0)
            vy = 0.4 * raw_vy_vel + 0.6 * prev_smooth
            self._vel_smooth_vy[gname] = vy

            vel_hold = 0.0
            if vy > opts.vel_ascend_fast:
                # Strong ascending: push toward airborne
                vel_hold = -0.25
            elif vy > opts.vel_ascend_slow:
                # Moderate ascending: mild push toward airborne
                t = (vy - opts.vel_ascend_slow) / (opts.vel_ascend_fast - opts.vel_ascend_slow)
                vel_hold = -0.05 - 0.20 * t
            elif abs(vy) < opts.vel_ascend_slow and h < opts.vel_descend_decel_zone:
                # Near-zero vy + low: hold grounded (mild)
                vel_hold = 0.05

            # =============================================================
            # PHASE UPDATE
            # =============================================================
            phase = self._phase[gname]

            # Decay toward 0 (uncertain)
            phase *= opts.vk_phase_decay

            # Apply sub-signal pushes
            phase += traj_push
            phase += td_impulse
            phase += vel_hold

            # Clamp
            phase = max(-1.0, min(1.0, phase))
            self._phase[gname] = phase

            # =============================================================
            # PHASE -> LOG-ODDS MAPPING (with dead zone)
            # =============================================================
            dz = opts.vk_dead_zone
            if phase > dz:
                t = (phase - dz) / (1.0 - dz) if dz < 1.0 else 0.0
                lo = opts.vk_max_positive_logodds * t
            elif phase < -dz:
                t = (-phase - dz) / (1.0 - dz) if dz < 1.0 else 0.0
                lo = opts.vk_max_negative_logodds * t
            else:
                lo = 0.0

            result[gname] = lo
            context[gname] = {
                'phase': phase,
                'traj_push': traj_push,
                'td_impulse': td_impulse,
                'vel_hold': vel_hold,
                'vy': vy,
            }

        return result, context


class KinematicStream:
    """Unified kinematic evidence: approach angle + touchdown impulse + settled state.

    Integrates vertical and horizontal velocity into a single soft phase
    variable via three sub-signals:

      1. Phase-conditional approach angle:
         - Not planted: shallow angle = sweeping (airborne), steep = landing (grounded)
         - Planted: steep angle = liftoff (airborne), shallow = skating (ignored)
         Gated by velocity magnitude (low speed → no opinion).

      2. Touchdown impulse (dual-EMA crossover):
         Detects the exact moment of landing from vy waveform shape.

      3. Settled / ascending state:
         Low-speed settled detection and ascending velocity liftoff.

    Phase maps through a dead zone to log-odds output.
    """

    # Trajectory confidence ramp (frames)
    PLANTED_PHASE_THRESH = 0.3  # phase above this = "planted"

    def __init__(self):
        self._phase = {}           # gname → float [-1, 1]
        # Approach angle tracking
        self._prev_pos = {}        # gname → (3,) previous position of representative joint
        self._prev_rep_j = {}      # gname → int (which joint was rep last frame)
        self._prev_joint_pos = {}  # (gname, joint_idx) → (3,) per-joint position history
        # Touchdown impulse tracking
        self._td_prev_h = {}       # gname → float
        self._td_fast_ema = {}     # gname → float
        self._td_slow_ema = {}     # gname → float
        self._td_prev_delta = {}   # gname → float
        # Settled/ascending tracking
        self._vel_prev_h = {}      # gname → float
        self._vel_smooth_vy = {}   # gname → float
        # Approach zone epoch tracking (for never-settled detection)
        self._approach_min_h = {}  # gname → float (min height since entering approach zone)
        self._in_approach = {}     # gname → bool (currently in approach zone)
        self._was_planted = {}     # gname → bool (was planted last frame, for transition detection)
        self._settle_count = {}    # gname → int (consecutive frames with low |vy|)

    def compute(self, pos, floor_height, dt, opts):
        """Compute per-group log-odds from unified kinematic analysis.

        Args:
            pos: (J, 3) joint positions
            floor_height: float
            dt: float (seconds per frame)
            opts: LogOddsContactOptions

        Returns:
            logodds per group, context per group
        """
        up = opts.up_axis
        plane = [0, 2] if up == 1 else [0, 1]
        heights = pos[:, up] - floor_height
        result = {}
        context = {}

        for gname, joints in get_active_groups(opts).items():
            valid = [j for j in joints if j < len(heights)]
            if not valid:
                result[gname] = 0.0
                context[gname] = {'phase': 0.0, 'angle_push': 0.0,
                                  'td_impulse': 0.0, 'vel_hold': 0.0,
                                  'angle_deg': 0.0, 'speed': 0.0}
                continue

            # --- Select representative joint for this group ---
            # Default: lowest joint by height.
            # When multiple joints are near the floor (both in contact),
            # use the one with the lowest vertical velocity (most stationary).
            # This prevents heel peel from triggering group liftoff —
            # the stationary ball becomes the reference during toe-off.
            contact_thresh = 0.04  # both joints below this = both in contact
            near_floor = [j for j in valid if heights[j] < contact_thresh]

            if len(near_floor) >= 2:
                # Multiple joints near floor — pick by lowest vertical velocity
                def _joint_vy(j):
                    prev = self._prev_joint_pos.get((gname, j))
                    if prev is None:
                        return 0.0
                    return (pos[j, up] - prev[up]) / max(dt, 1e-6)
                rep_j = min(near_floor, key=_joint_vy)
            else:
                rep_j = min(valid, key=lambda j: heights[j])

            h = heights[rep_j]
            pos_3d = pos[rep_j]

            # Compute per-joint vertical velocities BEFORE overwriting prev positions
            joint_vys = {}
            for j in valid:
                prev_j = self._prev_joint_pos.get((gname, j))
                if prev_j is not None:
                    joint_vys[j] = (pos[j, up] - prev_j[up]) / max(dt, 1e-6)
                else:
                    joint_vys[j] = 0.0

            # Store per-joint positions for all valid joints
            for j in valid:
                self._prev_joint_pos[(gname, j)] = pos[j].copy()

            # --- Initialize on first frame ---
            if gname not in self._phase:
                self._phase[gname] = 0.0
                self._prev_pos[gname] = pos_3d.copy()
                self._prev_rep_j[gname] = rep_j
                self._td_prev_h[gname] = h
                self._td_fast_ema[gname] = 0.0
                self._td_slow_ema[gname] = 0.0
                self._td_prev_delta[gname] = 0.0
                self._vel_prev_h[gname] = h
                self._vel_smooth_vy[gname] = 0.0
                self._approach_min_h[gname] = h
                self._in_approach[gname] = h < opts.kin_settle_approach_zone
                result[gname] = 0.0
                context[gname] = {'phase': 0.0, 'angle_push': 0.0,
                                  'td_impulse': 0.0, 'vel_hold': 0.0,
                                  'never_settled': 0.0,
                                  'angle_deg': 0.0, 'speed': 0.0}
                continue

            # =============================================================
            # SUB-SIGNAL 1: Phase-conditional approach angle
            # =============================================================
            # Use the current rep joint's OWN previous position to avoid
            # velocity spikes when the representative switches (e.g.,
            # heel→ball during toe-off). If we don't have this joint's
            # previous position, use current pos (zero velocity).
            prev_rep = self._prev_rep_j.get(gname, rep_j)
            if rep_j == prev_rep:
                # Same representative — use stored group prev_pos
                prev_pos_3d = self._prev_pos[gname]
            else:
                # Representative switched — use the new joint's own
                # previous position if available, else zero velocity
                prev_pos_3d = self._prev_joint_pos.get(
                    (gname, rep_j), pos_3d)
            vel_3d = (pos_3d - prev_pos_3d) / max(dt, 1e-6)
            self._prev_pos[gname] = pos_3d.copy()
            self._prev_rep_j[gname] = rep_j

            # If representative switched, reset height trackers to avoid
            # false velocity spikes from comparing different joints' heights
            if rep_j != prev_rep:
                self._td_prev_h[gname] = h
                self._vel_prev_h[gname] = h

            vy = vel_3d[up]  # positive = ascending in world coords
            vh = float(np.sqrt(sum(vel_3d[d] ** 2 for d in plane)))
            speed = float(np.sqrt(vy ** 2 + vh ** 2))

            # Approach angle: atan2(|vy|, vh)  [0° = horizontal, 90° = vertical]
            if speed > 0.01:
                angle_deg = float(np.degrees(np.arctan2(abs(vy), vh)))
            else:
                angle_deg = 0.0

            # Confidence from speed magnitude
            conf_range = opts.kin_angle_speed_hi - opts.kin_angle_speed_lo
            if conf_range > 1e-6:
                angle_confidence = max(0.0, min(1.0,
                    (speed - opts.kin_angle_speed_lo) / conf_range))
            else:
                angle_confidence = 1.0 if speed > opts.kin_angle_speed_lo else 0.0

            # Phase-conditional interpretation
            phase = self._phase[gname]
            is_planted = phase > self.PLANTED_PHASE_THRESH
            angle_push = 0.0

            if angle_confidence > 0.05:
                if not is_planted:
                    # NOT PLANTED: approaching or airborne
                    if angle_deg < opts.kin_angle_shallow_deg:
                        # Shallow → sweeping past floor
                        angle_push = opts.kin_angle_sweep_push * angle_confidence
                    elif angle_deg > opts.kin_angle_steep_deg:
                        if vy < 0:
                            # Steep + descending → landing
                            angle_push = opts.kin_angle_land_push * angle_confidence
                        else:
                            # Steep + ascending → liftoff reinforcement
                            angle_push = opts.kin_angle_liftoff_push * angle_confidence
                else:
                    # PLANTED: contact established
                    if angle_deg > opts.kin_angle_steep_deg:
                        # Steep while planted → genuine liftoff
                        angle_push = opts.kin_angle_planted_lift_push * angle_confidence
                    # Shallow while planted → skating → no opinion (angle_push stays 0)

            # =============================================================
            # SUB-SIGNAL 2: Touchdown impulse (dual-EMA crossover)
            # =============================================================
            raw_vy_td = (h - self._td_prev_h[gname]) / max(dt, 1e-6)
            self._td_prev_h[gname] = h

            alpha_fast = opts.td_alpha_fast
            alpha_slow = opts.td_alpha_slow
            fast = alpha_fast * raw_vy_td + (1.0 - alpha_fast) * self._td_fast_ema[gname]
            slow = alpha_slow * raw_vy_td + (1.0 - alpha_slow) * self._td_slow_ema[gname]
            self._td_fast_ema[gname] = fast
            self._td_slow_ema[gname] = slow

            delta = fast - slow
            prev_delta = self._td_prev_delta[gname]
            self._td_prev_delta[gname] = delta

            td_impulse = 0.0
            td_landing_imp = 0.0
            td_apex_imp = 0.0
            td_liftoff_imp = 0.0
            descent_speed = abs(slow)

            # Landing crossover: prev_delta < 0, delta >= 0, was genuinely falling.
            # Single-frame impulse fired at the moment vy approaches 0 from
            # the negative side — the kinematic signature of an arrest.
            if (prev_delta < 0 and delta >= 0
                    and descent_speed > opts.td_descent_gate
                    and h < opts.td_height_gate):
                strength = min(1.0, descent_speed / opts.td_descent_scale)
                td_landing_imp = 0.5 * strength
                td_impulse = td_landing_imp

            # Apex crossover: prev_delta > 0, delta <= 0, was genuinely rising.
            # Symmetric counterpart of the landing crossover — vy is approaching
            # 0 from the positive side, which is the kinematic signature of a
            # swing apex, not a contact.  Only fires when slow_ema is positive
            # (so we know the recent motion was upward) and the apex is in the
            # decision-relevant height band.  Single-frame impulse, no spillover
            # into the surrounding descent / ascent phases.  Strength is
            # normalised against td_apex_scale (smaller than td_descent_scale)
            # because rise velocities are intrinsically smaller than fall
            # velocities — see comment on td_apex_scale in options.
            elif (prev_delta >= 0 and delta < 0
                  and slow > opts.td_apex_rise_gate
                  and h < opts.td_apex_height_gate):
                strength = min(1.0, slow / max(opts.td_apex_scale, 1e-6))
                td_apex_imp = opts.td_apex_logodds * strength
                td_impulse = td_apex_imp

            # Liftoff crossover: fast drops below slow while near floor in
            # planted state.  This is a separate event from the apex above:
            # the foot was already in contact (phase > PLANTED_PHASE_THRESH)
            # and is now lifting off with low speed.
            elif (prev_delta >= 0 and delta < 0
                  and abs(slow) < 0.15
                  and h < opts.td_height_gate
                  and phase > self.PLANTED_PHASE_THRESH):
                td_liftoff_imp = -0.3
                td_impulse = td_liftoff_imp

            # =============================================================
            # LEVER ROTATION CORRECTION
            # =============================================================
            # During toe-off, the ankle rotates (heel rises) while the ball
            # stays on the floor as a pivot. The ball joint in SMPL sits
            # above the actual metatarsal contact, so ankle rotation causes
            # apparent ball vy that is rotation artifact, not translation.
            # Detect lever rotation: if any other joint in the group rises
            # much faster than the rep, the rep's vy is partly illusory.
            lever_vy_correction = 0.0
            if len(valid) >= 2 and h < 0.05:
                rep_vy_raw = joint_vys.get(rep_j, 0.0)
                max_other_vy = max(
                    (joint_vys.get(j, 0.0) for j in valid if j != rep_j),
                    default=0.0
                )
                # Lever rotation: other joint rises faster than rep
                if max_other_vy > 0.05 and rep_vy_raw > 0:
                    # Any differential in rise rate is rotation evidence.
                    # correction fraction: 0 at ratio ≤1, 1 at ratio ≥2
                    ratio = max_other_vy / max(rep_vy_raw, 1e-6)
                    lever_frac = min(1.0, max(0.0, ratio - 1.0))
                    lever_vy_correction = rep_vy_raw * lever_frac

            # =============================================================
            # SUB-SIGNAL 3: Settled / ascending state
            # =============================================================
            raw_vy_vel = (h - self._vel_prev_h[gname]) / max(dt, 1e-6)
            self._vel_prev_h[gname] = h

            # Apply lever correction: subtract rotation-induced vy
            effective_vy = raw_vy_vel - lever_vy_correction

            prev_smooth = self._vel_smooth_vy.get(gname, 0.0)
            vy_smooth = 0.4 * effective_vy + 0.6 * prev_smooth
            self._vel_smooth_vy[gname] = vy_smooth

            vel_hold = 0.0
            if vy_smooth > opts.vel_ascend_fast:
                vel_hold = -0.25
            elif vy_smooth > opts.vel_ascend_slow:
                t = (vy_smooth - opts.vel_ascend_slow) / (opts.vel_ascend_fast - opts.vel_ascend_slow)
                vel_hold = -0.05 - 0.20 * t
            elif abs(vy_smooth) < opts.vel_ascend_slow and h < opts.vel_descend_decel_zone:
                vel_hold = 0.05

            # =============================================================
            # SUB-SIGNAL 4: Never-settled detection
            # =============================================================
            # Track minimum height since entering the approach zone (h < 8cm).
            # Uses velocity-based settling: consecutive frames of low |vy|.
            # Push ramps continuously: full at count=0, zero at count=settle_frames.
            never_settled_push = 0.0

            # Reset epoch on planted → not-planted transition (liftoff)
            was_planted = self._was_planted.get(gname, False)
            self._was_planted[gname] = is_planted
            if was_planted and not is_planted:
                # Foot just transitioned to "not planted" — reset approach tracking
                self._approach_min_h[gname] = h
                self._in_approach[gname] = h < opts.kin_settle_approach_zone

            if h < opts.kin_settle_approach_zone:
                # In approach zone — track minimum and settle count
                if self._in_approach.get(gname, False):
                    prev_min = self._approach_min_h.get(gname, h)
                    rise_from_min = h - prev_min
                    if rise_from_min > opts.kin_settle_rise_reset:
                        self._approach_min_h[gname] = h
                        self._settle_count[gname] = 0
                    else:
                        self._approach_min_h[gname] = min(prev_min, h)
                else:
                    self._approach_min_h[gname] = h
                    self._in_approach[gname] = True
                    self._settle_count[gname] = 0

                # Track consecutive low-|effective_vy| frames
                if abs(effective_vy) < opts.kin_settle_vy_thresh:
                    self._settle_count[gname] = self._settle_count.get(gname, 0) + 1
                else:
                    self._settle_count[gname] = 0

                # Ramped never-settled push: scales from full at count=0
                # to zero at count=settle_frames. No hard threshold.
                settle_count = self._settle_count.get(gname, 0)
                if (speed < opts.kin_settle_speed_gate
                        and not is_planted):
                    settle_progress = min(1.0, settle_count / max(1, opts.kin_settle_frames))
                    never_settled_push = opts.kin_settle_never_push * (1.0 - settle_progress)
            else:
                # Above approach zone — reset epoch
                self._in_approach[gname] = False
                self._approach_min_h[gname] = h
                self._settle_count[gname] = 0

            # =============================================================
            # PHASE UPDATE
            # =============================================================
            increment = angle_push + td_impulse + vel_hold + never_settled_push
            phase *= opts.kin_phase_decay
            phase += increment
            phase = max(-1.0, min(1.0, phase))
            self._phase[gname] = phase

            # =============================================================
            # PHASE -> LOG-ODDS MAPPING (with dead zone)
            # =============================================================
            dz = opts.kin_dead_zone
            if phase > dz:
                t = (phase - dz) / (1.0 - dz) if dz < 1.0 else 0.0
                lo = opts.kin_max_positive_logodds * t
            elif phase < -dz:
                t = (-phase - dz) / (1.0 - dz) if dz < 1.0 else 0.0
                lo = opts.kin_max_negative_logodds * t
            else:
                lo = 0.0

            result[gname] = lo
            context[gname] = {
                'phase': phase,
                'angle_push': angle_push,
                'td_impulse': td_impulse,
                'td_landing_imp': td_landing_imp,
                'td_apex_imp': td_apex_imp,
                'td_liftoff_imp': td_liftoff_imp,
                'vel_hold': vel_hold,
                'never_settled': never_settled_push,
                'angle_deg': angle_deg,
                'speed': speed,
                'vy': vy,
                'vh': vh,
                'angle_confidence': angle_confidence,
                'is_planted': float(is_planted),
                'approach_min_h': self._approach_min_h.get(gname, h),
                'vy_fast_ema': self._td_fast_ema.get(gname, 0.0),
                'vy_slow_ema': self._td_slow_ema.get(gname, 0.0),
            }

        return result, context


class DivergenceStream:
    """Evidence from foot-CoM relative velocity alignment.

    During walking, a planted foot falls behind the CoM (relative velocity
    opposes CoM direction), while a swinging foot races ahead (relative
    velocity aligns with CoM direction). This produces a clean, IMU-robust
    signal that discriminates planted from swinging feet without depending
    on world-frame positions.

    The signal is the horizontal dot product:
        alignment = dot(foot_vel - com_vel, com_vel) / |com_vel|

    Positive alignment → diverging (swing) → anti-contact
    Negative alignment → converging (planted) → pro-contact
    Near-zero com_speed → neutral (can't discriminate)

    When both feet are planted (e.g., weight shifting), both have foot_vel ≈ 0,
    so both get rel_vel ≈ -com_vel → both read as 'trailing' → both pro-contact.
    """

    def __init__(self):
        self._prev_pos = {}      # gname → (3,) representative joint position
        self._prev_com = None    # (3,) previous CoM
        self._alignment_ema = {}  # gname → EMA-smoothed alignment

    def compute(self, pos, com, com_vel, floor_height, dt, opts):
        """Compute per-group divergence evidence.

        Args:
            pos: (J, 3) joint positions
            com: (3,) center of mass
            com_vel: (3,) CoM velocity (may be None)
            floor_height: float
            dt: float
            opts: LogOddsContactOptions

        Returns:
            (Dict[gname, float], Dict[gname, dict]):
                log-odds increment per group, context per group
        """
        up = opts.up_axis
        horiz = [0, 2] if up == 1 else [0, 1]
        active = get_active_groups(opts)
        result = {g: 0.0 for g in active}
        context = {g: {} for g in active}

        if com is None:
            return result, context

        # Compute foot velocities from frame-to-frame position change
        for gname, joints_list in active.items():
            valid_j = [j for j in joints_list if j < len(pos)]
            if not valid_j:
                continue

            # Representative position: lowest joint (most likely contact)
            heights = [(pos[j, up] - floor_height, j) for j in valid_j]
            _, rep_j = min(heights)
            curr_pos = pos[rep_j].copy()

            prev = self._prev_pos.get(gname)
            self._prev_pos[gname] = curr_pos

            if prev is None or self._prev_com is None:
                continue

            # Foot velocity and CoM velocity in horizontal plane
            foot_vel_h = ((curr_pos - prev) / max(dt, 1e-6))[horiz]
            com_vel_h = ((com - self._prev_com) / max(dt, 1e-6))[horiz]
            com_speed = np.linalg.norm(com_vel_h)

            # Gate: need meaningful CoM translation to discriminate
            min_speed = opts.div_min_com_speed
            if com_speed < min_speed:
                context[gname] = {'div_align': 0.0, 'com_speed': com_speed}
                continue

            # Relative velocity: how the foot moves relative to CoM
            rel_vel = foot_vel_h - com_vel_h
            # Alignment: positive = diverging (swing), negative = trailing (planted)
            alignment_raw = float(np.dot(rel_vel, com_vel_h) / com_speed)

            # EMA smoothing on alignment.  Damps the swing→stance flip so the
            # anti-contact vote persists through the end of the swing, when
            # the foot stops racing ahead but isn't yet a contact candidate.
            # Initialise from the first raw value to avoid a startup transient.
            ema_alpha = max(0.0, min(1.0, opts.div_alignment_ema_alpha))
            prev_align = self._alignment_ema.get(gname, alignment_raw)
            alignment = ema_alpha * alignment_raw + (1.0 - ema_alpha) * prev_align
            self._alignment_ema[gname] = alignment

            # Map alignment to log-odds:
            #   alignment > 0 → swing → negative evidence (anti-contact)
            #   alignment < 0 → trailing → positive evidence (pro-contact)
            # Scale continuously: |alignment| / scale → strength
            scale = opts.div_alignment_scale
            strength = min(1.0, abs(alignment) / scale)

            # Asymmetric gating: anti-contact at any height (swing is swing),
            # but pro-contact only when near the floor. A foot decelerating
            # at h=2cm shouldn't pull itself into contact — that's the
            # height stream's job.
            min_h = min(h for h, _ in heights)  # lowest joint in group
            ground_gate = float(np.exp(-max(0.0, min_h) / opts.div_ground_gate_scale))

            if alignment > 0:
                lo = -strength * opts.div_max_logodds  # anti-contact: unrestricted
            else:
                lo = strength * opts.div_max_logodds * ground_gate  # pro-contact: gated

            result[gname] = lo
            context[gname] = {
                'div_align': alignment,
                'div_align_raw': alignment_raw,
                'com_speed': com_speed,
                'div_ground_gate': ground_gate,
            }

        self._prev_com = com.copy() if com is not None else None
        return result, context


class StructuralStream:
    """Phase A structural evidence via DynamicFrameEvaluator.

    Provides log-odds evidence only when physics yields unambiguous signals:
      - Freefall: strong negative for all groups
      - Negative force (tension): strong negative for that group
        (physically impossible for floor contact)
      - Single-group dominance: when only one foot group has candidates,
        it bears all weight → strong positive, others → negative
      - Ambiguous: when multiple groups have similar positive forces,
        the FE cannot discriminate → neutral (0.0)

    Phase B (pressure distribution) is handled separately after
    log-odds accumulation, using evaluate_and_refine on consensus contacts.
    """

    # Base contact joints (always active)
    BASE_CONTACT_JOINTS = {10, 11, 28, 29, 20, 21, 22, 23}

    # Base group mapping
    BASE_JOINT_TO_GROUP = {
        10: 'LF', 28: 'LF',
        11: 'RF', 29: 'RF',
        20: 'LH', 22: 'LH',
        21: 'RH', 23: 'RH',
    }

    @staticmethod
    def _get_contact_joints(opts):
        """Return contact joints and joint-to-group mapping for active groups."""
        joints = set(StructuralStream.BASE_CONTACT_JOINTS)
        j2g = dict(StructuralStream.BASE_JOINT_TO_GROUP)
        if getattr(opts, 'enable_body_contacts', False):
            for gname, joint_list in BODY_GROUPS.items():
                for j in joint_list:
                    joints.add(j)
                    j2g[j] = gname
        return joints, j2g

    def __init__(self, evaluator=None):
        """
        Args:
            evaluator: DynamicFrameEvaluator instance (or None, set later)
        """
        self.evaluator = evaluator
        self._log_path = None
        self._log_frame = 0
        self._smoothed_group_forces = {}
        # State for root-relative kinematic diagnostics
        self._prev_root_pos = None
        self._prev_foot_pos = {}   # joint_idx -> (3,)
        self._prev_d_LR = None     # signed L-R distance projected on root forward
        env_path = os.environ.get('SMPL_STRUCTURAL_LOG')
        if env_path:
            self.set_log_path(env_path)

    def set_evaluator(self, evaluator):
        """Set or replace the frame evaluator instance."""
        self.evaluator = evaluator

    def set_log_path(self, path):
        """Begin writing per-frame diagnostic JSONL to `path` (None to disable).

        Truncates the file. One JSON object per line, one line per frame.
        """
        self._log_frame = 0
        if path is None:
            self._log_path = None
            return
        try:
            open(path, 'w').close()
            self._log_path = path
        except OSError:
            self._log_path = None

    def _emit_log(self, payload):
        if not self._log_path:
            return
        try:
            with open(self._log_path, 'a') as f:
                f.write(json.dumps(payload) + '\n')
        except OSError:
            pass

    # SMPL joint indices used for root-relative kinematics
    _PELVIS, _SPINE, _L_HIP, _R_HIP, _L_FOOT, _R_FOOT = 0, 3, 1, 2, 10, 11

    def _root_relative_kinematics(self, pos, dt, up):
        """Compute root-relative diagnostic signals for swing/stance discrimination.

        Returns a dict suitable for merging into the log payload:
          - root_forward:    [fwd_x, fwd_z]   (horizontal forward unit, plane coords)
          - v_rel_root_LF:   float            ((v_LF − v_root) · fwd, m/s)
          - v_rel_root_RF:   float            ((v_RF − v_root) · fwd, m/s)
          - d_LR_signed:     float            ((pos_LF − pos_RF) · fwd, m)
          - v_LR_signed:     float            (m/s)

        Robust to whole-body global drift because both v_foot and v_root
        receive the same drift; the difference cancels it. Forward axis
        is derived from R_hip − L_hip cross spine − pelvis (body frame),
        not from CoM velocity, so it works at low speeds.

        State is updated for the next frame on every call.
        """
        out = {
            'root_forward': None,
            'v_rel_root_LF': None,
            'v_rel_root_RF': None,
            'd_LR_signed': None,
            'v_LR_signed': None,
        }
        J = pos.shape[0]
        needed = max(self._PELVIS, self._SPINE, self._L_HIP, self._R_HIP,
                     self._L_FOOT, self._R_FOOT)
        if needed >= J:
            return out

        # Body-frame forward = side × up_body, projected to horizontal.
        side = pos[self._R_HIP] - pos[self._L_HIP]
        up_body = pos[self._SPINE] - pos[self._PELVIS]
        fwd = np.cross(side, up_body)
        fwd[up] = 0.0
        n = float(np.linalg.norm(fwd))
        if n < 1e-6:
            return out
        fwd_unit = fwd / n
        plane = [0, 2] if up == 1 else [0, 1]
        out['root_forward'] = [float(fwd_unit[plane[0]]), float(fwd_unit[plane[1]])]

        # Signed L−R distance is independent of velocity → can compute now.
        d_LR = float(np.dot(pos[self._L_FOOT] - pos[self._R_FOOT], fwd_unit))
        out['d_LR_signed'] = d_LR

        # Velocities require a previous frame.
        if self._prev_root_pos is None or dt <= 0:
            self._prev_root_pos = pos[self._PELVIS].copy()
            self._prev_foot_pos[self._L_FOOT] = pos[self._L_FOOT].copy()
            self._prev_foot_pos[self._R_FOOT] = pos[self._R_FOOT].copy()
            self._prev_d_LR = d_LR
            return out

        v_root = (pos[self._PELVIS] - self._prev_root_pos) / dt
        v_LF = (pos[self._L_FOOT] - self._prev_foot_pos[self._L_FOOT]) / dt
        v_RF = (pos[self._R_FOOT] - self._prev_foot_pos[self._R_FOOT]) / dt

        out['v_rel_root_LF'] = float(np.dot(v_LF - v_root, fwd_unit))
        out['v_rel_root_RF'] = float(np.dot(v_RF - v_root, fwd_unit))
        if self._prev_d_LR is not None:
            out['v_LR_signed'] = (d_LR - self._prev_d_LR) / dt

        self._prev_root_pos = pos[self._PELVIS].copy()
        self._prev_foot_pos[self._L_FOOT] = pos[self._L_FOOT].copy()
        self._prev_foot_pos[self._R_FOOT] = pos[self._R_FOOT].copy()
        self._prev_d_LR = d_LR
        return out

    def compute(self, pos, com, com_acc, com_vel, floor_height, dt, opts,
                raw_com_acc=None, intensities=None, gravity_torque_vecs=None):
        """Phase A: Compute per-group log-odds from structural necessity.

        Provides evidence only where physics is unambiguous.
        Uses inverted pendulum dynamics: when CoM is falling, the trailing
        foot is the support pivot; the leading foot is not yet catching.
        Returns neutral (0.0) when it can't discriminate.

        Args:
            pos: (J, 3) joint positions
            com: (3,) center of mass position
            com_acc: (3,) CoM acceleration (filtered)
            floor_height: float
            dt: float
            opts: LogOddsContactOptions
            raw_com_acc: (3,) optional raw (unfiltered) CoM acceleration

        Returns:
            Dict[group_name, float]: log-odds increment per group
        """
        up = opts.up_axis
        plane_dims = [0, 2] if up == 1 else [0, 1]
        g_mag = opts.struct_gravity
        active = get_active_groups(opts)
        result = {g: 0.0 for g in active}

        # --- Root-relative kinematics ---
        # Updates internal state every frame so velocity is correct even
        # across freefall / no-evaluator / no-candidate branches that
        # short-circuit later. Cheap; safe to run unconditionally.
        rrk = self._root_relative_kinematics(pos, dt, up)

        # --- Diagnostic snapshot (only built if logging enabled) ---
        if self._log_path:
            log = {
                'frame': self._log_frame,
                'com_hz': [float(com[plane_dims[0]]), float(com[plane_dims[1]])],
                'com_h': float(com[up] - floor_height),
                'a_hz_filt': [float(com_acc[plane_dims[0]]),
                              float(com_acc[plane_dims[1]])],
                'a_vert_filt': float(com_acc[up]),
                'a_hz_raw': ([float(raw_com_acc[plane_dims[0]]),
                              float(raw_com_acc[plane_dims[1]])]
                             if raw_com_acc is not None else None),
                'a_vert_raw': (float(raw_com_acc[up])
                               if raw_com_acc is not None else None),
                'floor_h': float(floor_height),
                'dt': float(dt),
                'branch': None,
                'candidates': [],
                'group_reps': {},
                'rep_pos_hz': {},
                'group_forces': {},
                'zmp_approx': None,
                'zmp_displacement': None,
                'support_frac_filt': None,
                'support_frac_raw': None,
            }
            log.update(rrk)
        else:
            log = None

        # --- Freefall detection (raw acceleration if available) ---
        acc_for_freefall = raw_com_acc if raw_com_acc is not None else com_acc
        if acc_for_freefall is not None:
            f_up = acc_for_freefall[up] + g_mag
            support_frac = max(0.0, f_up / g_mag)
            if log is not None:
                log['support_frac_raw'] = float(support_frac)
                log['support_frac_filt'] = float(
                    max(0.0, (com_acc[up] + g_mag) / g_mag))
            if support_frac < opts.struct_freefall_thresh:
                for gname in active:
                    result[gname] = opts.struct_freefall_logodds
                if log is not None:
                    log['branch'] = 'freefall'
                    log['result'] = dict(result)
                    self._emit_log(log)
                    self._log_frame += 1
                return result

            # --- Push-off evidence (continuous above body weight) ---
            # Excess support fraction above 1.0 is direct evidence of
            # ground contact: accelerating the body requires MORE than
            # body weight in GRF. Below 1.0, the FE's inverted pendulum
            # handles per-foot discrimination — blanket evidence here
            # would override that directional signal.
            excess = support_frac - 1.0
            if excess > 0.01:
                pushoff_strength = min(1.0, excess)  # 0→1 over sup 1.0→2.0
                pushoff_logodds = pushoff_strength * opts.struct_pushoff_logodds
                heights_local = pos[:, up] - floor_height
                for gname, joints_list in active.items():
                    valid_j = [j for j in joints_list if j < len(heights_local)]
                    if valid_j:
                        min_h = max(0.0, min(heights_local[j] for j in valid_j))
                        # Ground affinity: exponential decay with height
                        ground_affinity = float(np.exp(-min_h / 0.04))
                        if ground_affinity > 0.01:
                            result[gname] = pushoff_logodds * ground_affinity
                        elif min_h > opts.struct_candidate_height:
                            result[gname] = opts.struct_unnecessary_logodds

        # --- If no evaluator available, return neutral ---
        if self.evaluator is None:
            if log is not None:
                log['branch'] = 'no_evaluator'
                log['result'] = dict(result)
                self._emit_log(log)
                self._log_frame += 1
            return result

        contact_joints, joint_to_group = self._get_contact_joints(opts)

        # --- Find candidates by height ---
        heights = pos[:, up] - floor_height
        candidates = set()
        # Wider candidacy threshold for body contacts: complex poses
        # have larger mocap error, so joints may be 20cm up yet touching.
        cand_h = 0.25 if getattr(opts, 'enable_body_contacts', False) else opts.struct_candidate_height
        for j in contact_joints:
            if j < len(heights) and heights[j] < cand_h:
                candidates.add(j)

        if log is not None:
            log['candidates'] = sorted(int(j) for j in candidates)

        if not candidates:
            # No joints near floor → mild negative for foot groups
            for gname in FOOT_GROUPS:
                result[gname] = opts.struct_unnecessary_logodds
            if log is not None:
                log['branch'] = 'no_candidates'
                log['result'] = dict(result)
                self._emit_log(log)
                self._log_frame += 1
            return result

        # --- Identify which groups have candidates ---
        groups_with_candidates = {}
        for j in candidates:
            gname = joint_to_group.get(j)
            if gname:
                if gname not in groups_with_candidates:
                    groups_with_candidates[gname] = []
                groups_with_candidates[gname].append(j)

        # --- Select one representative per group for FE evaluation ---
        # Using individual joints causes spread regularization to kill
        # the inter-group lever arm (within-foot pairs are ~15cm apart
        # with 8cm spread each → effective lever = 0 → 50/50 blend).
        # Use the lowest joint in each group as representative.
        group_reps = {}  # gname → representative joint index
        rep_candidates = set()
        for gname, joints in groups_with_candidates.items():
            rep = min(joints, key=lambda j: heights[j])
            group_reps[gname] = rep
            rep_candidates.add(rep)

        # --- Evaluate structural frame with group representatives ---
        eval_result = self.evaluator.evaluate(
            rep_candidates, pos, com, com_acc, floor_height, up,
            gravity_torque_vecs=gravity_torque_vecs,
            relief_enable=getattr(opts, 'fe_relief_enable', False),
            relief_strain_threshold=getattr(opts, 'fe_relief_strain_threshold', 25.0))
        # Expose for downstream (frame_eval_zmp output)
        self.last_eval_result = eval_result

        # --- Extract per-group force from representative ---
        group_forces_raw = {}
        for gname in groups_with_candidates:
            rep = group_reps[gname]
            group_forces_raw[gname] = eval_result.per_contact_force.get(rep, 0.0)

        # --- Temporal EMA on per-group forces ---
        # Per-frame ZMP noise can flip the lever-rule split many times per
        # stance. EMA smooths this without changing the FE itself. alpha=1
        # disables smoothing (raw passthrough); smaller alpha = heavier.
        alpha = max(0.0, min(1.0, opts.struct_force_ema_alpha))
        if alpha >= 0.999:
            group_forces = group_forces_raw
        else:
            group_forces = {}
            for gname in groups_with_candidates:
                raw = group_forces_raw[gname]
                prev = self._smoothed_group_forces.get(gname, raw)
                group_forces[gname] = alpha * raw + (1.0 - alpha) * prev
            # Decay groups that lost candidacy this frame so they don't
            # linger forever — pull them toward 0 at the same rate.
            for gname in list(self._smoothed_group_forces.keys()):
                if gname not in groups_with_candidates:
                    self._smoothed_group_forces[gname] = (
                        (1.0 - alpha) * self._smoothed_group_forces[gname]
                    )
            # Update state with the just-computed smoothed values
            for gname, val in group_forces.items():
                self._smoothed_group_forces[gname] = val

        if log is not None:
            log['group_reps'] = {g: int(r) for g, r in group_reps.items()}
            log['rep_pos_hz'] = {
                g: [float(pos[r, plane_dims[0]]),
                    float(pos[r, plane_dims[1]])]
                for g, r in group_reps.items()
            }
            log['group_forces_raw'] = {g: float(f) for g, f in group_forces_raw.items()}
            log['group_forces'] = {g: float(f) for g, f in group_forces.items()}
            zmp = eval_result.zmp_approx
            log['zmp_approx'] = [float(zmp[0]), float(zmp[1])]
            log['zmp_displacement'] = [
                float(com[plane_dims[0]] - zmp[0]),
                float(com[plane_dims[1]] - zmp[1]),
            ]
            log['force_ema_alpha'] = float(alpha)

        # --- Signal 1: Negative force (tension) detection ---
        # If the FE says a contact needs to PULL, that's physically impossible
        for gname, force in group_forces.items():
            if force < -0.5:
                result[gname] = opts.struct_pulling_logodds

        # --- Signal 2: Two-pass force-based necessity ---
        # Pass 1: Check if established contacts (already in contact)
        # provide sufficient support. Only recruit new candidates when
        # there's a genuine support deficit.
        all_candidate_groups = list(groups_with_candidates.keys())
        total_force = sum(group_forces.get(g, 0.0) for g in all_candidate_groups)

        # Classify candidates as established vs new
        ESTABLISHED_THRESH = 0.3
        established = set()
        new_candidates = set()
        for gname in all_candidate_groups:
            if intensities and intensities.get(gname, 0.0) > ESTABLISHED_THRESH:
                established.add(gname)
            else:
                new_candidates.add(gname)

        # Determine if established contacts have a support deficit.
        # Run FE with only established contacts' reps.
        has_deficit = True  # default: recruit freely
        if (established and new_candidates
                and self.evaluator is not None
                and getattr(opts, 'enable_body_contacts', False)):
            est_reps = set()
            for gname in established:
                if gname in group_reps:
                    est_reps.add(group_reps[gname])
            if est_reps:
                est_result = self.evaluator.evaluate(
                    est_reps, pos, com, com_acc, floor_height, up)
                # Check residual: if established contacts can balance the
                # body (residual small), no deficit — don't recruit.
                residual_mag = float(np.linalg.norm(est_result.residual))
                # Total established force
                est_total = sum(est_result.per_contact_force.get(j, 0.0)
                                for j in est_reps)
                # Deficit = significant residual (body tips laterally) OR
                # tension (negative force → body tips over them) OR
                # insufficient vertical support.
                has_tension = any(est_result.per_contact_force.get(j, 0.0) < -1.0
                                  for j in est_reps)
                has_deficit = (has_tension
                               or est_total < self.evaluator.total_mass * 0.5
                               or residual_mag > 5.0)

        if len(all_candidate_groups) == 1:
            sole_group = all_candidate_groups[0]
            sole_force = group_forces.get(sole_group, 0.0)
            if sole_force > opts.struct_force_mild:
                result[sole_group] = opts.struct_necessary_logodds
        elif len(all_candidate_groups) >= 2 and total_force > 1.0:
            # Adaptive thresholds: with N groups, equal share = 1/N.
            body_contacts_on = getattr(opts, 'enable_body_contacts', False)
            n_groups = len(all_candidate_groups)
            equal_share = 1.0 / n_groups
            for gname in all_candidate_groups:
                force = group_forces.get(gname, 0.0)
                frac = force / total_force

                # Determine structural evidence for this group
                if frac < equal_share * 0.3:
                    result[gname] = opts.struct_unnecessary_logodds
                elif frac < equal_share * 0.6:
                    result[gname] = opts.struct_pendulum_leading_logodds
                elif frac >= equal_share * 0.8:
                    if body_contacts_on and (gname in established or has_deficit):
                        # Body-contact mode: established or deficit → full.
                        result[gname] = opts.struct_necessary_logodds
                    elif gname in established and gname in FOOT_GROUPS:
                        # Foot-only mode: an ESTABLISHED foot bearing ≥80%
                        # of weight is unambiguously supporting the body.
                        # Provide full structural evidence to counterbalance
                        # kinematic false-liftoff from postural sway.
                        # New candidates stay neutral (0.0) to avoid
                        # premature recruitment during walking swing.
                        result[gname] = opts.struct_necessary_logodds
                    else:
                        # Walking or new candidate: stay neutral
                        result[gname] = 0.0
                else:
                    if body_contacts_on and (gname in established or has_deficit):
                        result[gname] = opts.struct_mild_logodds
                    elif gname in established and gname in FOOT_GROUPS:
                        # Moderate share + established foot → mild evidence
                        result[gname] = opts.struct_mild_logodds
                    else:
                        result[gname] = 0.0

        # --- Groups not in candidates get mild negative ---
        # Only foot groups get structural unnecessary penalty: hands and body
        # contacts rely on the height stream for negative evidence (their
        # structural necessity is harder to infer when the body is low).
        for gname in FOOT_GROUPS:
            if gname not in groups_with_candidates and result.get(gname, 0.0) == 0.0:
                result[gname] = opts.struct_unnecessary_logodds

        # --- Effort-relief override for hand groups ---
        # When the FE flagged a hand candidate as bearing trunk strain
        # (high gravity-torque on a spine joint, lever arm on lean side),
        # raise the hand group's structural log-odds to a small positive.
        # Floor — does not reduce a positive value already emitted.
        relief_promoted = getattr(eval_result, 'effort_relief_promoted', None)
        relief_logodds = getattr(opts, 'struct_relief_logodds', 0.3)
        if relief_promoted and relief_logodds > 0:
            for gname in HAND_GROUPS:
                rep = group_reps.get(gname)
                if rep is not None and rep in relief_promoted:
                    if result.get(gname, 0.0) < relief_logodds:
                        result[gname] = relief_logodds

        if log is not None:
            if len(all_candidate_groups) == 1:
                log['branch'] = 'single_group'
            elif len(all_candidate_groups) >= 2:
                log['branch'] = 'multi_group'
            else:
                log['branch'] = 'no_candidates_with_force'
            log['candidate_groups'] = all_candidate_groups
            log['established'] = sorted(established)
            log['new_candidates'] = sorted(new_candidates)
            log['has_deficit'] = has_deficit
            if relief_promoted:
                log['effort_relief_promoted'] = sorted(int(j) for j in relief_promoted)
            log['result'] = dict(result)
            self._emit_log(log)
            self._log_frame += 1

        return result


class VelocityStream:
    """Vertical velocity + deceleration evidence.

    Three regimes:
      Ascending fast:  strong negative (liftoff)
      Ascending slow:  moderate negative
      Stationary+low:  weak positive (planted)
      Descending then decelerating near floor: strong positive (landing)
      Descending far from floor: neutral (approaching but not yet contact)

    Key insight: descent alone is possibility of future contact, not evidence
    of current contact. Deceleration after descent near the floor is the
    actual contact signal.
    """

    def __init__(self):
        self._prev_vy = {}     # gname → previous frame's smoothed vy
        self._prev_h = {}      # gname → previous frame's height
        self._smooth_vy = {}   # gname → EMA-smoothed vy

    def compute(self, pos, floor_height, dt, opts):
        """Compute per-group log-odds increment from velocity.

        Returns:
            Tuple[Dict[str, float], Dict[str, dict]]:
                logodds per group, context per group
        """
        up = opts.up_axis
        heights = pos[:, up] - floor_height
        result = {}
        context = {}  # per-group context for valving/diagnostics

        for gname, joints in get_active_groups(opts).items():
            valid = [j for j in joints if j < len(heights)]
            if not valid:
                result[gname] = 0.0
                context[gname] = {'vy': 0.0, 'decel': 0.0, 'height_m': 999.0}
                continue

            # Representative joint (lowest)
            rep_j = min(valid, key=lambda j: heights[j])
            h = heights[rep_j]

            # Compute raw velocity from position delta
            prev_h = self._prev_h.get(gname)
            if prev_h is None:
                self._prev_h[gname] = h
                self._prev_vy[gname] = 0.0
                self._smooth_vy[gname] = 0.0
                result[gname] = 0.0
                context[gname] = {'vy': 0.0, 'decel': 0.0, 'height_m': h}
                continue

            raw_vy = (h - prev_h) / max(dt, 1e-6)  # positive = ascending
            self._prev_h[gname] = h

            # EMA-smooth velocity (alpha=0.4 -> ~3 frame effective window)
            prev_smooth = self._smooth_vy.get(gname, 0.0)
            vy = 0.4 * raw_vy + 0.6 * prev_smooth
            prev_vy = self._prev_vy.get(gname, 0.0)
            self._smooth_vy[gname] = vy
            self._prev_vy[gname] = vy

            # Vertical acceleration (for deceleration detection)
            vy_accel = (vy - prev_vy) / max(dt, 1e-6)
            # Deceleration factor: positive when descending limb is slowing
            # (prev_vy negative, vy less negative => accel positive)
            if prev_vy < -0.05 and vy_accel > 0:
                decel_factor = min(1.0, vy_accel / 5.0)  # normalize to [0, 1]
            else:
                decel_factor = 0.0

            # Compute horizontal speed for stillness check
            plane = [0, 2] if up == 1 else [0, 1]
            pos_hz = pos[rep_j, plane]
            prev_pos_hz = getattr(self, '_prev_pos_hz', {}).get(gname)
            if prev_pos_hz is not None:
                hspeed = float(np.linalg.norm(pos_hz - prev_pos_hz) / max(dt, 1e-6))
            else:
                hspeed = 0.0
            if not hasattr(self, '_prev_pos_hz'):
                self._prev_pos_hz = {}
            self._prev_pos_hz[gname] = pos_hz.copy()

            # --- Classify regime ---
            lo = 0.0

            if vy > opts.vel_ascend_fast:
                lo = opts.vel_ascend_fast_logodds
            elif vy > opts.vel_ascend_slow:
                t = (vy - opts.vel_ascend_slow) / (
                    opts.vel_ascend_fast - opts.vel_ascend_slow)
                lo = (opts.vel_ascend_slow_logodds * (1.0 - t)
                      + opts.vel_ascend_fast_logodds * t)
            elif vy > -opts.vel_ascend_slow:
                if h < opts.vel_descend_decel_zone and hspeed < 0.1:
                    lo = opts.vel_stationary_logodds
                else:
                    lo = 0.0
            else:
                if h < opts.vel_descend_decel_zone:
                    accel = (vy - prev_vy) / max(dt, 1e-6)
                    if accel > 0:
                        decel_strength = min(1.0, accel / 2.0)
                        lo = opts.vel_decel_logodds * decel_strength
                    else:
                        lo = 0.2
                else:
                    lo = 0.0

            result[gname] = lo
            context[gname] = {
                'vy': vy,
                'decel': decel_factor,
                'height_m': h,
            }

        return result, context


class TrajectoryStream:
    """Adaptive-window trajectory analysis with persistent epoch tracking.

    The window grows forward from the most recent 'regime break.'
    Once a break is detected, old data is permanently discarded.

    A break is detected when:
      - Height velocity changes sign significantly
      - Height suddenly starts changing after being stable
      - Horizontal movement suddenly starts or stops

    Between breaks, the window grows, building confidence.
    """

    # Thresholds for break detection (in m/s, converted per-frame using dt)
    HEIGHT_VEL_BREAK_MPS = 0.25   # m/s: height velocity that qualifies as a regime change
    HZ_VEL_BREAK_MPS = 0.40       # m/s: horizontal velocity that qualifies

    # Dual-EMA for horizontal deceleration detection
    HZ_EMA_FAST_ALPHA = 0.5
    HZ_EMA_SLOW_ALPHA = 0.15

    def __init__(self):
        self._h_history = {}       # gname -> list of heights (post-epoch)
        self._hz_history = {}      # gname -> list of hz positions (post-epoch)
        self._prev_h = {}          # gname -> previous frame height
        self._prev_hz = {}         # gname -> previous frame hz position
        self._prev_dh = {}         # gname -> previous height velocity
        self._hz_ema_fast = {}     # gname -> fast EMA of horizontal speed
        self._hz_ema_slow = {}     # gname -> slow EMA of horizontal speed
        self._hz_prev_delta = {}   # gname -> previous (fast - slow) for crossover

    def compute(self, pos, floor_height, dt, opts):
        """Compute per-group log-odds increment from trajectory."""
        up = opts.up_axis
        plane = [0, 2] if up == 1 else [0, 1]
        heights = pos[:, up] - floor_height
        result = {}

        # Convert m/s thresholds to per-frame
        h_break = self.HEIGHT_VEL_BREAK_MPS * dt
        hz_break = self.HZ_VEL_BREAK_MPS * dt

        for gname, joints in get_active_groups(opts).items():
            valid = [j for j in joints if j < len(heights)]
            if not valid:
                result[gname] = 0.0
                continue

            rep_j = min(valid, key=lambda j: heights[j])
            h = heights[rep_j]
            pos_hz = pos[rep_j, plane].copy()

            # Initialize on first frame
            if gname not in self._prev_h:
                self._prev_h[gname] = h
                self._prev_hz[gname] = pos_hz.copy()
                self._prev_dh[gname] = 0.0
                self._h_history[gname] = [h]
                self._hz_history[gname] = [pos_hz]
                self._hz_ema_fast[gname] = 0.0
                self._hz_ema_slow[gname] = 0.0
                self._hz_prev_delta[gname] = 0.0
                result[gname] = 0.0
                continue

            # Current frame deltas
            dh = h - self._prev_h[gname]
            dhz = float(np.linalg.norm(pos_hz - self._prev_hz[gname]))
            prev_dh = self._prev_dh[gname]

            # --- Dual-EMA crossover on horizontal speed ---
            hspeed_inst = dhz / max(dt, 1e-6)
            af = self.HZ_EMA_FAST_ALPHA
            als = self.HZ_EMA_SLOW_ALPHA
            self._hz_ema_fast[gname] = af * hspeed_inst + (1 - af) * self._hz_ema_fast[gname]
            self._hz_ema_slow[gname] = als * hspeed_inst + (1 - als) * self._hz_ema_slow[gname]
            delta = self._hz_ema_fast[gname] - self._hz_ema_slow[gname]
            prev_delta = self._hz_prev_delta[gname]
            # Crossover: fast just dropped below slow while speed was significant
            # Only trigger on high-speed events (jumps, fast transitions)
            # not normal walking approach (foot at ~0.3-0.5 m/s)
            hz_decel_cross = (delta < 0 and prev_delta >= 0
                              and self._hz_ema_slow[gname] > 0.8)
            self._hz_prev_delta[gname] = delta

            # --- Break detection on THIS frame ---
            is_break = False

            # Break 1: height velocity sign reversal (significant)
            if (dh * prev_dh < 0
                    and (abs(dh) > h_break
                         or abs(prev_dh) > h_break)):
                is_break = True

            # Break 2: height starts moving after being stable
            if (abs(dh) > h_break
                    and abs(prev_dh) < h_break * 0.3):
                is_break = True

            # Break 3: horizontal movement starts after being still
            if len(self._hz_history[gname]) >= 2:
                prev_dhz = float(np.linalg.norm(
                    self._hz_history[gname][-1] - self._hz_history[gname][-2]))
                if (dhz > hz_break
                        and prev_dhz < hz_break * 0.3):
                    is_break = True

            # Break 4: horizontal deceleration crossover
            if hz_decel_cross:
                is_break = True

            # If break detected, reset epoch
            if is_break:
                self._h_history[gname] = [h]
                self._hz_history[gname] = [pos_hz]
            else:
                self._h_history[gname].append(h)
                self._hz_history[gname].append(pos_hz)
                if len(self._h_history[gname]) > 20:
                    self._h_history[gname] = self._h_history[gname][-20:]
                    self._hz_history[gname] = self._hz_history[gname][-20:]

            # Update state for next frame
            self._prev_h[gname] = h
            self._prev_hz[gname] = pos_hz.copy()
            self._prev_dh[gname] = dh

            # --- Compute trajectory evidence from post-epoch data ---
            eff_h = self._h_history[gname]
            eff_hz = self._hz_history[gname]

            if len(eff_h) < 2:
                # Just had a break: output neutral
                result[gname] = 0.0
                continue

            arr = np.array(eff_h)
            mean_h = float(np.mean(arr))
            std_h = float(np.std(arr))
            nn = len(arr)
            x = np.arange(nn, dtype=float)
            slope = float(np.polyfit(x, arr, 1)[0]) if nn >= 2 else 0.0

            hz_arr = np.array(eff_hz)
            hz_displacement = float(np.linalg.norm(hz_arr[-1] - hz_arr[0]))

            lo = 0.0

            if hz_displacement > opts.traj_hz_travel_thresh:
                scale = min(1.0, hz_displacement / (opts.traj_hz_travel_thresh * 4))
                lo = opts.traj_hz_travel_logodds * scale
            elif (mean_h < opts.traj_height_thresh
                  and std_h < opts.traj_stable_std_thresh
                  and nn >= 3):
                lo = opts.traj_stable_low_logodds
            elif slope > opts.traj_trend_thresh:
                scale = min(1.0, slope / (opts.traj_trend_thresh * 5))
                lo = opts.traj_rising_logodds * scale
            elif slope < -opts.traj_trend_thresh:
                scale = min(1.0, -slope / (opts.traj_trend_thresh * 5))
                lo = opts.traj_falling_logodds * scale

            # Confidence ramp: don't trust evidence from a freshly-broken epoch.
            # At swing nadir, a break fires and the window has only 2-3 frames
            # before the foot rises again — not enough to be meaningful.
            # Ramp confidence from 0 at 2 frames to 1.0 at 7 frames.
            MIN_CONFIDENCE_FRAMES = 2
            FULL_CONFIDENCE_FRAMES = 7
            if nn <= MIN_CONFIDENCE_FRAMES:
                confidence = 0.0
            else:
                confidence = min(1.0, (nn - MIN_CONFIDENCE_FRAMES) /
                                 (FULL_CONFIDENCE_FRAMES - MIN_CONFIDENCE_FRAMES))
            lo *= confidence

            result[gname] = lo

        return result


class HorizontalSpeedStream:
    """Horizontal velocity magnitude evidence with dual-EMA deceleration detection.

    A swinging leg has distinctly faster horizontal movement than a planted one.
    - Fast horizontal movement → negative (swinging)
    - Slow/stationary when low → weak positive (planted)
    - Slow/stationary when high → neutral

    Refinements:
    - Path straightness ratio distinguishes genuine swing from jitter.
    - Dual-EMA crossover detects deceleration: when the fast EMA drops
      below the slow EMA, negative logodds are attenuated.
    """

    PATH_WINDOW = 8  # frames of horizontal position history

    # Dual-EMA parameters for deceleration detection
    DECEL_FAST_ALPHA = 0.5
    DECEL_SLOW_ALPHA = 0.15

    def __init__(self):
        self._prev_pos_hz = {}   # gname → previous horizontal position
        self._pos_history = {}   # gname → deque of recent hz positions
        self._ema_fast = {}      # gname → fast EMA of horizontal speed
        self._ema_slow = {}      # gname → slow EMA of horizontal speed

    def compute(self, pos, floor_height, dt, opts):
        """Compute per-group log-odds increment from horizontal speed.

        Returns:
            Tuple[Dict[str, float], Dict[str, dict]]:
                logodds per group, context per group
        """
        up = opts.up_axis
        plane = [0, 2] if up == 1 else [0, 1]
        heights = pos[:, up] - floor_height
        result = {}
        context = {}

        for gname, joints in get_active_groups(opts).items():
            valid = [j for j in joints if j < len(heights)]
            if not valid:
                result[gname] = 0.0
                context[gname] = {'hspeed': 0.0, 'straightness': 0.0}
                continue

            rep_j = min(valid, key=lambda j: heights[j])
            h = heights[rep_j]
            pos_hz = pos[rep_j, plane].copy()

            prev = self._prev_pos_hz.get(gname)
            if prev is None:
                self._prev_pos_hz[gname] = pos_hz
                self._pos_history[gname] = deque(maxlen=self.PATH_WINDOW)
                self._pos_history[gname].append(pos_hz)
                self._ema_fast[gname] = 0.0
                self._ema_slow[gname] = 0.0
                result[gname] = 0.0
                context[gname] = {'hspeed': 0.0, 'straightness': 0.0, 'decel': 0.0}
                continue

            hspeed = float(np.linalg.norm(pos_hz - prev) / max(dt, 1e-6))
            self._prev_pos_hz[gname] = pos_hz
            self._pos_history[gname].append(pos_hz)

            # --- Dual-EMA deceleration detection ---
            af = self.DECEL_FAST_ALPHA
            als = self.DECEL_SLOW_ALPHA
            self._ema_fast[gname] = af * hspeed + (1 - af) * self._ema_fast[gname]
            self._ema_slow[gname] = als * hspeed + (1 - als) * self._ema_slow[gname]

            # Deceleration strength: how much fast is below slow (normalized)
            slow = self._ema_slow[gname]
            fast = self._ema_fast[gname]
            if slow > 0.05:
                decel_ratio = max(0.0, (slow - fast) / slow)  # 0=no decel, 1=strong decel
            else:
                decel_ratio = 0.0

            # Compute path straightness ratio over recent window
            hist = self._pos_history[gname]
            if len(hist) >= 3:
                pts = list(hist)
                path_length = sum(
                    np.linalg.norm(np.array(pts[i+1]) - np.array(pts[i]))
                    for i in range(len(pts) - 1))
                net_disp = float(np.linalg.norm(
                    np.array(pts[-1]) - np.array(pts[0])))
                if path_length > 0.001:
                    straightness = net_disp / path_length
                else:
                    straightness = 0.0
            else:
                straightness = 0.5

            if hspeed > opts.hspeed_fast:
                lo = opts.hspeed_fast_logodds * max(straightness, 0.1)
            elif hspeed > opts.hspeed_slow:
                t = (hspeed - opts.hspeed_slow) / (opts.hspeed_fast - opts.hspeed_slow)
                raw = (opts.hspeed_planted_logodds * (1.0 - t)
                       + opts.hspeed_fast_logodds * t)
                if raw < 0:
                    lo = raw * max(straightness, 0.1)
                else:
                    lo = raw
            else:
                if h < 0.08:
                    lo = opts.hspeed_planted_logodds
                else:
                    lo = 0.0

            # Attenuate negative signal during deceleration
            # Only when the foot was genuinely moving fast (not normal walking approach)
            if lo < 0 and decel_ratio > 0.1 and slow > 0.8:
                attenuation = 1.0 - min(0.9, decel_ratio)  # cap at 90% reduction
                lo *= attenuation

            result[gname] = lo
            context[gname] = {
                'hspeed': hspeed,
                'straightness': straightness,
                'decel': decel_ratio,
            }

        return result, context


class TouchdownStream:
    """Landing detection via dual-EMA crossover on vertical velocity.

    Uses the difference between a fast EMA and a slow EMA of vertical
    velocity as a noise-resistant acceleration proxy. The zero crossing
    of (fast_EMA - slow_EMA) is a leading indicator of touchdown:

      Descending: vy negative, fast leads slow downward → delta < 0
      Impact:     vy suddenly goes toward zero, fast responds first
                  → delta crosses zero from below = TOUCHDOWN
      Planted:    both EMAs near zero → delta ≈ 0 (neutral)
      Liftoff:    vy goes positive, fast leads slow upward → delta > 0
                  (not used here — velocity stream handles liftoff)

    This fires a strong positive burst on the zero crossing frame,
    providing a leading contact signal that other streams can't match.
    """

    def __init__(self):
        self._fast_ema = {}  # gname → fast EMA of vy
        self._slow_ema = {}  # gname → slow EMA of vy
        self._prev_delta = {}  # gname → previous (fast - slow) for crossing detection
        self._prev_h = {}  # gname → previous height

    def compute(self, pos, floor_height, dt, opts):
        """Compute per-group log-odds increment from dual-EMA crossing.

        Returns:
            Dict[str, float]: log-odds increment per group
        """
        up = opts.up_axis
        heights = pos[:, up] - floor_height
        result = {}

        alpha_fast = opts.td_alpha_fast
        alpha_slow = opts.td_alpha_slow

        for gname, joints in get_active_groups(opts).items():
            valid = [j for j in joints if j < len(heights)]
            if not valid:
                result[gname] = 0.0
                continue

            rep_j = min(valid, key=lambda j: heights[j])
            h = heights[rep_j]

            # Compute raw vy
            prev_h = self._prev_h.get(gname)
            if prev_h is None:
                self._prev_h[gname] = h
                self._fast_ema[gname] = 0.0
                self._slow_ema[gname] = 0.0
                self._prev_delta[gname] = 0.0
                result[gname] = 0.0
                continue

            raw_vy = (h - prev_h) / max(dt, 1e-6)  # positive = ascending
            self._prev_h[gname] = h

            # Update dual EMAs
            fast = alpha_fast * raw_vy + (1.0 - alpha_fast) * self._fast_ema[gname]
            slow = alpha_slow * raw_vy + (1.0 - alpha_slow) * self._slow_ema[gname]
            self._fast_ema[gname] = fast
            self._slow_ema[gname] = slow

            # Delta = fast - slow
            # Negative when descending (fast leads slow down)
            # Crosses zero upward at touchdown inflection
            delta = fast - slow
            prev_delta = self._prev_delta[gname]
            self._prev_delta[gname] = delta

            lo = 0.0

            # Zero crossing detection: prev_delta < 0 AND delta >= 0
            # = deceleration inflection (landing moment)
            # Gate by abs(slow_ema): the foot must have been genuinely falling.
            # Swing nadir has slow_ema ≈ -0.02 (blocked by descent_gate=0.1)
            # Walking heel strike has slow_ema ≈ -0.5 (passes)
            # Leap landing has slow_ema ≈ -2.4 (passes, full strength)
            descent_speed = abs(slow)
            if (prev_delta < 0 and delta >= 0
                    and descent_speed > opts.td_descent_gate):
                if h < opts.td_height_gate:
                    # Strength proportional to how fast the foot was falling
                    strength = min(1.0, descent_speed / opts.td_descent_scale)
                    lo = opts.td_landing_logodds * strength

            # Post-landing settling: delta positive + near floor + was descending
            elif (delta > 0 and h < opts.td_height_gate
                  and descent_speed > opts.td_descent_gate):
                lo = opts.td_settling_logodds * min(1.0, descent_speed / opts.td_descent_scale)

            result[gname] = lo

        return result


class EquilibriumStream:
    """Structural necessity evidence via ZMP proximity.

    The Zero Moment Point (ZMP) tells us where the body's balance
    is concentrated. When the ZMP snaps to one foot, the other foot
    is not supporting weight — a strong liftoff signal.

    ZMP = CoM_hz - (CoM_height / (g + acc_up)) * acc_hz

    For each foot group:
      - ZMP near this foot → positive (supporting weight)
      - ZMP far from this foot → negative (not supporting)

    Also includes freefall detection and physical assertion layer.
    """

    def __init__(self):
        pass

    def compute(self, pos, com, com_vel, com_acc, intensities,
                floor_height, total_mass, opts):
        """Compute per-group log-odds from ZMP proximity.

        Args:
            pos: (J, 3) joint positions
            com: (3,) CoM position
            com_vel: (3,) CoM velocity
            com_acc: (3,) CoM acceleration
            intensities: Dict[gname, float] current intensities
            floor_height: float
            total_mass: float
            opts: LogOddsContactOptions

        Returns:
            Dict[group_name, float]: log-odds increment
        """
        up = opts.up_axis
        plane = [0, 2] if up == 1 else [0, 1]

        # Check freefall
        accel_up = com_acc[up]
        freefall_ratio = abs(accel_up + opts.eq_gravity) / opts.eq_gravity
        is_freefall = freefall_ratio < (1.0 - opts.eq_freefall_thresh)

        result = {}

        if is_freefall:
            for gname in get_active_groups(opts):
                result[gname] = opts.eq_unnecessary_logodds
            return result

        # Compute ZMP
        com_h = max(com[up] - floor_height, 0.01)
        denom = opts.eq_gravity + accel_up
        if abs(denom) < 0.5:
            denom = 0.5 * np.sign(denom) if denom != 0 else 0.5
        zmp_hz = com[plane] - (com_h / denom) * com_acc[plane]

        # Dynamic confidence: high acceleration → unreliable ZMP → scale down
        # Horizontal acceleration magnitude is the key indicator
        hz_acc_mag = float(np.linalg.norm(com_acc[plane]))
        # Scale from 1.0 (calm, <2 m/s²) to 0.0 (very dynamic, >8 m/s²)
        dynamic_confidence = max(0.0, min(1.0, 1.0 - (hz_acc_mag - 2.0) / 6.0))

        # Compute distance from ZMP to each foot group centroid
        group_dist = {}
        group_centroid = {}
        for gname, joints in FOOT_GROUPS.items():
            valid = [j for j in joints if j < pos.shape[0]]
            if valid:
                centroid = np.mean([pos[j, plane] for j in valid], axis=0)
                group_centroid[gname] = centroid
                group_dist[gname] = float(np.linalg.norm(zmp_hz - centroid))
            else:
                group_dist[gname] = 999.0

        # For hands, use simple height proximity (ZMP is foot-relevant)
        heights = pos[:, up] - floor_height
        for gname, joints in HAND_GROUPS.items():
            valid = [j for j in joints if j < len(heights)]
            if valid:
                min_h = min(heights[j] for j in valid)
                if min_h < 0.05:
                    result[gname] = 0.5  # mild contact hint
                else:
                    result[gname] = 0.0
            else:
                result[gname] = 0.0

        # Score feet based on ZMP proximity, attenuated by dynamic confidence
        foot_names = list(FOOT_GROUPS.keys())
        if len(foot_names) >= 2 and all(group_dist[g] < 900 for g in foot_names):
            dists = [group_dist[g] for g in foot_names]
            total_dist = sum(dists)

            for i, gname in enumerate(foot_names):
                d = dists[i]

                if total_dist < 0.01:
                    result[gname] = opts.eq_necessity_logodds * 0.5 * dynamic_confidence
                else:
                    relative = d / total_dist

                    if relative < 0.35:
                        strength = (0.35 - relative) / 0.35
                        result[gname] = opts.eq_necessity_logodds * strength * dynamic_confidence
                    elif relative > 0.65:
                        # Negative evidence is more trustworthy —
                        # ZMP far from foot is informative even during dynamic movement
                        strength = (relative - 0.65) / 0.35
                        result[gname] = opts.eq_unnecessary_logodds * (1.0 + strength * 3.0)
                    else:
                        # ZMP equidistant — genuinely ambiguous, no opinion
                        result[gname] = 0.0
        else:
            for gname in foot_names:
                result[gname] = 0.0

        # --- Physical assertion layer ---
        # Only for steady-state conditions where the body needs support
        # but no foot has clear contact. NOT for dynamic transitions.
        if not is_freefall and dynamic_confidence > 0.5:
            any_contact = any(intensities.get(g, 0.0) > 0.2 for g in get_active_groups(opts))
            if not any_contact:
                # Only boost feet that are genuinely ambiguous (> 0.3),
                # not feet that are already fading to zero
                candidates = [g for g in foot_names
                              if intensities.get(g, 0.0) > 0.3
                              and group_dist.get(g, 999) < 900]
                if candidates:
                    closest = min(candidates, key=lambda g: group_dist[g])
                    result[closest] = result.get(closest, 0.0) + opts.eq_assertion_logodds

        return result


# ─────────────────────────────────────────────────────────────────────
# Log-Odds Accumulator
# ─────────────────────────────────────────────────────────────────────

class LogOddsAccumulator:
    """Per-group log-odds state with temporal decay.

    Each frame:
      1. state *= decay_rate  (decay toward neutral=0)
      2. state += sum(stream increments)
      3. intensity = sigmoid(state)
    """

    def __init__(self):
        self._state = {}  # gname → float (log-odds)

    def reset(self, initial=0.0):
        for g in self._state:
            self._state[g] = initial

    def update(self, increments, decay_rate, max_logodds=8.0):
        """Update accumulated state.

        Args:
            increments: Dict[gname, float] — total log-odds increment per group
            decay_rate: float or Dict[gname, float] — per-frame decay factor
                (0.9 = 10% decay toward 0). If dict, per-group decay rates.
            max_logodds: float — clamp magnitude to prevent over-certainty

        Returns:
            Dict[gname, float]: updated log-odds state
        """
        for g, inc in increments.items():
            prev = self._state.get(g, 0.0)
            # Per-group or global decay
            dr = decay_rate.get(g, 0.9) if isinstance(decay_rate, dict) else decay_rate
            # Decay toward neutral (0)
            decayed = prev * dr
            # Add new evidence and clamp
            new_val = decayed + inc
            new_val = max(-max_logodds, min(max_logodds, new_val))
            self._state[g] = new_val

        return dict(self._state)

    def get_intensities(self, temperature=2.0):
        """Map log-odds state to intensity [0, 1] via sigmoid.

        Args:
            temperature: Controls slope of sigmoid. Higher = more gradual.
                T=1.0: standard sigmoid (binary). T=2.0: gradual transitions.
        """
        result = {}
        for g, lo in self._state.items():
            # Clamp to avoid overflow in exp
            scaled = max(-10.0, min(10.0, lo / temperature))
            result[g] = 1.0 / (1.0 + np.exp(-scaled))
        return result

    @property
    def state(self):
        return dict(self._state)


# ─────────────────────────────────────────────────────────────────────
# Pressure Distribution (intensity-weighted)
# ─────────────────────────────────────────────────────────────────────

class IntensityPressureModel:
    """Distribute force based on continuous intensity rather than binary state.

    Uses XCoM lever rule for inter-foot and ball-heel splitting,
    weighted by each group's intensity.
    """

    def __init__(self):
        self._split_weights = {}  # gname → {j: weight}
        self._prev_joint_pos = {}  # gname → {j: float} vertical positions

    def distribute(self, intensities, pos, xcom_hz, total_mass,
                   floor_height, opts, surface_dists=None):
        """Distribute force based on contact intensities.

        Args:
            intensities: Dict[gname, float] — 0..1 intensity per group
            pos: (J, 3) joint positions
            xcom_hz: (2,) XCoM horizontal position
            total_mass: float
            floor_height: float
            opts: LogOddsContactOptions
            surface_dists: Optional (J,) per-joint distance from joint
                center to mesh surface in floor-ward direction

        Returns:
            np.ndarray: (J,) pressure per joint in kg
        """
        up = opts.up_axis
        plane = [0, 2] if up == 1 else [0, 1]
        J = pos.shape[0]
        pressure = np.zeros(J)

        # Unified force distribution: all contact groups (feet AND hands)
        # compete for body weight via intensity-weighted lever rule.
        # During cartwheels, hands bear full body weight.
        active_groups = {}
        groups_map = get_active_groups(opts)
        BODY_DEADZONE = 0.15  # Body contacts need higher confidence
        for g in groups_map:
            intensity = intensities.get(g, 0.0)
            dz = BODY_DEADZONE if g in BODY_GROUPS else opts.intensity_deadzone
            if intensity > dz:
                active_groups[g] = intensity

        if not active_groups:
            return pressure

        # Intensity-weighted XCoM lever rule between all active groups
        total_intensity = sum(active_groups.values())
        group_forces = {}
        all_groups_map = groups_map

        if len(active_groups) == 1:
            g = list(active_groups.keys())[0]
            group_forces[g] = total_mass * active_groups[g]
        else:
            # Two-regime distribution.
            #
            # Naive per-group force (matches single-group rule): each group
            # bears force proportional to its own intensity.  Sum these.
            #
            # Below body-weight total: use the naive forces directly — the
            # contacts are too uncertain to be claiming the full body, so
            # force scales with confidence and the body weight is "missing"
            # (consistent with the kinematic interpretation: the body is in
            # transit / the contacts are tentative).
            #
            # At or above body-weight total: switch to the XCoM lever rule
            # to redistribute exactly total_mass among the groups, weighted
            # by both proximity to XCoM and intensity.  This is the standard
            # established-contact behaviour.
            naive_forces = {g: total_mass * active_groups[g]
                            for g in active_groups}
            total_naive = sum(naive_forces.values())

            if total_naive <= total_mass:
                group_forces = naive_forces
            else:
                centroids = {}
                for gname in active_groups:
                    joints = all_groups_map[gname]
                    valid = [j for j in joints if j < J]
                    if valid:
                        centroids[gname] = np.mean(
                            [pos[j][plane] for j in valid], axis=0)

                if len(centroids) >= 2:
                    inv_dists = {}
                    for g, c in centroids.items():
                        d = max(0.01, np.linalg.norm(c - xcom_hz))
                        # Weight by both proximity and intensity
                        inv_dists[g] = active_groups[g] / d
                    total_inv = sum(inv_dists.values())
                    for g in active_groups:
                        group_forces[g] = (total_mass
                                           * inv_dists.get(g, 0)
                                           / max(total_inv, 1e-6))
                else:
                    for g in active_groups:
                        w = active_groups[g] / total_intensity
                        group_forces[g] = total_mass * w

        # Within-group joint split for foot groups (ball/heel)
        for gname in active_groups:
            if gname not in FOOT_GROUPS:
                continue
            gforce = group_forces.get(gname, 0)
            if gforce <= 0:
                continue

            ball_j = BALL_MAP.get(gname)
            heel_j = HEEL_MAP.get(gname)
            if ball_j is None or heel_j is None:
                continue
            members = [ball_j, heel_j]
            valid = [j for j in members if j < J]

            if len(valid) < 2:
                for j in valid:
                    pressure[j] = gforce / max(len(valid), 1)
                continue

            # XCoM lever rule projection
            ball_hz = pos[ball_j][plane]
            heel_hz = pos[heel_j][plane]
            foot_vec = ball_hz - heel_hz
            foot_len = np.linalg.norm(foot_vec)

            if foot_len > 0.01:
                t = np.dot(xcom_hz - heel_hz, foot_vec) / (foot_len ** 2)
                # Moderate clamp: lever angle handles extremes, XCoM only
                # needs to work in the flat-foot regime
                t = float(np.clip(t, 0.15, 0.85))
                xcom_weights = {ball_j: t, heel_j: 1.0 - t}
            else:
                xcom_weights = {ball_j: 0.5, heel_j: 0.5}

            # Lever angle: the pitch of the ball-heel vector directly
            # encodes the ankle (or wrist) lever state. This is the same
            # information as the joint angle, read from world positions.
            # At 0° (flat foot): both joints equal, XCoM fine-tunes.
            # At large angles (dorsiflexed/plantarflexed): the lower joint
            # smoothly takes all the weight.
            # Use surface contact positions if available: joint height
            # minus the mesh surface offset gives the actual ground
            # contact height, correcting for SMPL's structural bias
            # where the ball joint sits higher than the heel.
            ball_surface_h = pos[ball_j, up]
            heel_surface_h = pos[heel_j, up]
            if surface_dists is not None:
                if ball_j < len(surface_dists):
                    ball_surface_h -= surface_dists[ball_j]
                if heel_j < len(surface_dists):
                    heel_surface_h -= surface_dists[heel_j]

            # Lever angle from surface contact heights
            foot_vec_hz = pos[ball_j] - pos[heel_j]
            foot_len_hz = np.linalg.norm(foot_vec_hz[plane])
            surface_h_diff = ball_surface_h - heel_surface_h

            if foot_len_hz > 0.01:
                # sin(pitch) from surface heights, not joint heights
                sin_pitch = float(np.clip(surface_h_diff / max(foot_len_hz, 0.01), -1.0, 1.0))
            else:
                sin_pitch = 0.0

            # Lever weights: sharp transfer — any significant pitch
            # pushes all weight to the lower joint. Only near flat
            # (within ±~6°) is pressure shared.
            transfer_scale = 0.2  # sin(11°)≈0.2: full transfer at ~11°
            mapped = float(np.clip(sin_pitch / transfer_scale, -1.0, 1.0))

            lever_weights = {
                ball_j: 0.5 * (1.0 - mapped),   # ball lower → more weight
                heel_j: 0.5 * (1.0 + mapped),    # heel lower → more weight
            }

            # Blend: when flat (|mapped|≈0), XCoM decides.
            # When lever active (|mapped|→1), lever dominates.
            lever_blend = abs(mapped)
            raw_weights = {}
            for j in valid:
                raw_weights[j] = ((1.0 - lever_blend) * xcom_weights.get(j, 0.5)
                                  + lever_blend * lever_weights.get(j, 0.5))

            # Normalize and apply
            w_sum = sum(raw_weights.values())
            if w_sum > 0:
                for j in raw_weights:
                    raw_weights[j] /= w_sum

            self._split_weights[gname] = raw_weights
            for j in valid:
                pressure[j] = gforce * raw_weights.get(j, 0.5)

        # Within-group joint split for hand groups (wrist/hand)
        for gname in active_groups:
            if gname not in HAND_GROUPS:
                continue
            gforce = group_forces.get(gname, 0)
            if gforce <= 0:
                continue
            joints = HAND_GROUPS[gname]
            valid = [j for j in joints if j < J]
            if not valid:
                continue
            # Equal split for now (hands don't have ball/heel asymmetry)
            force_per = gforce / len(valid)
            for j in valid:
                pressure[j] = force_per

        # Body groups: single joint, no intra-group split needed
        for gname in active_groups:
            if gname not in BODY_GROUPS:
                continue
            gforce = group_forces.get(gname, 0)
            if gforce <= 0:
                continue
            joints = BODY_GROUPS[gname]
            for j in joints:
                if j < J:
                    pressure[j] = gforce

        return pressure


# ─────────────────────────────────────────────────────────────────────
# Main Estimator
# ─────────────────────────────────────────────────────────────────────

class LogOddsContactEstimator:
    """Continuous contact estimator using additive log-odds.

    Usage:
        estimator = LogOddsContactEstimator(total_mass_kg=75.0)
        result = estimator.process_frame(pos, com, com_vel, com_acc,
                                          floor_height, dt, opts)
        # result.intensity['LF']  → 0.0 to 1.0
        # result.per_stream['LF']['height'] → signed log-odds contribution
        # result.pressure_array → (J,) force distribution
    """

    def __init__(self, framerate=60.0, total_mass_kg=75.0, segment_masses=None,
                 frame_evaluator=None):
        self.framerate = framerate
        self.total_mass = total_mass_kg

        # Components
        self.accumulator = LogOddsAccumulator()
        self.height_stream = HeightStream()
        # New 3-stream architecture
        self.kinematic_stream = KinematicStream()
        self.structural_stream = StructuralStream(evaluator=frame_evaluator)
        self.divergence_stream = DivergenceStream()
        self.pressure_model = IntensityPressureModel()
        # Legacy streams (kept for A/B testing)
        self.vertical_kinematic_stream = VerticalKinematicStream()
        self.hspeed_stream = HorizontalSpeedStream()
        self.equilibrium_stream = EquilibriumStream()
        self.velocity_stream = VelocityStream()
        self.trajectory_stream = TrajectoryStream()
        self.touchdown_stream = TouchdownStream()

    def _valve_streams(self, increments, context, opts, pos, floor_height, dt):
        """Adjust stream contributions based on cross-stream context.

        When opts.enable_valving is False, this is a pass-through.
        When True, applies data-driven valving rules to suppress
        misleading stream contributions.

        Args:
            increments: Dict[stream_name, Dict[gname, float]] — raw log-odds
            context: Dict[gname, Dict[str, float]] — per-group context
                     (vy, decel, height_m, hspeed, straightness)
            opts: LogOddsContactOptions
            pos: (J, 3) joint positions
            floor_height: float
            dt: float

        Returns:
            Dict[stream_name, Dict[gname, float]] — adjusted increments
        """
        if not opts.enable_valving:
            return increments

        # Deep copy to avoid mutating originals
        valved = {s: dict(d) for s, d in increments.items()}

        # ─── Push-off rate tracking ───
        # Track the rate of change of the secondary-primary joint height
        # differential (ankle-ball for feet, wrist-hand for hands).
        # During push-off, the secondary rises relative to the primary.
        # EMA-smoothed to reject frame-to-frame noise.
        if not hasattr(self, '_valve_pushoff_prev_diff'):
            self._valve_pushoff_prev_diff = {}
            self._valve_pushoff_ema_rate = {}

        PUSHOFF_EMA_ALPHA = 0.4
        up = opts.up_axis
        n_joints = pos.shape[0]
        pushoff_rates = {}  # gname → EMA'd rate

        for gname, (sec_j, pri_j) in PUSHOFF_JOINTS.items():
            if sec_j >= n_joints or pri_j >= n_joints:
                continue
            diff = float(pos[sec_j, up] - pos[pri_j, up])
            prev = self._valve_pushoff_prev_diff.get(gname)
            if prev is not None:
                raw_rate = (diff - prev) / max(dt, 1e-6)
            else:
                raw_rate = 0.0
            self._valve_pushoff_prev_diff[gname] = diff

            prev_ema = self._valve_pushoff_ema_rate.get(gname, 0.0)
            ema_rate = PUSHOFF_EMA_ALPHA * raw_rate + (1.0 - PUSHOFF_EMA_ALPHA) * prev_ema
            self._valve_pushoff_ema_rate[gname] = ema_rate
            pushoff_rates[gname] = ema_rate

        # ─── Rule 1: Vertical motion attenuates positive height evidence ───
        # A foot/hand moving vertically through the contact zone provides
        # unreliable height evidence. Uses the kinematic stream's
        # dual-EMA fast value (smoothed vy, ~2-frame time constant)
        # as a noise-robust velocity estimate.
        #
        # Shape: quartic Butterworth-like rolloff
        #   valve = 1 / (1 + (vy_fast / sigma_eff)^4)
        # This is flat near zero (noise-transparent) and drops
        # smoothly for genuine vertical motion. No hard thresholds.
        #
        # Push-off amplification: when the ankle/wrist is rising relative
        # to the ball/hand (push-off biomechanics), sigma_eff decreases,
        # making the valve more aggressive. This gives earlier liftoff
        # detection during push-off without affecting planted noise.
        # Relevé safety: in relevé the ankle rises but the ball vy stays
        # near zero, so the valve still produces ~1.0 (no attenuation).
        #
        # Only attenuates POSITIVE height evidence (contact zone).
        # Negative evidence (foot clearly high) passes unchanged.
        #
        # Restricted to PRIMARY_GROUPS (feet + hands). Body contacts
        # (knees, elbows, head, pelvis) have different kinematics
        # and are not affected by this valve.
        if 'height' in valved:
            SIGMA_BASE = 0.15     # m/s: default rolloff scale
            PUSHOFF_GAIN = 3.0    # how much push-off rate reduces sigma
            for gname in valved['height']:
                if gname not in PRIMARY_GROUPS:
                    continue
                h_lo = valved['height'][gname]
                if h_lo > 0:
                    vy_fast = context.get(gname, {}).get('vy_fast_ema', 0.0)
                    # Push-off modulation: only positive rates (rising secondary)
                    po_rate = max(0.0, pushoff_rates.get(gname, 0.0))
                    sigma_eff = SIGMA_BASE / (1.0 + PUSHOFF_GAIN * po_rate)
                    ratio = vy_fast / sigma_eff
                    valve = 1.0 / (1.0 + ratio * ratio * ratio * ratio)
                    valved['height'][gname] = h_lo * valve

        # ─── Rule 2: Rapid descent attenuates positive divergence evidence ───
        # The divergence stream can fire positive for a descending foot
        # that hasn't landed yet (e.g., swing foot passing near the floor).
        # When vy_fast is strongly negative (rapid descent), attenuate
        # positive divergence evidence. Once the foot settles (vy_fast→0),
        # divergence evidence passes through normally.
        #
        # Uses a larger sigma than the height valve since the divergence
        # stream is useful during slow near-floor movement (walking).
        # Only attenuates for descent (negative vy_fast).
        if 'divergence' in valved:
            DIV_SIGMA = 0.30  # m/s: descent speed for 50% attenuation
            for gname in valved['divergence']:
                if gname not in PRIMARY_GROUPS:
                    continue
                div_lo = valved['divergence'][gname]
                if div_lo > 0:
                    vy_fast = context.get(gname, {}).get('vy_fast_ema', 0.0)
                    if vy_fast < 0:
                        # Only attenuate during descent
                        ratio = vy_fast / DIV_SIGMA
                        valve = 1.0 / (1.0 + ratio * ratio * ratio * ratio)
                        valved['divergence'][gname] = div_lo * valve

        return valved

    def _check_foot_structural_deficit(self, plane):
        """Check if foot contacts alone can explain the current dynamics.

        Uses the previous frame's structural evaluation to determine
        whether the ZMP lies inside the support polygon of foot-only
        contacts.  If not, or if no foot contacts are active, returns
        True indicating a structural deficit that may require body
        contacts.

        Args:
            plane: [int, int] horizontal plane indices (e.g. [0, 2] for Y-up)

        Returns:
            bool: True if feet can't explain dynamics (body contacts may
                  be needed), False if feet are sufficient.
        """
        fe = getattr(self.structural_stream, 'last_eval_result', None)
        if fe is None:
            return False  # No data yet — don't activate

        # Check if any foot contacts are active with meaningful force
        foot_joints = set()
        for gname in FOOT_GROUPS:
            foot_joints.update(FOOT_GROUPS[gname])
        foot_force = sum(
            fe.per_contact_force.get(j, 0.0) for j in foot_joints
        )
        if foot_force < 1.0:
            # No meaningful foot force → structural deficit
            return True

        # Get foot-only contact positions in horizontal plane
        foot_positions = []
        for j, force in fe.per_contact_force.items():
            if j in foot_joints and force > 0.5:
                foot_positions.append(fe.zmp_approx * 0)  # placeholder
        # Actually use the support polygon from the eval result,
        # but filter to foot contacts only
        if not fe.support_polygon or len(fe.support_polygon) < 1:
            return True  # No support polygon → deficit

        # Check ZMP vs foot support polygon:
        # Simple approach — is the ZMP within reasonable distance
        # of the foot contact centroid?  For a full sitting-down,
        # the ZMP moves far behind the feet.
        zmp = fe.zmp_approx  # (2,) horizontal
        if zmp is None or np.any(np.isnan(zmp)):
            return False

        # Compute foot centroid from support polygon
        poly = np.array(fe.support_polygon)  # (N, 2)
        if len(poly) == 0:
            return True

        centroid = poly.mean(axis=0)

        # Simple containment: is ZMP within the support polygon?
        # Use signed area method for convex polygon check.
        # For simplicity, use distance from centroid vs polygon radius.
        max_radius = np.max(np.linalg.norm(poly - centroid, axis=1))
        max_radius = max(max_radius, 0.05)  # min 5cm

        zmp_dist = np.linalg.norm(zmp - centroid)

        # If ZMP is more than 1.5x the polygon radius from centroid,
        # feet can't explain it → structural deficit
        return zmp_dist > max_radius * 1.5

    def process_frame(self, pos, com, com_vel, com_acc,
                      floor_height, dt, opts=None, raw_com_acc=None,
                      surface_dists=None, gravity_torque_vecs=None):
        """Process one frame.

        Args:
            pos: (J, 3) joint positions
            com: (3,) center of mass
            com_vel: (3,) CoM velocity
            com_acc: (3,) CoM acceleration (filtered)
            floor_height: float
            dt: float
            opts: LogOddsContactOptions
            raw_com_acc: (3,) optional unfiltered CoM acceleration. Used by
                the structural stream for freefall detection. Falls back to
                ``self._raw_com_acc`` for backward compatibility.

        Returns:
            LogOddsContactResult
        """
        if opts is None:
            opts = LogOddsContactOptions()
        if raw_com_acc is None:
            raw_com_acc = getattr(self, '_raw_com_acc', None)

        up = opts.up_axis
        plane = [0, 2] if up == 1 else [0, 1]

        # --- CoM-gate body contacts ---
        # Body groups only activate when CoM is near the floor AND the
        # feet alone can't explain the dynamics (structural deficit).
        # This prevents false body contacts during squats where CoM is
        # low but feet perfectly explain the balance.
        # Hysteresis: activate at < 0.5m, deactivate at > 0.7m.
        BODY_ACTIVATE_THRESH = 0.5   # CoM must drop below this to engage
        BODY_DEACTIVATE_THRESH = 0.7  # CoM must rise above this to disengage
        if opts.enable_body_contacts and com is not None:
            com_h = com[up] - floor_height
            was_active = getattr(self, '_body_contacts_active', False)

            # --- Structural deficit check ---
            # Check if foot contacts alone can explain the dynamics.
            # Use the previous frame's foot-only structural evaluation:
            # if ZMP is inside the foot support polygon, feet handle it.
            has_structural_deficit = self._check_foot_structural_deficit(plane)

            if was_active:
                body_active = com_h <= BODY_DEACTIVATE_THRESH
            else:
                # Require BOTH low CoM AND structural deficit to activate
                body_active = (com_h <= BODY_ACTIVATE_THRESH
                               and has_structural_deficit)
            self._body_contacts_active = body_active
            if not body_active:
                opts = replace(opts, enable_body_contacts=False)

        # --- Compute per-stream increments ---
        stream_increments = {}  # stream_name → {gname: float}
        active_groups = get_active_groups(opts)
        all_group_names = list(active_groups.keys())

        if opts.enable_height:
            h_inc = self.height_stream.compute(
                pos, floor_height, opts, surface_dists=surface_dists)
            stream_increments['height'] = {
                g: v * opts.weight_height for g, v in h_inc.items()
            }

        # Collect context from streams that provide it
        stream_context = {g: {} for g in all_group_names}

        # --- New 3-stream architecture ---
        if opts.enable_kinematic:
            kin_inc, kin_ctx = self.kinematic_stream.compute(
                pos, floor_height, dt, opts)
            # Attenuate kinematic NEGATIVE evidence for non-foot groups
            # when CoM is low (floor work). Hands/body parts can slide
            # on the floor — the "moving means airborne" assumption
            # doesn't hold during floor work.
            if (opts.enable_body_contacts and com is not None):
                com_h = com[up] - floor_height
                # Scale: 1.0 at com_h >= 0.5m, 0.0 at com_h <= 0.2m
                kin_atten = max(0.0, min(1.0, (com_h - 0.2) / 0.3))
                for g in list(kin_inc.keys()):
                    if g not in FOOT_GROUPS and kin_inc[g] < 0:
                        kin_inc[g] *= kin_atten
            stream_increments['kinematic'] = {
                g: v * opts.weight_kinematic for g, v in kin_inc.items()
            }
            for g in all_group_names:
                stream_context[g].update(kin_ctx.get(g, {}))

        if opts.enable_structural:
            # Get current intensities so structural stream can prioritize
            # established contacts over new candidates
            curr_int = self.accumulator.get_intensities(opts.sigmoid_temperature)
            struct_inc = self.structural_stream.compute(
                pos, com, com_acc, com_vel, floor_height, dt, opts,
                raw_com_acc=raw_com_acc, intensities=curr_int,
                gravity_torque_vecs=gravity_torque_vecs)
            stream_increments['structural'] = {
                g: v * opts.weight_structural for g, v in struct_inc.items()
            }

        if opts.enable_divergence:
            div_inc, div_ctx = self.divergence_stream.compute(
                pos, com, com_vel, floor_height, dt, opts)
            stream_increments['divergence'] = {
                g: v * opts.weight_divergence for g, v in div_inc.items()
            }
            for g in all_group_names:
                stream_context[g].update(div_ctx.get(g, {}))

        # --- Legacy streams (for A/B testing, disabled by default) ---
        if opts.enable_vertical_kinematic:
            vk_inc, vk_ctx = self.vertical_kinematic_stream.compute(
                pos, floor_height, dt, opts)
            stream_increments['vertical_kinematic'] = {
                g: v * opts.weight_vertical_kinematic for g, v in vk_inc.items()
            }
            for g in all_group_names:
                stream_context[g].update(vk_ctx.get(g, {}))

        if opts.enable_velocity:
            v_inc, v_ctx = self.velocity_stream.compute(pos, floor_height, dt, opts)
            stream_increments['velocity'] = {
                g: v * opts.weight_velocity for g, v in v_inc.items()
            }
            for g in all_group_names:
                stream_context[g].update(v_ctx.get(g, {}))

        if opts.enable_trajectory:
            t_inc = self.trajectory_stream.compute(pos, floor_height, dt, opts)
            stream_increments['trajectory'] = {
                g: v * opts.weight_trajectory for g, v in t_inc.items()
            }

        if opts.enable_touchdown:
            td_inc = self.touchdown_stream.compute(pos, floor_height, dt, opts)
            stream_increments['touchdown'] = {
                g: v * opts.weight_touchdown for g, v in td_inc.items()
            }

        if opts.enable_hspeed:
            hs_inc, hs_ctx = self.hspeed_stream.compute(pos, floor_height, dt, opts)
            stream_increments['hspeed'] = {
                g: v * opts.weight_hspeed for g, v in hs_inc.items()
            }
            for g in all_group_names:
                stream_context[g].update(hs_ctx.get(g, {}))

        if opts.enable_equilibrium:
            curr_intensities = self.accumulator.get_intensities(opts.sigmoid_temperature)
            eq_inc = self.equilibrium_stream.compute(
                pos, com, com_vel, com_acc, curr_intensities,
                floor_height, self.total_mass, opts)
            stream_increments['equilibrium'] = {
                g: v * opts.weight_equilibrium for g, v in eq_inc.items()
            }

        # --- Valve step ---
        stream_increments = self._valve_streams(
            stream_increments, stream_context, opts, pos, floor_height, dt)

        # --- Sum increments across streams ---
        total_increments = {g: 0.0 for g in all_group_names}
        for stream_name, inc_dict in stream_increments.items():
            for g in all_group_names:
                total_increments[g] += inc_dict.get(g, 0.0)

        # --- Update accumulator ---
        # Single decay rate for all groups. The structural stream's
        # established-contact priority provides persistence; no need
        # for per-group sticky decay rates.
        self.accumulator.update(total_increments, opts.decay_rate, opts.max_logodds)
        intensities = self.accumulator.get_intensities(opts.sigmoid_temperature)
        log_odds_state = self.accumulator.state

        # --- 1-frame onset latency filter ---
        # Suppress single-frame contact flickers: a group must be above
        # the deadzone for 2 consecutive frames before intensity passes
        # through.  Release (drop below threshold) is immediate.
        # The accumulator state is NOT affected — only the output.
        dz = opts.intensity_deadzone
        if not hasattr(self, '_prev_above_dz'):
            self._prev_above_dz = {}
        filtered = {}
        for g, raw_i in intensities.items():
            above = raw_i > dz
            was_above = self._prev_above_dz.get(g, False)
            if above and not was_above:
                # First frame above threshold — suppress
                filtered[g] = 0.0
            else:
                filtered[g] = raw_i
            self._prev_above_dz[g] = above
        intensities = filtered

        # --- Build per-stream diagnostic ---
        per_stream = {}
        for g in all_group_names:
            per_stream[g] = {
                'decay': -(1.0 - opts.decay_rate) * (
                    log_odds_state.get(g, 0.0) - total_increments.get(g, 0.0)),
            }
            for stream_name, inc_dict in stream_increments.items():
                per_stream[g][stream_name] = inc_dict.get(g, 0.0)
            per_stream[g]['total_increment'] = total_increments.get(g, 0.0)

        # --- Binary contact (for compatibility) ---
        contact_state = {g: (intensities.get(g, 0.0) > 0.5)
                         for g in all_group_names}


        # --- Pressure distribution ---
        # XCoM for lever rule
        com_h = com[up] - floor_height
        g_mag = 9.81
        xcom_scale = np.sqrt(max(com_h, 0.01) / g_mag)
        xcom = com.copy()
        if com_vel is not None:
            xcom += com_vel * xcom_scale
        xcom_hz = xcom[plane]

        pressure = self.pressure_model.distribute(
            intensities, pos, xcom_hz, self.total_mass,
            floor_height, opts, surface_dists=surface_dists)

        return LogOddsContactResult(
            intensity=intensities,
            log_odds_state=log_odds_state,
            per_stream=per_stream,
            stream_context=stream_context,
            pressure_array=pressure,
            contact_state=contact_state,
        )
