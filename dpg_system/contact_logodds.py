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

import numpy as np
from dataclasses import dataclass, field
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
ALL_GROUPS = {**FOOT_GROUPS, **HAND_GROUPS}
HEEL_MAP = {'LF': 28, 'RF': 29}
BALL_MAP = {'LF': 10, 'RF': 11}


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
    max_logodds: float = 3.0      # Clamp: sigmoid(3)=0.953, sigmoid(-3)=0.047
                                   # Lower clamp = less time stuck at extremes,
                                   # more visible intensity gradation

    # Stream enables (3-stream architecture)
    enable_height: bool = True
    enable_kinematic: bool = True      # Unified kinematic stream (approach angle + touchdown + settled)
    enable_structural: bool = True     # Frame evaluator structural necessity
    # Legacy enables (kept for backward compatibility / A/B testing)
    enable_vertical_kinematic: bool = False  # Old unified vertical-only stream
    enable_hspeed: bool = False              # Old separate horizontal speed stream
    enable_equilibrium: bool = False         # Old ZMP proximity stream
    enable_velocity: bool = False
    enable_trajectory: bool = False
    enable_touchdown: bool = False

    # Stream weights (multiplier on each stream's raw log-odds)
    weight_height: float = 1.0
    weight_kinematic: float = 1.0      # Unified kinematic stream
    weight_structural: float = 1.0     # Structural necessity stream
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
    kin_settle_floor_thresh: float = 0.005   # min_h below this = foot has reached the floor
    kin_settle_never_push: float = -0.15     # Airborne push when foot never settled
    kin_settle_speed_gate: float = 0.18      # Only apply when speed < this (low-speed gap filler)
    kin_settle_rise_reset: float = 0.012     # Rise above min_h to trigger epoch reset

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
    # Inverted pendulum: CoM falling → trailing foot is pivot,
    # leading foot is proven NOT fully supporting (pendulum still swinging)
    struct_pendulum_vy_thresh: float = -0.03  # m/s: below this = CoM is falling
    struct_pendulum_leading_logodds: float = -0.3  # negative for leading (not yet full support)
    struct_pendulum_min_separation: float = 0.15   # feet must be this far apart for directionality


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

    def compute(self, pos, floor_height, opts):
        """Compute per-group log-odds increment from height.

        Args:
            pos: (J, 3) joint positions
            floor_height: float
            opts: LogOddsContactOptions

        Returns:
            Dict[group_name, float]: log-odds increment per group
        """
        up = opts.up_axis
        heights = pos[:, up] - floor_height

        result = {}
        for gname, joints in ALL_GROUPS.items():
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

        for gname, joints in ALL_GROUPS.items():
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
                    # Stable and low: mild push toward grounded (arming only)
                    # Steady state: 0.04 / (1 - 0.85) = 0.27 (inside dead zone 0.30)
                    traj_push = 0.04 * confidence
                elif slope < -opts.traj_trend_thresh:
                    # Falling: push toward grounded (approaching/arming)
                    scale = min(1.0, -slope / (opts.traj_trend_thresh * 5))
                    traj_push = 0.03 * scale * confidence
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

        for gname, joints in ALL_GROUPS.items():
            valid = [j for j in joints if j < len(heights)]
            if not valid:
                result[gname] = 0.0
                context[gname] = {'phase': 0.0, 'angle_push': 0.0,
                                  'td_impulse': 0.0, 'vel_hold': 0.0,
                                  'angle_deg': 0.0, 'speed': 0.0}
                continue

            rep_j = min(valid, key=lambda j: heights[j])
            h = heights[rep_j]
            pos_3d = pos[rep_j]

            # --- Initialize on first frame ---
            if gname not in self._phase:
                self._phase[gname] = 0.0
                self._prev_pos[gname] = pos_3d.copy()
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
            prev_pos_3d = self._prev_pos[gname]
            vel_3d = (pos_3d - prev_pos_3d) / max(dt, 1e-6)
            self._prev_pos[gname] = pos_3d.copy()

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
            descent_speed = abs(slow)

            # Landing crossover: prev_delta < 0, delta >= 0, was genuinely falling
            if (prev_delta < 0 and delta >= 0
                    and descent_speed > opts.td_descent_gate
                    and h < opts.td_height_gate):
                strength = min(1.0, descent_speed / opts.td_descent_scale)
                td_impulse = 0.5 * strength

            # Liftoff crossover: fast drops below slow while near floor
            elif (prev_delta >= 0 and delta < 0
                  and abs(slow) < 0.15
                  and h < opts.td_height_gate
                  and phase > self.PLANTED_PHASE_THRESH):
                td_impulse = -0.3

            # =============================================================
            # SUB-SIGNAL 3: Settled / ascending state
            # =============================================================
            raw_vy_vel = (h - self._vel_prev_h[gname]) / max(dt, 1e-6)
            self._vel_prev_h[gname] = h

            prev_smooth = self._vel_smooth_vy.get(gname, 0.0)
            vy_smooth = 0.4 * raw_vy_vel + 0.6 * prev_smooth
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
            # If the foot has never reached near-zero height, it hasn't actually
            # landed — it's at the end of a swing arc or hovering.
            # Only pushes when speed is low (when the angle sub-signal has no opinion).
            never_settled_push = 0.0

            # Reset epoch on planted → not-planted transition (liftoff)
            was_planted = self._was_planted.get(gname, False)
            self._was_planted[gname] = is_planted
            if was_planted and not is_planted:
                # Foot just transitioned to "not planted" — reset approach tracking
                self._approach_min_h[gname] = h
                self._in_approach[gname] = h < opts.kin_settle_approach_zone

            if h < opts.kin_settle_approach_zone:
                # In approach zone — track minimum
                if self._in_approach.get(gname, False):
                    prev_min = self._approach_min_h.get(gname, h)
                    # Reset epoch if foot has risen significantly above its minimum
                    # (e.g., was at h=0.001 planted, now at h=0.020 lifting off)
                    rise_from_min = h - prev_min
                    if rise_from_min > opts.kin_settle_rise_reset:
                        # Foot departed from its minimum — start fresh epoch
                        self._approach_min_h[gname] = h
                    else:
                        self._approach_min_h[gname] = min(prev_min, h)
                else:
                    # Just entered approach zone — start epoch
                    self._approach_min_h[gname] = h
                    self._in_approach[gname] = True

                # Never-settled push: foot is in approach zone, speed is low
                # (angle has no opinion), but min_h never reached floor
                approach_min = self._approach_min_h.get(gname, h)
                if (speed < opts.kin_settle_speed_gate
                        and approach_min > opts.kin_settle_floor_thresh
                        and not is_planted):
                    # Foot never reached the floor — push toward airborne
                    # Scale by how far min_h is from the floor threshold
                    # (0.008m denominator: full strength at 1.3cm above floor thresh)
                    gap = min(1.0, (approach_min - opts.kin_settle_floor_thresh) / 0.008)
                    never_settled_push = opts.kin_settle_never_push * gap
            else:
                # Above approach zone — reset epoch
                self._in_approach[gname] = False
                self._approach_min_h[gname] = h

            # =============================================================
            # PHASE UPDATE
            # =============================================================
            phase *= opts.kin_phase_decay
            phase += angle_push
            phase += td_impulse
            phase += vel_hold
            phase += never_settled_push
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
                'vel_hold': vel_hold,
                'never_settled': never_settled_push,
                'angle_deg': angle_deg,
                'speed': speed,
                'vy': vy,
                'vh': vh,
                'angle_confidence': angle_confidence,
                'is_planted': float(is_planted),
                'approach_min_h': self._approach_min_h.get(gname, h),
            }

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

    # Contact joint indices that can be candidates
    CONTACT_JOINTS = {10, 11, 28, 29, 20, 21, 22, 23}

    # Group mapping: which CONTACT_JOINTS belong to which ALL_GROUPS group
    JOINT_TO_GROUP = {
        10: 'LF', 28: 'LF',
        11: 'RF', 29: 'RF',
        20: 'LH', 22: 'LH',
        21: 'RH', 23: 'RH',
    }

    def __init__(self, evaluator=None):
        """
        Args:
            evaluator: DynamicFrameEvaluator instance (or None, set later)
        """
        self.evaluator = evaluator

    def set_evaluator(self, evaluator):
        """Set or replace the frame evaluator instance."""
        self.evaluator = evaluator

    def compute(self, pos, com, com_acc, com_vel, floor_height, dt, opts,
                raw_com_acc=None):
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
        g_mag = opts.struct_gravity
        result = {g: 0.0 for g in ALL_GROUPS}

        # --- Freefall detection (raw acceleration if available) ---
        acc_for_freefall = raw_com_acc if raw_com_acc is not None else com_acc
        if acc_for_freefall is not None:
            f_up = acc_for_freefall[up] + g_mag
            support_frac = max(0.0, f_up / g_mag)
            if support_frac < opts.struct_freefall_thresh:
                for gname in ALL_GROUPS:
                    result[gname] = opts.struct_freefall_logodds
                return result

        # --- If no evaluator available, return neutral ---
        if self.evaluator is None:
            return result

        # --- Find candidates by height ---
        heights = pos[:, up] - floor_height
        candidates = set()
        for j in self.CONTACT_JOINTS:
            if j < len(heights) and heights[j] < opts.struct_candidate_height:
                candidates.add(j)

        if not candidates:
            # No joints near floor → mild negative for foot groups
            for gname in FOOT_GROUPS:
                result[gname] = opts.struct_unnecessary_logodds
            return result

        # --- Identify which groups have candidates ---
        groups_with_candidates = {}
        for j in candidates:
            gname = self.JOINT_TO_GROUP.get(j)
            if gname:
                if gname not in groups_with_candidates:
                    groups_with_candidates[gname] = []
                groups_with_candidates[gname].append(j)

        # --- Evaluate structural frame ---
        eval_result = self.evaluator.evaluate(
            candidates, pos, com, com_acc, floor_height, up)

        # --- Extract per-group max force ---
        group_forces = {}
        for gname, joints in groups_with_candidates.items():
            forces = [eval_result.per_contact_force.get(j, 0.0) for j in joints]
            group_forces[gname] = max(forces) if forces else 0.0

        # --- Signal 1: Negative force (tension) detection ---
        # If the FE says a contact needs to PULL, that's physically impossible
        for gname, force in group_forces.items():
            if force < -0.5:
                result[gname] = opts.struct_pulling_logodds

        # --- Signal 2: Single-group dominance ---
        # When only one foot group has candidates, it bears all weight
        foot_groups_present = [g for g in groups_with_candidates if g in FOOT_GROUPS]

        if len(foot_groups_present) == 1:
            sole_group = foot_groups_present[0]
            sole_force = group_forces.get(sole_group, 0.0)
            if sole_force > opts.struct_force_mild:
                result[sole_group] = opts.struct_necessary_logodds
            # Other foot groups get mild negative (not present)
            for gname in FOOT_GROUPS:
                if gname != sole_group and gname not in groups_with_candidates:
                    result[gname] = opts.struct_unnecessary_logodds

        elif len(foot_groups_present) >= 2:
            # Multiple foot groups present
            # Use inverted pendulum dynamics: when CoM is falling,
            # the pendulum is still anchored on the trailing foot.
            # This PROVES the leading foot is not (fully) supporting yet —
            # if it were, the CoM would start rising over the new pivot.
            plane = [0, 2] if up == 1 else [0, 1]

            # Compute per-group centroid in horizontal plane
            group_centroids = {}
            for gname in foot_groups_present:
                joints = groups_with_candidates[gname]
                centroid = np.mean([pos[j, plane] for j in joints], axis=0)
                group_centroids[gname] = centroid

            # Foot separation
            centroids_list = list(group_centroids.values())
            if len(centroids_list) >= 2:
                separation = np.linalg.norm(centroids_list[0] - centroids_list[1])
            else:
                separation = 0.0

            # CoM vertical velocity
            com_vy = com_vel[up] if com_vel is not None else 0.0

            if (com_vy < opts.struct_pendulum_vy_thresh
                    and separation > opts.struct_pendulum_min_separation
                    and com_vel is not None):
                # CoM is falling → pendulum still swinging on trailing foot
                # The leading foot is proven NOT fully supporting
                com_hz_vel = com_vel[plane]
                hz_speed = np.linalg.norm(com_hz_vel)

                if hz_speed > 0.05:  # meaningful horizontal movement
                    forward = com_hz_vel / hz_speed
                    com_hz = com[plane]

                    for gname, centroid in group_centroids.items():
                        offset = centroid - com_hz
                        projection = np.dot(offset, forward)

                        if projection >= 0:
                            # Leading foot: dynamics prove not fully engaged
                            result[gname] = opts.struct_pendulum_leading_logodds
                        # Trailing foot: neutral (pivot, but may be departing)
            # else: CoM not clearly falling → neutral for all

        # --- Groups with no candidates get mild negative ---
        for gname in FOOT_GROUPS:
            if gname not in groups_with_candidates and result[gname] == 0.0:
                result[gname] = opts.struct_unnecessary_logodds

        # Hand groups: negative if not in candidates
        for gname in HAND_GROUPS:
            if gname not in groups_with_candidates:
                result[gname] = -0.3

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

        for gname, joints in ALL_GROUPS.items():
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

        for gname, joints in ALL_GROUPS.items():
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

        for gname, joints in ALL_GROUPS.items():
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

        for gname, joints in ALL_GROUPS.items():
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
            for gname in ALL_GROUPS:
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
            any_contact = any(intensities.get(g, 0.0) > 0.2 for g in ALL_GROUPS)
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
            decay_rate: float — per-frame decay factor (0.9 = 10% decay toward 0)
            max_logodds: float — clamp magnitude to prevent over-certainty

        Returns:
            Dict[gname, float]: updated log-odds state
        """
        for g, inc in increments.items():
            prev = self._state.get(g, 0.0)
            # Decay toward neutral (0)
            decayed = prev * decay_rate
            # Add new evidence and clamp
            new_val = decayed + inc
            new_val = max(-max_logodds, min(max_logodds, new_val))
            self._state[g] = new_val

        return dict(self._state)

    def get_intensities(self):
        """Map log-odds state to intensity [0, 1] via sigmoid."""
        result = {}
        for g, lo in self._state.items():
            # Clamp to avoid overflow in exp
            lo_clamped = max(-10.0, min(10.0, lo))
            result[g] = 1.0 / (1.0 + np.exp(-lo_clamped))
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

    def distribute(self, intensities, pos, xcom_hz, total_mass,
                   floor_height, opts):
        """Distribute force based on contact intensities.

        Args:
            intensities: Dict[gname, float] — 0..1 intensity per group
            pos: (J, 3) joint positions
            xcom_hz: (2,) XCoM horizontal position
            total_mass: float
            floor_height: float
            opts: LogOddsContactOptions

        Returns:
            np.ndarray: (J,) pressure per joint in kg
        """
        up = opts.up_axis
        plane = [0, 2] if up == 1 else [0, 1]
        J = pos.shape[0]
        pressure = np.zeros(J)

        # Foot groups: intensity-weighted force sharing
        foot_intensities = {}
        for g in FOOT_GROUPS:
            intensity = intensities.get(g, 0.0)
            if intensity > opts.intensity_deadzone:
                foot_intensities[g] = intensity

        if not foot_intensities:
            return pressure

        # Intensity-weighted XCoM lever rule between feet
        total_intensity = sum(foot_intensities.values())
        group_forces = {}

        if len(foot_intensities) == 1:
            g = list(foot_intensities.keys())[0]
            group_forces[g] = total_mass * foot_intensities[g]
        else:
            # XCoM-based lever rule, scaled by intensity
            centroids = {}
            for gname in foot_intensities:
                joints = FOOT_GROUPS[gname]
                valid = [j for j in joints if j < J]
                if valid:
                    centroids[gname] = np.mean(
                        [pos[j][plane] for j in valid], axis=0)

            if len(centroids) >= 2:
                inv_dists = {}
                for g, c in centroids.items():
                    d = max(0.01, np.linalg.norm(c - xcom_hz))
                    # Weight by both proximity and intensity
                    inv_dists[g] = foot_intensities[g] / d
                total_inv = sum(inv_dists.values())
                for g in foot_intensities:
                    group_forces[g] = (total_mass
                                       * inv_dists.get(g, 0)
                                       / max(total_inv, 1e-6))
            else:
                for g in foot_intensities:
                    w = foot_intensities[g] / total_intensity
                    group_forces[g] = total_mass * w

        # Within-group ball-heel split (same lever rule as unified)
        for gname in foot_intensities:
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
                t = float(np.clip(t, 0.0, 1.0))
                xcom_weights = {ball_j: t, heel_j: 1.0 - t}
            else:
                xcom_weights = {ball_j: 0.5, heel_j: 0.5}

            # Height-based weights: the joint closer to the floor
            # gets more weight. This prevents heel getting force when
            # it's raised and only the ball is on the ground.
            ball_h = max(0.0, pos[ball_j, up] - floor_height)
            heel_h = max(0.0, pos[heel_j, up] - floor_height)

            # SMPL structural compensation: the virtual heel sits ~2cm
            # above the ball even when flat-footed. Without this offset,
            # ball always dominates proximity weighting. We measure heel
            # height relative to ball height so that when the heel is at
            # its natural flat-foot position, it's treated as grounded.
            heel_h_relative = max(0.0, heel_h - ball_h)

            # Convert height to "ground proximity" (0 = high, 1 = on floor)
            ball_prox = max(0.0, 1.0 - ball_h / 0.05)
            heel_prox = max(0.0, 1.0 - heel_h_relative / 0.05)
            prox_sum = ball_prox + heel_prox
            if prox_sum > 0.01:
                height_weights = {ball_j: ball_prox / prox_sum,
                                  heel_j: heel_prox / prox_sum}
            else:
                height_weights = {ball_j: 0.5, heel_j: 0.5}

            # Blend XCoM and height weights (height has priority
            # when there's a clear difference)
            height_diff = abs(ball_h - heel_h)
            height_blend = min(1.0, height_diff / 0.03)  # 0-1 based on height gap
            raw_weights = {}
            for j in valid:
                raw_weights[j] = ((1.0 - height_blend) * xcom_weights.get(j, 0.5)
                                  + height_blend * height_weights.get(j, 0.5))

            # EMA smoothing
            prev = self._split_weights.get(gname, {})
            smooth = {}
            alpha = opts.split_alpha
            for j in valid:
                pw = prev.get(j, 0.0)
                rw = raw_weights.get(j, 0.5)
                smooth[j] = alpha * rw + (1.0 - alpha) * pw

            w_sum = sum(smooth.values())
            if w_sum > 0:
                for j in smooth:
                    smooth[j] /= w_sum

            self._split_weights[gname] = smooth
            for j in valid:
                pressure[j] = gforce * smooth.get(j, 0.5)

        # Hand groups: simple intensity-scaled pressure
        for gname in HAND_GROUPS:
            intensity = intensities.get(gname, 0.0)
            if intensity > opts.intensity_deadzone:
                joints = HAND_GROUPS[gname]
                valid = [j for j in joints if j < J]
                force_per = 5.0 * intensity / max(len(valid), 1)
                for j in valid:
                    pressure[j] = force_per

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
        self.pressure_model = IntensityPressureModel()
        # Legacy streams (kept for A/B testing)
        self.vertical_kinematic_stream = VerticalKinematicStream()
        self.hspeed_stream = HorizontalSpeedStream()
        self.equilibrium_stream = EquilibriumStream()
        self.velocity_stream = VelocityStream()
        self.trajectory_stream = TrajectoryStream()
        self.touchdown_stream = TouchdownStream()

    def _valve_streams(self, increments, context, opts):
        """Adjust stream contributions based on cross-stream context.

        When opts.enable_valving is False, this is a pass-through.
        When True, applies data-driven valving rules to suppress
        misleading stream contributions.

        Args:
            increments: Dict[stream_name, Dict[gname, float]] — raw log-odds
            context: Dict[gname, Dict[str, float]] — per-group context
                     (vy, decel, height_m, hspeed, straightness)
            opts: LogOddsContactOptions

        Returns:
            Dict[stream_name, Dict[gname, float]] — adjusted increments
        """
        if not opts.enable_valving:
            return increments

        # --- Valving rules (to be populated from diagnostic data) ---
        # Deep copy to avoid mutating originals
        valved = {s: dict(d) for s, d in increments.items()}

        # TODO: Add data-driven valving rules here
        # e.g. deceleration suppresses hspeed/trajectory negatives
        # e.g. both-feet-planted suppresses equilibrium negatives

        return valved

    def process_frame(self, pos, com, com_vel, com_acc,
                      floor_height, dt, opts=None):
        """Process one frame.

        Args:
            pos: (J, 3) joint positions
            com: (3,) center of mass
            com_vel: (3,) CoM velocity
            com_acc: (3,) CoM acceleration
            floor_height: float
            dt: float
            opts: LogOddsContactOptions

        Returns:
            LogOddsContactResult
        """
        if opts is None:
            opts = LogOddsContactOptions()

        up = opts.up_axis
        plane = [0, 2] if up == 1 else [0, 1]

        # --- Compute per-stream increments ---
        stream_increments = {}  # stream_name → {gname: float}
        all_group_names = list(ALL_GROUPS.keys())

        if opts.enable_height:
            h_inc = self.height_stream.compute(pos, floor_height, opts)
            stream_increments['height'] = {
                g: v * opts.weight_height for g, v in h_inc.items()
            }

        # Collect context from streams that provide it
        stream_context = {g: {} for g in all_group_names}

        # --- New 3-stream architecture ---
        if opts.enable_kinematic:
            kin_inc, kin_ctx = self.kinematic_stream.compute(
                pos, floor_height, dt, opts)
            stream_increments['kinematic'] = {
                g: v * opts.weight_kinematic for g, v in kin_inc.items()
            }
            for g in all_group_names:
                stream_context[g].update(kin_ctx.get(g, {}))

        if opts.enable_structural:
            # Pass raw_com_acc if available (for freefall detection)
            raw_com_acc = getattr(self, '_raw_com_acc', None)
            struct_inc = self.structural_stream.compute(
                pos, com, com_acc, com_vel, floor_height, dt, opts,
                raw_com_acc=raw_com_acc)
            stream_increments['structural'] = {
                g: v * opts.weight_structural for g, v in struct_inc.items()
            }

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
            curr_intensities = self.accumulator.get_intensities()
            eq_inc = self.equilibrium_stream.compute(
                pos, com, com_vel, com_acc, curr_intensities,
                floor_height, self.total_mass, opts)
            stream_increments['equilibrium'] = {
                g: v * opts.weight_equilibrium for g, v in eq_inc.items()
            }

        # --- Valve step (currently pass-through) ---
        stream_increments = self._valve_streams(
            stream_increments, stream_context, opts)

        # --- Sum increments across streams ---
        total_increments = {g: 0.0 for g in all_group_names}
        for stream_name, inc_dict in stream_increments.items():
            for g in all_group_names:
                total_increments[g] += inc_dict.get(g, 0.0)

        # --- Update accumulator ---
        self.accumulator.update(total_increments, opts.decay_rate, opts.max_logodds)
        intensities = self.accumulator.get_intensities()
        log_odds_state = self.accumulator.state

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
            floor_height, opts)

        return LogOddsContactResult(
            intensity=intensities,
            log_odds_state=log_odds_state,
            per_stream=per_stream,
            stream_context=stream_context,
            pressure_array=pressure,
            contact_state=contact_state,
        )
