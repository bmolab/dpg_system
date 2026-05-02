"""
Patch-based contact frame evaluator.

Treats each broad ground-contact region as a patch with anchor joints and
a convex polygon in the ground plane. The solver allocates per-patch normal
force and a center-of-pressure (CoP) location within each polygon, subject
to whole-body force/moment balance.

Phase 1 scope:
- Closed-form solves for K=1 (single active patch) and K=2 (two active patches)
- Predefined multi-anchor patches for foot (ankle, ball, heel) and hand (wrist, hand)
- Single-anchor degenerate patches for everything else
- Approximate K>=3 fallback (lever-arm regularized) deferred to phase 2

The patch model fixes the flat-foot to ball-of-foot transition by treating
heel-rise as a CoP migration within the foot polygon, not a lift-off event.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class ContactPatch:
    """A region of skin in contact (or candidate for contact) with the floor.

    Polygon vertices are anchor positions projected onto the ground plane,
    in the same order as anchor_joints.
    """
    name: str
    anchor_joints: List[int]
    anchor_positions: np.ndarray   # (N_anchors, 3) world-space anchor positions
    surface_offsets: np.ndarray    # (N_anchors,) joint→skin distance toward floor
    polygon_hz: np.ndarray         # (N_anchors, 2) ground-plane polygon vertices
    plausibility: float            # [0, 1] kinematic+sensory likelihood
    prev_active: bool              # was this patch active last frame?


@dataclass
class PatchForce:
    """Per-patch result from the patch frame evaluator."""
    name: str
    total_force_kg: float          # normal force on the whole patch
    cop_hz: np.ndarray             # (2,) center of pressure in ground plane
    anchor_share: Dict[int, float] # joint_idx -> fraction of total force [0,1]
    necessity: str                 # 'necessary' | 'marginal' | 'unnecessary'
    saturated: bool                # True if CoP was clamped to polygon boundary


@dataclass
class PatchEvalResult:
    """Whole-frame patch evaluation result."""
    per_patch: Dict[str, PatchForce] = field(default_factory=dict)
    f_required: np.ndarray = field(default_factory=lambda: np.zeros(3))
    zmp_hz: np.ndarray = field(default_factory=lambda: np.zeros(2))
    residual: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pruned: Set[str] = field(default_factory=set)
    suggested: Set[str] = field(default_factory=set)


# -----------------------------------------------------------------------------
# Polygon geometry helpers
# -----------------------------------------------------------------------------

def _project_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray
                               ) -> Tuple[np.ndarray, float, float]:
    """Project p onto segment [a,b]. Returns (projected_point, t, distance).

    t in [0,1] is the segment parameter. If p projects outside the segment,
    t is clamped and the returned point is the nearest endpoint.
    """
    d = b - a
    dd = float(np.dot(d, d))
    if dd < 1e-12:
        return a.copy(), 0.0, float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, d) / dd)
    t = max(0.0, min(1.0, t))
    proj = a + t * d
    return proj, t, float(np.linalg.norm(p - proj))


def _polygon_signed_area(polygon: np.ndarray) -> float:
    """Signed area of a polygon via the shoelace formula (2D)."""
    N = polygon.shape[0]
    if N < 3:
        return 0.0
    area2 = 0.0
    for i in range(N):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % N]
        area2 += x1 * y2 - x2 * y1
    return 0.5 * area2


def _point_in_convex_polygon(p: np.ndarray, polygon: np.ndarray) -> bool:
    """Test if p is inside (or on the boundary of) a convex polygon.

    polygon is (N, 2) with vertices in either CW or CCW order. Uses the
    sign-of-cross-product test on consecutive edges. Returns False for
    degenerate (zero-area) polygons so callers fall back to edge-projection.
    """
    N = polygon.shape[0]
    if N < 3:
        return False
    if abs(_polygon_signed_area(polygon)) < 1e-9:
        return False  # Collinear / degenerate — treat as not containing any point
    sign = 0
    for i in range(N):
        a = polygon[i]
        b = polygon[(i + 1) % N]
        edge = b - a
        rel = p - a
        cross = float(edge[0] * rel[1] - edge[1] * rel[0])
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def clamp_to_polygon(p: np.ndarray, polygon: np.ndarray
                     ) -> Tuple[np.ndarray, bool]:
    """Project p onto the convex polygon, returning (clamped_point, was_clamped).

    Polygon may have 1 vertex (point), 2 vertices (segment), or 3+ vertices
    (convex polygon). Returns the nearest point in the polygon to p, and a
    flag indicating whether clamping moved p (i.e., p was outside).
    """
    N = polygon.shape[0]
    if N == 0:
        return p.copy(), False
    if N == 1:
        clamped = polygon[0].copy()
        was_clamped = float(np.linalg.norm(p - clamped)) > 1e-6
        return clamped, was_clamped
    if N == 2:
        proj, _, _ = _project_point_on_segment(p, polygon[0], polygon[1])
        was_clamped = float(np.linalg.norm(p - proj)) > 1e-6
        return proj, was_clamped

    # N >= 3: convex polygon.
    if _point_in_convex_polygon(p, polygon):
        return p.copy(), False

    # Outside — find nearest point on any edge.
    best_pt = polygon[0].copy()
    best_d = float('inf')
    for i in range(N):
        a = polygon[i]
        b = polygon[(i + 1) % N]
        proj, _, d = _project_point_on_segment(p, a, b)
        if d < best_d:
            best_d = d
            best_pt = proj
    return best_pt, True


def _polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    """Centroid (mean of vertices). Sufficient for our small polygons."""
    return np.mean(polygon, axis=0)


def _barycentric_anchor_share(cop: np.ndarray, polygon: np.ndarray
                              ) -> np.ndarray:
    """Compute non-negative anchor weights summing to 1, such that the
    weighted sum of polygon vertices equals (or approximates) cop.

    For N=1, returns [1.0].
    For N=2, returns the segment-parameter split.
    For N=3, returns barycentric coordinates (clamped to non-negative).
    For N>=4, falls back to inverse-distance weighting (sufficient for
    phase 1; phase 2 can swap to Wachspress / mean-value coordinates).
    """
    N = polygon.shape[0]
    if N == 1:
        return np.array([1.0])
    if N == 2:
        d = polygon[1] - polygon[0]
        dd = float(np.dot(d, d))
        if dd < 1e-12:
            return np.array([0.5, 0.5])
        t = float(np.dot(cop - polygon[0], d) / dd)
        t = max(0.0, min(1.0, t))
        return np.array([1.0 - t, t])
    if N == 3:
        # Standard triangle barycentric.
        a, b, c = polygon[0], polygon[1], polygon[2]
        v0 = b - a
        v1 = c - a
        v2 = cop - a
        d00 = float(np.dot(v0, v0))
        d01 = float(np.dot(v0, v1))
        d11 = float(np.dot(v1, v1))
        d20 = float(np.dot(v2, v0))
        d21 = float(np.dot(v2, v1))
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            return np.ones(3) / 3.0
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        weights = np.array([u, v, w])
        # Clamp negatives (CoP outside triangle) and renormalize.
        weights = np.maximum(weights, 0.0)
        s = float(np.sum(weights))
        if s < 1e-12:
            return np.ones(3) / 3.0
        return weights / s
    # N >= 4: inverse distance to each vertex.
    dists = np.array([max(1e-4, float(np.linalg.norm(cop - v))) for v in polygon])
    inv = 1.0 / dists
    return inv / float(np.sum(inv))


# -----------------------------------------------------------------------------
# Patch frame evaluator
# -----------------------------------------------------------------------------

class PatchFrameEvaluator:
    """Patch-granularity contact solver.

    Consumes a list of ContactPatch candidates and the body's required GRF /
    ZMP, produces per-patch (force, CoP, anchor_share). Phase 1 implements
    closed-form K=1 and K=2; K>=3 uses a regularized fallback.
    """

    NECESSARY_THRESHOLD_KG = 2.0
    MARGINAL_THRESHOLD_KG = 0.5
    SATURATION_EPSILON = 0.005  # 5mm — CoP within this of polygon edge counts as saturated
    PLAUSIBILITY_THRESHOLD = 0.10  # patches below this score are not considered active candidates
    HYSTERESIS_BOOST = 0.40  # additive boost for prev-active patches when scoring
    MULTI_ANCHOR_BOOST = 0.05  # small bias toward multi-anchor (foot/hand) patches over noisy single-anchor ones
    # --- Recruitment-by-residual (K=1 -> K=2) ---
    # Trigger recruitment when K=1 leaves a moment residual the body can't
    # produce alone. Threshold is fraction of body weight (in kg·m units),
    # which is small enough to fire for any reasonable stance width.
    RECRUIT_RESIDUAL_FRAC = 0.04  # 3 kg·m for a 75kg body; ~4cm lever
    RECRUIT_IMPROVEMENT_RATIO = 0.50  # accept K=2 only if residual drops below 50% of K=1

    def __init__(self, total_mass: float):
        self.total_mass = total_mass

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate_patches(self,
                         candidate_patches: List[ContactPatch],
                         com: np.ndarray,
                         com_acc: np.ndarray,
                         up_axis: int = 1,
                         prev_result: Optional[PatchEvalResult] = None,
                         ) -> PatchEvalResult:
        """Solve per-patch force and CoP for one frame."""

        plane_dims = [0, 2] if up_axis == 1 else [0, 1]
        g_mag = 9.81

        # --- Required GRF and ZMP ---
        g_vec = np.zeros(3)
        g_vec[up_axis] = -g_mag
        f_required = self.total_mass * (com_acc - g_vec)
        f_total_kg = max(0.0, float(f_required[up_axis]) / g_mag)

        zmp_hz = self._compute_zmp(com, com_acc, up_axis, plane_dims, g_mag)

        # --- Free-fall short-circuit ---
        if f_total_kg < 0.1 or not candidate_patches:
            return PatchEvalResult(
                per_patch={},
                f_required=f_required,
                zmp_hz=zmp_hz,
                residual=f_required.copy(),
                pruned=set(),
                suggested=set(),
            )

        # --- Score and select active patches ---
        # Score = plausibility + small bias toward multi-anchor patches + hysteresis
        # for prev-active. The active-set filter uses score (not raw plausibility),
        # so prev-active patches stay in the set even when consensus dips briefly.
        scored = []
        for p in candidate_patches:
            score = p.plausibility
            if len(p.anchor_joints) >= 2:
                score += self.MULTI_ANCHOR_BOOST
            if p.prev_active and p.plausibility >= 0.05:
                score += self.HYSTERESIS_BOOST
            scored.append((score, p))
        scored.sort(key=lambda x: -x[0])

        active = [p for s, p in scored if s >= self.PLAUSIBILITY_THRESHOLD]
        # Always keep at least one patch in the active set if any candidate exists.
        if not active and scored:
            active = [scored[0][1]]

        # --- Initial solve based on K ---
        K = len(active)
        if K == 1:
            patch_results = self._solve_k1(active[0], zmp_hz, f_total_kg)
        elif K == 2:
            patch_results = self._solve_k2(active, zmp_hz, f_total_kg)
        else:
            patch_results = self._solve_kn_fallback(active, zmp_hz, f_total_kg)

        # --- Recruitment by residual (K=1 -> K=2) ---
        # If a single-patch solve leaves a moment residual the body cannot
        # produce alone, recruit the next-best inactive candidate and try K=2.
        # This is what brings RF back into a squat when consensus has dropped
        # it: physics demands two contacts, so we add one even if its
        # plausibility is below the active-set threshold.
        if K == 1 and f_total_kg > 0.5:
            res_k1 = self._moment_residual_norm(patch_results, zmp_hz, f_total_kg)
            recruit_threshold = self.RECRUIT_RESIDUAL_FRAC * f_total_kg
            if res_k1 > recruit_threshold:
                inactive = [(s, p) for s, p in scored
                            if p.name not in patch_results]
                if inactive:
                    candidate = inactive[0][1]
                    trial_active = [active[0], candidate]
                    trial_results = self._solve_k2(trial_active, zmp_hz, f_total_kg)
                    res_k2 = self._moment_residual_norm(
                        trial_results, zmp_hz, f_total_kg
                    )
                    if res_k2 < res_k1 * self.RECRUIT_IMPROVEMENT_RATIO:
                        patch_results = trial_results
                        active = trial_active
                        K = 2

        # --- Compute residual ---
        applied_force = sum(pr.total_force_kg for pr in patch_results.values())
        applied_moment = np.zeros(2)
        for pr in patch_results.values():
            applied_moment += pr.total_force_kg * pr.cop_hz
        target_moment = f_total_kg * zmp_hz
        moment_residual = target_moment - applied_moment
        force_residual_up = f_total_kg - applied_force
        residual = np.zeros(3)
        residual[up_axis] = force_residual_up * g_mag  # back to Newtons-equivalent
        residual[plane_dims[0]] = moment_residual[0] * g_mag
        residual[plane_dims[1]] = moment_residual[1] * g_mag

        # --- Necessity classification ---
        for name, pr in patch_results.items():
            if pr.total_force_kg > self.NECESSARY_THRESHOLD_KG:
                pr.necessity = 'necessary'
            elif pr.total_force_kg > self.MARGINAL_THRESHOLD_KG:
                pr.necessity = 'marginal'
            else:
                pr.necessity = 'unnecessary'

        # --- Suggested patches (recruitment hint) ---
        suggested = set()
        residual_kg = float(np.linalg.norm(moment_residual))
        any_saturated = any(pr.saturated for pr in patch_results.values())
        if any_saturated and residual_kg > 0.05 * self.total_mass:
            # Look for the highest-plausibility inactive candidate.
            inactive = [p for s, p in scored
                        if p.name not in patch_results and p.plausibility > 0.0]
            if inactive:
                suggested.add(inactive[0].name)

        # --- Pruned: any active patch whose force ended up below marginal ---
        pruned = {name for name, pr in patch_results.items()
                  if pr.necessity == 'unnecessary'}

        return PatchEvalResult(
            per_patch=patch_results,
            f_required=f_required,
            zmp_hz=zmp_hz,
            residual=residual,
            pruned=pruned,
            suggested=suggested,
        )

    # ------------------------------------------------------------------
    # ZMP from CoM dynamics (mirrors DynamicFrameEvaluator)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_zmp(com: np.ndarray, com_acc: np.ndarray,
                     up_axis: int, plane_dims, g_mag: float) -> np.ndarray:
        com_hz = com[plane_dims]
        h_com = max(0.0, float(com[up_axis]))
        a_hz = com_acc[plane_dims]
        a_vert = float(com_acc[up_axis]) + g_mag
        if a_vert > 2.0 and h_com > 0.01:
            disp = (h_com / a_vert) * a_hz
            disp_mag = float(np.linalg.norm(disp))
            if disp_mag > 2.0:
                disp = disp * (2.0 / disp_mag)
            return com_hz - disp
        return com_hz.copy()

    # ------------------------------------------------------------------
    # Closed-form solvers
    # ------------------------------------------------------------------

    def _solve_k1(self, patch: ContactPatch, zmp_hz: np.ndarray,
                  f_total_kg: float) -> Dict[str, PatchForce]:
        """K=1: single active patch carries all force; CoP = clamp(ZMP, polygon)."""
        cop, was_clamped = clamp_to_polygon(zmp_hz, patch.polygon_hz)
        share = _barycentric_anchor_share(cop, patch.polygon_hz)
        anchor_share = {j: float(share[i]) for i, j in enumerate(patch.anchor_joints)}
        saturated = was_clamped or self._cop_on_boundary(cop, patch.polygon_hz)
        return {
            patch.name: PatchForce(
                name=patch.name,
                total_force_kg=float(f_total_kg),
                cop_hz=cop,
                anchor_share=anchor_share,
                necessity='necessary',
                saturated=saturated,
            )
        }

    def _solve_k2(self, patches: List[ContactPatch], zmp_hz: np.ndarray,
                  f_total_kg: float) -> Dict[str, PatchForce]:
        """K=2: lever-arm split along inter-centroid line, CoP = clamp(ZMP, polygon).

        This is approximate: CoP for each patch is independently clamped to its
        polygon, and force is allocated by ZMP's projection onto the centroid
        line. The moment-balance residual surfaces in the result.residual; large
        residual + saturated polygon edges trigger recruitment in the caller.
        """
        p1, p2 = patches[0], patches[1]
        c1 = _polygon_centroid(p1.polygon_hz)
        c2 = _polygon_centroid(p2.polygon_hz)
        d = c2 - c1
        d_len = float(np.linalg.norm(d))

        if d_len < 1e-4:
            # Patches are stacked — fall back to equal split.
            f1 = f2 = 0.5 * f_total_kg
        else:
            t = float(np.dot(zmp_hz - c1, d) / (d_len * d_len))
            t = max(0.0, min(1.0, t))
            f1 = (1.0 - t) * f_total_kg
            f2 = t * f_total_kg

        results: Dict[str, PatchForce] = {}
        for patch, force in [(p1, f1), (p2, f2)]:
            cop, was_clamped = clamp_to_polygon(zmp_hz, patch.polygon_hz)
            share = _barycentric_anchor_share(cop, patch.polygon_hz)
            anchor_share = {j: float(share[i]) for i, j in enumerate(patch.anchor_joints)}
            saturated = was_clamped or self._cop_on_boundary(cop, patch.polygon_hz)
            results[patch.name] = PatchForce(
                name=patch.name,
                total_force_kg=float(force),
                cop_hz=cop,
                anchor_share=anchor_share,
                necessity='necessary',
                saturated=saturated,
            )
        return results

    def _solve_kn_fallback(self, patches: List[ContactPatch],
                           zmp_hz: np.ndarray, f_total_kg: float
                           ) -> Dict[str, PatchForce]:
        """K>=3 phase-1 fallback: inverse-distance weighting of ZMP→centroid.

        Each patch independently clamps ZMP to its polygon for CoP. Force is
        distributed by inverse distance from ZMP to each polygon centroid.
        Phase 2 will replace this with a proper QP. This is a placeholder
        sufficient for prone/crawling test cases without crashing.
        """
        centroids = np.array([_polygon_centroid(p.polygon_hz) for p in patches])
        dists = np.array([max(0.01, float(np.linalg.norm(c - zmp_hz)))
                          for c in centroids])
        inv = 1.0 / dists
        fractions = inv / float(np.sum(inv))

        results: Dict[str, PatchForce] = {}
        for i, patch in enumerate(patches):
            force = float(fractions[i] * f_total_kg)
            cop, was_clamped = clamp_to_polygon(zmp_hz, patch.polygon_hz)
            share = _barycentric_anchor_share(cop, patch.polygon_hz)
            anchor_share = {j: float(share[i2]) for i2, j in enumerate(patch.anchor_joints)}
            saturated = was_clamped or self._cop_on_boundary(cop, patch.polygon_hz)
            results[patch.name] = PatchForce(
                name=patch.name,
                total_force_kg=force,
                cop_hz=cop,
                anchor_share=anchor_share,
                necessity='necessary',
                saturated=saturated,
            )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _moment_residual_norm(patch_results: Dict[str, PatchForce],
                              zmp_hz: np.ndarray, f_total_kg: float) -> float:
        """Norm of the unbalanced moment in kg·m: ||F_total·Z - Σ f_k·c_k||."""
        applied_moment = np.zeros(2)
        for pr in patch_results.values():
            applied_moment = applied_moment + pr.total_force_kg * pr.cop_hz
        target_moment = f_total_kg * zmp_hz
        return float(np.linalg.norm(target_moment - applied_moment))

    def _cop_on_boundary(self, cop: np.ndarray, polygon: np.ndarray) -> bool:
        """Return True if cop lies within SATURATION_EPSILON of any polygon edge."""
        N = polygon.shape[0]
        if N == 1:
            return True  # single-anchor patch: CoP is always "at the boundary"
        if N == 2:
            _, t, _ = _project_point_on_segment(cop, polygon[0], polygon[1])
            return t <= 1e-3 or t >= 1.0 - 1e-3
        for i in range(N):
            a = polygon[i]
            b = polygon[(i + 1) % N]
            _, _, d = _project_point_on_segment(cop, a, b)
            if d < self.SATURATION_EPSILON:
                return True
        return False
