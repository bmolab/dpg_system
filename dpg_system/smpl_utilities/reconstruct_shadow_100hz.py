"""Reconstruct a uniform 100 Hz SMPL stream from a Shadow IMU file sampled at 60 Hz.

Shadow IMU captures are originally 100 Hz, downsampled by the recording pipeline
to 60 Hz. The downsampling uses nearest-neighbour pulldown so every 3 output
frames span 5 input frames, with input-time gaps alternating (2, 2, 1). This
produces the 3-frame cadence visible in pose-delta sequences: two LONG steps
followed by one SHORT step at half the magnitude, repeating.

The cadence is a periodic artifact that:
  - confuses downstream noise detectors (cadence variation looks like spike
    activity, fragmenting clean segments and inflating noise scores)
  - misrepresents motion velocity (LONG steps double-count time vs SHORT)
  - prevents direct comparison with native 100 Hz capture data

This tool reconstructs the underlying 100 Hz timeline by:
  1. Tracking the cadence phase across the whole file via a Viterbi HMM (3 phase
     states with sticky transitions). Cadence is stable for tens of seconds at
     a time but shifts ~62 times across a typical 17-min Shadow file due to
     network drops/duplications upstream of the resampler.
  2. Computing the implied input-time index for each output frame (cumulative
     from the per-step gap pattern).
  3. For each joint, fitting a quintic-Hermite RotationSpline in SO(3) space
     over the whole file. Evaluating each spline at the uniform 100 Hz output
     grid gives the reconstructed pose stream.

Why rotation-space interpolation rather than axis-angle vector interpolation:
Axis-angle is not a faithful representation of SO(3) — the same rotation can
have multiple axis-angle vectors, and the SMPL solver occasionally flips
between equivalent representations during fast motion (especially wrists in
dynamic dance styles). Cubic spline in axis-angle vector space then overshoots
across these representation discontinuities. scipy's RotationSpline interpolates
on the unit quaternion sphere directly, which has correct topology and
naturally handles wrap-around.

Empirically on Subject2_take3 (Contemporary):
  - noise_score: 2.51 (orig) -> 1.14 (recon) - 55% reduction
  - spike rate per 1000 fr: 9.24 -> 1.10 - 8.4x reduction
  - clean fraction: 95.0% -> 97.8%
  - stream_breaks: 5 -> 1
On Subject1_take6 (Waacking, fast wrist motion), prior axis-angle cubic spline
produced 61° overshoots at SMPL solver representation flips; RotationSpline
produces smooth rotations everywhere.

Examples:
    python reconstruct_shadow_100hz.py path/to/take.npz
    python reconstruct_shadow_100hz.py path/to/directory --recursive
    python reconstruct_shadow_100hz.py take.npz --output custom_name.npz
"""

import argparse
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline


# ----------------------------------------------------------------------
# Viterbi cadence phase tracking
# ----------------------------------------------------------------------

CADENCE_PERIOD = 3
VITERBI_WINDOW = 90
VITERBI_STEP = 15
PRIOR_STAY = 0.995
TEMPERATURE = 6.0
TRUNK_JOINTS = [0, 3, 6, 9]   # pelvis + spine1/2/3 - clean cadence proxy


def _window_medians(deltas, start, end, period):
    if end - start < period * 3:
        return None
    return np.array([
        float(np.median(deltas[np.arange(start + ((ph - start) % period), end, period)]))
        for ph in range(period)
    ])


def _viterbi_phase_path(delta_signal, period=CADENCE_PERIOD):
    """Return (observation_centers, phase_path) over windowed cadence observations.

    The phase_path tracks which residue class (mod period) corresponds to the
    SHORT-step phase in each window via a sticky HMM. Returned indices are
    output-frame indices identifying the center of each window.
    """
    N = len(delta_signal)
    prior_shift = (1 - PRIOR_STAY) / (period - 1)

    centers, medians = [], []
    for s in range(0, N - VITERBI_WINDOW, VITERBI_STEP):
        m = _window_medians(delta_signal, s, s + VITERBI_WINDOW, period)
        if m is None:
            continue
        centers.append(s + VITERBI_WINDOW // 2)
        medians.append(m)
    medians = np.array(medians)
    n_obs = medians.shape[0]
    if n_obs == 0:
        raise RuntimeError('Too few frames for cadence detection')

    emiss = np.zeros_like(medians)
    for i in range(n_obs):
        m = medians[i]
        spread = max(m.max() - m.min(), 1e-9)
        emiss[i] = -(m - m.min()) / spread * TEMPERATURE

    trans = np.full((period, period), np.log(prior_shift))
    np.fill_diagonal(trans, np.log(PRIOR_STAY))

    vit = np.full((n_obs, period), -np.inf)
    back = np.zeros((n_obs, period), dtype=int)
    vit[0] = emiss[0]
    for t in range(1, n_obs):
        for j in range(period):
            scores = vit[t - 1] + trans[:, j]
            b = int(np.argmax(scores))
            back[t, j] = b
            vit[t, j] = scores[b] + emiss[t, j]
    path = np.zeros(n_obs, dtype=int)
    path[-1] = int(np.argmax(vit[-1]))
    for t in range(n_obs - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return np.array(centers), path


# Note on Viterbi gap-assignment limitations: in regions where the cadence
# locally shifts (e.g., around a single-frame skip not large enough to be
# classified as a real discontinuity), Viterbi can mis-assign a LONG step as
# SHORT, leaving a "big unsplit step" in the output. We attempted post-process
# correction (promote outlier SHORTs to LONG) but found it impossible to
# discriminate missed-LONGs from naturally-large SHORTs in fast-motion regions
# using magnitude alone — the over-promotion rate (~10%) introduced 2%+
# timeline stretching. Left as known limitation; occurrence rate is low
# (1 case found in 14 files during validation).


LS_SWAP_RATIO = 1.5          # SHORT body_pose_d > LONG (adjacent) body_pose_d * this
LS_SWAP_POSE_FLOOR = 0.05    # SHORT body_pose_d (rad-sum across 22 body joints) must exceed this
# We use full-body pose delta (sum across all 22 body joints, including arms,
# which dominate during dance) rather than trans-jump alone. Trans is fooled
# by translation-only motion (e.g., shifting body weight without rotation);
# pose-delta correctly captures real body motion. See project memory for the
# tuning history.


def _compute_actual_gaps(delta_trunk, period=CADENCE_PERIOD, body_pose_d=None):
    """Return the input-unit gap for each output step.

    Normal cadence: SHORT step = 1 input-unit, LONG step = 2 input-units.
    At a phase shift (slip), the step's gap is larger:
        delta-phase +2 mod period: 1 extra input-unit (one dropped input frame)
        delta-phase +1 mod period: 2 extra input-units (two dropped frames -
            no duplicates exist in Shadow files; this is the most consistent
            interpretation)

    Post-process LOCAL L<->S SWAP: a dropped 100 Hz input frame within a
    phase-stable segment can manifest as 3 consecutive LONG-sized motion
    bursts where Viterbi places one of them at the SHORT cadence position.
    Detect via full-body pose-delta sum (sum across 22 body joints): if a
    SHORT-then-LONG (or LONG-then-SHORT) pair has SHORT's body_pose_d >>
    LONG's, swap them so the genuinely-large step gets split.
    """
    N = len(delta_trunk)
    centers, path = _viterbi_phase_path(delta_trunk, period)

    step_phase = path[np.array([
        int(np.argmin(np.abs(centers - i))) for i in range(N)
    ])]
    is_short = (np.arange(N) % period) == step_phase
    expected_gap = np.where(is_short, 1, 2)

    phase_diff = np.zeros(N, dtype=int)
    for i in range(1, N):
        if step_phase[i] != step_phase[i - 1]:
            dph = (step_phase[i] - step_phase[i - 1]) % period
            phase_diff[i] = 1 if dph == 2 else 2

    actual_gap = expected_gap + phase_diff

    # Local L<->S swap correction using full-body pose-delta.
    # Only fires above an absolute pose floor to avoid noise swaps in still
    # sections. Mark swaps in a single pass, then apply non-cascading.
    if body_pose_d is not None and len(body_pose_d) == N:
        swaps = []   # list of (i, j) pairs where gap[i] and gap[j] swap
        for i in range(N - 1):
            j = i + 1
            # SHORT-then-LONG: gap[i]=1, gap[j]=2
            if actual_gap[i] == 1 and actual_gap[j] == 2:
                short_p, long_p = body_pose_d[i], body_pose_d[j]
            # LONG-then-SHORT: gap[i]=2, gap[j]=1
            elif actual_gap[i] == 2 and actual_gap[j] == 1:
                short_p, long_p = body_pose_d[j], body_pose_d[i]
            else:
                continue
            if short_p < LS_SWAP_POSE_FLOOR or long_p < 1e-9:
                continue
            if short_p >= LS_SWAP_RATIO * long_p:
                swaps.append((i, j))
        # Apply swaps, but skip any swap that overlaps with a previous swap
        # already applied in this pass (prevents cascading)
        used = set()
        for i, j in swaps:
            if i in used or j in used:
                continue
            actual_gap[i], actual_gap[j] = actual_gap[j], actual_gap[i]
            used.add(i); used.add(j)

    return actual_gap


# ----------------------------------------------------------------------
# Axis-angle continuity (post-spline unwrap)
# ----------------------------------------------------------------------

_TWO_PI = 2.0 * np.pi


def _unwrap_rotvec_continuous(rotvecs):
    """Walk a single joint's axis-angle sequence and choose the representation
    that's closest to the previous frame at each step.

    Every rotation has two axis-angle representations: v (canonical, |v|<=pi)
    and v' = v * (1 - 2*pi/|v|) (the "long way around", |v'| = 2*pi - |v|).
    Picking the v that minimizes the jump from the previous frame keeps the
    axis-angle trajectory continuous, even when the rotation passes through
    the canonical boundary.

    Args:
        rotvecs: (T, 3) per-frame axis-angle for one joint

    Returns:
        (T, 3) unwrapped axis-angle sequence
    """
    out = rotvecs.copy()
    for i in range(1, len(out)):
        v = out[i]
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            continue
        v_alt = v * (1.0 - _TWO_PI / norm)
        prev = out[i - 1]
        if np.sum((v_alt - prev) ** 2) < np.sum((v - prev) ** 2):
            out[i] = v_alt
    return out


# ----------------------------------------------------------------------
# Discontinuity detection (real trans/pose shifts, NOT cadence skips)
# ----------------------------------------------------------------------

# A trans jump is a "real shift" if its position lands well off the trajectory
# extrapolated from preceding frames. Thresholds chosen to match the events
# the noise detector itself flags as stream_breaks (it uses 20x median; we
# require both a magnitude bump and a strongly-off-trajectory residual).
TRANS_DISCONT_MAGNITUDE_MIN = 5.0   # step must be >= this multiple of local median
TRANS_DISCONT_ONNESS_MIN = 5.0      # residual / scale to count as off-trajectory
TRANS_DISCONT_LOCAL_FLOOR = 0.0005  # 0.5mm/frame; below this, skip detection
TRANS_DISCONT_WINDOW = 12           # frames on each side for local stats
TRANS_DISCONT_CLUSTER_GAP = 5       # collapse adjacent detections within this many frames


def detect_trans_discontinuities(trans):
    """Return sorted list of original-frame indices `d` where the step from
    frame d-1 to frame d is a real translation discontinuity (sensor reference
    shift), as opposed to a normal cadence-driven LONG step or fast motion.

    Discriminator (two-stage):
      Stage 1 ("off-trajectory"): extrapolate quadratically from frames d-3,
        d-2, d-1 and compare to the actual position at d.  If the residual
        is well above the local typical step magnitude, the position is
        off-trajectory.
      Stage 2 ("persistent offset"): a real teleport produces a CONSTANT
        position offset that persists in post-jump frames - the body keeps
        whatever motion it had, but starting from the shifted location.
        A motion-onset (fast acceleration from still) has GROWING residual
        as the body continues to move away from the extrapolated still
        trajectory.  We require the residual to remain large but stable
        across 3+ post-jump frames.

    Adjacent detections (within TRANS_DISCONT_CLUSTER_GAP frames) are collapsed
    to a single event at the highest-onness position.
    """
    T = trans.shape[0]
    if T < 8:
        return []
    trans_disp = np.linalg.norm(np.diff(trans, axis=0), axis=1)   # (T-1,)
    raw = []   # list of (frame_idx, onness)
    for i in range(3, T - 4):       # need at least 3 post-jump frames
        step = trans_disp[i - 1]
        lo = max(0, i - TRANS_DISCONT_WINDOW - 1)
        hi = min(len(trans_disp), i - 1 + TRANS_DISCONT_WINDOW + 1)
        local = np.concatenate([trans_disp[lo:i - 1], trans_disp[i:hi]])
        if len(local) < 4:
            continue
        local_med = float(np.median(local))
        if local_med < TRANS_DISCONT_LOCAL_FLOOR:
            continue
        if step / local_med < TRANS_DISCONT_MAGNITUDE_MIN:
            continue

        # Stage 1: off-trajectory test at frame i
        fit_t = np.array([i - 3, i - 2, i - 1], dtype=float)
        fit_p = trans[i - 3:i]
        coeffs = [np.polyfit(fit_t, fit_p[:, c], 2) for c in range(3)]
        def predict(t):
            return np.array([np.polyval(coeffs[c], t) for c in range(3)])
        pred_i = predict(float(i))
        residual_i = float(np.linalg.norm(trans[i] - pred_i))
        onness_i = residual_i / max(local_med, 1e-9)
        if onness_i < TRANS_DISCONT_ONNESS_MIN:
            continue

        # Stage 2: persistence test - residuals at i+1, i+2, i+3 should remain
        # roughly the same magnitude as at i (constant offset = real teleport).
        # For motion-onset, the residual grows as motion continues.
        residuals = [residual_i]
        for k in (1, 2, 3):
            if i + k >= T:
                break
            pred_k = predict(float(i + k))
            residuals.append(float(np.linalg.norm(trans[i + k] - pred_k)))
        if len(residuals) < 4:
            continue
        # If the residual at frame i+3 is much larger than at frame i, the
        # body kept moving away from the extrapolated trajectory: motion onset,
        # not a teleport.
        late_to_initial = residuals[-1] / max(residuals[0], 1e-9)
        if late_to_initial > 1.6:
            continue   # motion-onset, not a real shift
        raw.append((i, onness_i))

    if not raw:
        return []
    # Cluster: keep best-onness frame within each connected run
    raw.sort(key=lambda x: x[0])
    clusters = [[raw[0]]]
    for entry in raw[1:]:
        if entry[0] - clusters[-1][-1][0] <= TRANS_DISCONT_CLUSTER_GAP:
            clusters[-1].append(entry)
        else:
            clusters.append([entry])
    return sorted(max(cluster, key=lambda x: x[1])[0] for cluster in clusters)


# ----------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------

def _reconstruct_segment_100hz(poses, trans):
    """Reconstruct a single phase-stable segment (with no discontinuities) to
    100 Hz uniform timing. Returns (out_poses, out_trans, output_n_frames).

    This is the per-segment workhorse - it does the cadence/Viterbi tracking
    plus per-joint RotationSpline interpolation on that segment alone.

    For segments too small for cadence/Viterbi detection (< window*1.5 frames),
    we skip Viterbi and either copy frames as-is (very small) or assume the
    dominant 2,2,1 cadence pattern with phase=0.
    """
    poses = np.asarray(poses, dtype=np.float64)
    if trans is not None:
        trans = np.asarray(trans, dtype=np.float64)
    T = poses.shape[0]
    n_joints = poses.shape[1] // 3

    # Tiny segments: just emit the original frames (no cadence correction possible)
    if T < 4:
        return (poses.astype(np.float32),
                (trans.astype(np.float32) if trans is not None else None),
                T)

    # Per-segment cadence detection (trunk-only).  Viterbi needs at least one
    # full observation window (VITERBI_WINDOW frames) to fire; for medium-small
    # segments we fall back to assuming the canonical 2,2,1 cadence with phase=0.
    p3 = poses.reshape(T, n_joints, 3)
    body = p3[:, :22, :]
    delta_body = np.linalg.norm(body[1:] - body[:-1], axis=2)
    delta_trunk = delta_body[:, TRUNK_JOINTS].sum(axis=1)
    # Full-body pose delta (sum across 22 body joints, including arms which
    # dominate in dance motion). Used for L<->S swap detection in
    # _compute_actual_gaps. The trunk-only metric is still used for cadence
    # phase tracking (Viterbi) because trunk is immune to limb-flick noise.
    body_pose_d = delta_body.sum(axis=1)
    if T - 1 >= VITERBI_WINDOW + VITERBI_STEP:
        actual_gap = _compute_actual_gaps(delta_trunk, body_pose_d=body_pose_d)
    else:
        # Assume canonical cadence pattern, phase 0 (SHORT every 3rd step at i%3==0)
        n_steps = T - 1
        is_short = (np.arange(n_steps) % CADENCE_PERIOD) == 0
        actual_gap = np.where(is_short, 1, 2)

    # Cumulative input-time per output frame
    input_time = np.zeros(T, dtype=np.float64)
    for i in range(T - 1):
        input_time[i + 1] = input_time[i] + actual_gap[i]

    total_input_units = int(round(input_time[-1]))
    output_times = np.arange(total_input_units + 1, dtype=np.float64)
    n_out = len(output_times)

    out_poses = np.zeros((n_out, n_joints * 3), dtype=np.float32)
    for j in range(n_joints):
        joint_rotvecs = p3[:, j, :]
        rots = Rotation.from_rotvec(joint_rotvecs)
        spline = RotationSpline(input_time, rots)
        interp_rots = spline(output_times)
        canonical_aa = interp_rots.as_rotvec()
        canonical_aa[0] = joint_rotvecs[0]
        unwrapped = _unwrap_rotvec_continuous(canonical_aa)
        out_poses[:, j * 3:(j + 1) * 3] = unwrapped.astype(np.float32)

    out_trans_arr = None
    if trans is not None:
        cs_trans = CubicSpline(input_time, trans, bc_type='natural', extrapolate=False)
        out_trans_arr = cs_trans(output_times).astype(np.float32)

    return out_poses, out_trans_arr, n_out


def reconstruct_100hz(poses, trans, verbose=True):
    """Reconstruct a uniform 100 Hz pose sequence from cadenced 60 Hz input,
    preserving real translation discontinuities (sensor reference shifts).

    Trans-discontinuity detection runs first to identify "real shift" events
    where the body position teleports off-trajectory. The file is then split
    into segments at each detected discontinuity; each segment is reconstructed
    independently and the segments are concatenated. The discontinuity is
    preserved as a sharp jump between adjacent output frames.

    Args:
        poses: (T, 156) axis-angle SMPL-H poses (52 joints, 3 components each)
        trans: (T, 3) or None - body translation (cubic spline in R^3)
        verbose: print progress and stats

    Returns:
        out_poses: (T_new, 156) reconstructed poses
        out_trans: (T_new, 3) or None - reconstructed translations
        stats: dict with reconstruction statistics
    """
    poses = np.asarray(poses, dtype=np.float64)
    if trans is not None:
        trans = np.asarray(trans, dtype=np.float64)
    T = poses.shape[0]

    # Detect real translation discontinuities (off-trajectory jumps)
    discontinuities = []
    if trans is not None:
        discontinuities = detect_trans_discontinuities(trans)
    if verbose:
        print(f'  Detected {len(discontinuities)} translation discontinuities '
              f'(real shifts preserved as sharp jumps): {discontinuities}', flush=True)

    # Build segment boundaries: [0, d1, d2, ..., T]
    boundaries = [0] + discontinuities + [T]

    # Reconstruct each segment independently, concatenate
    poses_parts = []
    trans_parts = []
    total_inserted = 0
    total_original = 0
    for k in range(len(boundaries) - 1):
        seg_lo, seg_hi = boundaries[k], boundaries[k + 1]
        if seg_hi - seg_lo <= 0:
            continue
        seg_poses = poses[seg_lo:seg_hi]
        seg_trans = trans[seg_lo:seg_hi] if trans is not None else None
        seg_out_poses, seg_out_trans, seg_n_out = _reconstruct_segment_100hz(seg_poses, seg_trans)
        poses_parts.append(seg_out_poses)
        if seg_out_trans is not None:
            trans_parts.append(seg_out_trans)
        seg_inserted = seg_n_out - (seg_hi - seg_lo)
        total_inserted += max(0, seg_inserted)
        total_original += seg_hi - seg_lo
        if verbose and len(discontinuities) > 0:
            print(f'    segment {k}: orig frames [{seg_lo}, {seg_hi}) -> {seg_n_out} output frames', flush=True)

    out_poses = np.concatenate(poses_parts, axis=0)
    out_trans_arr = np.concatenate(trans_parts, axis=0) if trans_parts else None

    if verbose:
        print(f'  Output frames: {len(out_poses)} (target ~{int(T * 5 / 3)})', flush=True)

    stats = {
        'original_n_frames': T,
        'output_n_frames': len(out_poses),
        'inserted_frames': total_inserted,
        'n_discontinuities': len(discontinuities),
        'discontinuity_frames': discontinuities,
    }
    return out_poses, out_trans_arr, stats


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------

def reconstruct_file(input_path, output_path=None, verbose=True):
    """Read a Shadow npz, reconstruct to 100 Hz, write output."""
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        if base.endswith('_100hz'):
            if verbose:
                print(f'  Skipping (already _100hz): {input_path}', flush=True)
            return None
        output_path = base + '_100hz' + ext

    if verbose:
        print(f'Reading: {input_path}', flush=True)
    with np.load(input_path, allow_pickle=True) as src:
        poses = src['poses']
        trans = src['trans'] if 'trans' in src.files else None
        src_fps = float(src['mocap_framerate']) if 'mocap_framerate' in src.files else 60.0
        # Preserve all other arrays
        extras = {k: src[k] for k in src.files
                  if k not in ('poses', 'trans', 'mocap_framerate')}

    if verbose:
        dur = poses.shape[0] / src_fps
        print(f'  {poses.shape[0]} frames @ {src_fps} fps ({dur:.1f}s)', flush=True)

    out_poses, out_trans, stats = reconstruct_100hz(poses, trans, verbose=verbose)

    save_dict = {
        'poses': out_poses,
        'mocap_framerate': np.array(100.0, dtype=np.float64),
        'reconstructed_from': np.array(os.path.abspath(input_path)),
        'reconstruction_method': np.array(
            'viterbi-cadence + RotationSpline (SO(3)) + discontinuity-preserving segmentation'),
        'original_n_frames': np.array(stats['original_n_frames']),
        'inserted_frames': np.array(stats['inserted_frames']),
        'n_discontinuities': np.array(stats.get('n_discontinuities', 0)),
        'discontinuity_frames': np.array(stats.get('discontinuity_frames', []), dtype=np.int64),
    }
    if out_trans is not None:
        save_dict['trans'] = out_trans
    save_dict.update(extras)

    np.savez(output_path, **save_dict)
    if verbose:
        out_dur = stats['output_n_frames'] / 100.0
        print(f'  Wrote: {output_path}', flush=True)
        print(f'    output: {stats["output_n_frames"]} frames @ 100 Hz ({out_dur:.1f}s)', flush=True)
        print(f'    inserted: {stats["inserted_frames"]}', flush=True)
    return output_path


def main():
    ap = argparse.ArgumentParser(
        description='Reconstruct 100 Hz SMPL stream from cadenced 60 Hz Shadow file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('inputs', nargs='+',
                    help='npz file(s) or directory(ies) to process')
    ap.add_argument('--output', '-o',
                    help='Explicit output path (only valid for a single input file)')
    ap.add_argument('--recursive', '-r', action='store_true',
                    help='Walk directories recursively')
    ap.add_argument('--quiet', '-q', action='store_true',
                    help='Suppress per-file progress output')
    args = ap.parse_args()

    verbose = not args.quiet

    # Resolve input list
    files = []
    for inp in args.inputs:
        if os.path.isdir(inp):
            if args.recursive:
                for root, _, names in os.walk(inp):
                    for n in sorted(names):
                        if n.endswith('.npz') and not n.endswith('_100hz.npz'):
                            files.append(os.path.join(root, n))
            else:
                for n in sorted(os.listdir(inp)):
                    if n.endswith('.npz') and not n.endswith('_100hz.npz'):
                        files.append(os.path.join(inp, n))
        elif os.path.isfile(inp):
            files.append(inp)
        else:
            print(f'Skipping (not found): {inp}', file=sys.stderr)

    if args.output is not None and len(files) != 1:
        ap.error('--output requires exactly one input file')

    if not files:
        ap.error('No npz files to process')

    if verbose:
        print(f'Processing {len(files)} file(s)', flush=True)

    n_done = 0
    n_skip = 0
    n_err = 0
    for i, fp in enumerate(files, 1):
        if verbose and len(files) > 1:
            print(f'\n[{i}/{len(files)}]', flush=True)
        try:
            result = reconstruct_file(fp, output_path=args.output, verbose=verbose)
            if result is None:
                n_skip += 1
            else:
                n_done += 1
        except Exception as e:
            n_err += 1
            print(f'  ERROR processing {fp}: {e}', file=sys.stderr)

    if verbose:
        print(f'\nDone. {n_done} reconstructed, {n_skip} skipped, {n_err} errors.',
              flush=True)


if __name__ == '__main__':
    main()
