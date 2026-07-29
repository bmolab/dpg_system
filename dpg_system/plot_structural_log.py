#!/usr/bin/env python3
"""Plot per-frame diagnostics from StructuralStream JSONL log.

Reads the file written when SMPL_STRUCTURAL_LOG (or
StructuralStream.set_log_path) is enabled. Four figures are available:

  --overview (default): 5-panel time series — a_hz raw vs filt,
    |zmp_displacement|, LF/RF share, branch tape, support_frac.

  --zmp-trail: ZMP horizontal trajectory (x vs z) over the selected
    window, colored by frame, with foot rep positions overlaid.
    A clean walk produces a recognizable "butterfly" pattern; noise
    looks like a scribble.

  --share-flips: LF share over time + d(share)/dt, with sign-change
    markers and a flips-per-second metric.

  --root-relative: per-foot velocity in root-forward direction
    (v_rel_root_LF/RF), plus signed L−R distance and its rate.
    Robust to global drift and apparent foot-slide.

Summary always prints: a_hz attenuation quartiles, near-50/50 rate,
|zmp_displacement| medians by branch, share flips per second, plus
v_rel_root zero-crossing rates (the candidate touchdown signal).

Usage:
    python plot_structural_log.py LOG.jsonl [--save out.png]
                                            [--frames START:END]
                                            [--branch multi_foot,single_foot]
                                            [--zmp-trail] [--share-flips]
                                            [--root-relative]
                                            [--no-overview]
"""

import argparse
import json
import sys

import numpy as np
import matplotlib.pyplot as plt


BRANCH_COLORS = {
    'freefall':         '#cc3333',
    'no_evaluator':     '#888888',
    'no_candidates':    '#bbbbbb',
    'no_foot_candidates': '#999999',
    'single_foot':      '#3377cc',
    'multi_foot':       '#33aa55',
}


def load_jsonl(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"warning: skipping malformed line: {e}", file=sys.stderr)
    return rows


def vec_mag(v):
    if v is None:
        return np.nan
    return float(np.hypot(v[0], v[1]))


def filter_rows(rows, branches=None, frame_range=None):
    out = rows
    if branches:
        out = [r for r in out if r.get('branch') in branches]
    if frame_range:
        lo, hi = frame_range
        out = [r for r in out if lo <= r.get('frame', -1) < hi]
    return out


def parse_frame_range(s):
    if not s:
        return None
    a, b = s.split(':')
    return int(a), int(b)


def summarize(rows):
    if not rows:
        print("no rows")
        return

    # a_hz attenuation
    ratios = []
    for r in rows:
        raw = r.get('a_hz_raw')
        filt = r.get('a_hz_filt')
        if raw is None or filt is None:
            continue
        m_raw = vec_mag(raw)
        m_filt = vec_mag(filt)
        if m_raw > 0.05:  # ignore near-zero raw
            ratios.append(m_filt / m_raw)
    if ratios:
        print(f"a_hz attenuation (filt/raw):  median={np.median(ratios):.3f}  "
              f"p25={np.percentile(ratios, 25):.3f}  "
              f"p75={np.percentile(ratios, 75):.3f}  "
              f"(N={len(ratios)})")

    # 50/50 collapse rate among multi_foot frames
    multi = [r for r in rows if r.get('branch') == 'multi_foot']
    near_5050 = 0
    informative = 0
    for r in multi:
        gf = r.get('group_forces') or {}
        lf = gf.get('LF', 0.0) or 0.0
        rf = gf.get('RF', 0.0) or 0.0
        tot = lf + rf
        if tot < 1.0:
            continue
        share_lf = lf / tot
        informative += 1
        if abs(share_lf - 0.5) < 0.05:
            near_5050 += 1
    if informative:
        pct = 100.0 * near_5050 / informative
        print(f"multi_foot frames within ±0.05 of 50/50:  {near_5050}/{informative} "
              f"({pct:.1f}%)")

    # ZMP displacement by branch
    for branch in ('single_foot', 'multi_foot'):
        mags = [vec_mag(r.get('zmp_displacement'))
                for r in rows
                if r.get('branch') == branch and r.get('zmp_displacement')]
        mags = [m for m in mags if not np.isnan(m)]
        if mags:
            print(f"|zmp_displacement| in {branch:12s}:  "
                  f"median={np.median(mags):.3f} m  "
                  f"p90={np.percentile(mags, 90):.3f} m  (N={len(mags)})")

    # branch tally
    print("branch counts:")
    counts = {}
    for r in rows:
        b = r.get('branch') or 'None'
        counts[b] = counts.get(b, 0) + 1
    for b, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {b:20s} {c}")

    # share flip rate (sign changes of share - 0.5 in multi_foot frames)
    flips_raw, total_raw = _count_share_flips(rows, key='group_forces_raw')
    flips_sm,  total_sm  = _count_share_flips(rows, key='group_forces')
    if total_raw > 0:
        print(f"LF/RF share flips (raw):       {flips_raw} across "
              f"{total_raw:.2f} s ({flips_raw/total_raw:.2f} /s)")
    if total_sm > 0:
        print(f"LF/RF share flips (post-EMA):  {flips_sm} across "
              f"{total_sm:.2f} s ({flips_sm/total_sm:.2f} /s)")

    # v_rel_root zero-crossings (candidate touchdown/liftoff events)
    for name, key in (('LF', 'v_rel_root_LF'), ('RF', 'v_rel_root_RF')):
        zc, sec = _count_zero_crossings(rows, key)
        if sec > 0:
            print(f"v_rel_root_{name} zero-crossings: {zc} across "
                  f"{sec:.2f} s ({zc/sec:.2f} /s)")


def _lf_share_series(rows, key='group_forces'):
    """Return parallel arrays (frames, lf_share, dt_per_frame) for multi_foot frames.

    `key` selects the source dict: 'group_forces' (post-EMA, default) or
    'group_forces_raw' (pre-EMA; falls back to group_forces if absent).
    lf_share is NaN where the frame is not multi_foot or has total force < 1.
    """
    frames = np.array([r.get('frame', i) for i, r in enumerate(rows)], dtype=float)
    share = np.full(len(rows), np.nan)
    dts = np.array([r.get('dt') or np.nan for r in rows], dtype=float)
    for i, r in enumerate(rows):
        if r.get('branch') != 'multi_foot':
            continue
        gf = r.get(key) or r.get('group_forces') or {}
        lf = gf.get('LF', 0.0) or 0.0
        rf = gf.get('RF', 0.0) or 0.0
        tot = lf + rf
        if tot >= 1.0:
            share[i] = lf / tot
    return frames, share, dts


def _count_zero_crossings(rows, key, dead_band=0.0):
    """Count sign changes of rows[i][key] across consecutive valid frames.

    Frames where the value is None or absent are treated as gaps: crossings
    only count between adjacent populated frames. `dead_band` ignores
    crossings where both values are within ±dead_band of zero.

    Returns (count, total_seconds_observed).
    """
    count = 0
    seconds = 0.0
    last_idx = None
    last_val = None
    for i, r in enumerate(rows):
        v = r.get(key)
        if v is None:
            last_idx = None
            last_val = None
            continue
        if last_idx is not None:
            if (abs(v) > dead_band or abs(last_val) > dead_band) \
                    and last_val * v < 0:
                count += 1
            dt = r.get('dt') or 0.0
            seconds += dt
        last_idx = i
        last_val = v
    return count, seconds


def _count_share_flips(rows, key='group_forces'):
    """Sign changes of (lf_share - 0.5) across consecutive multi_foot frames.

    Returns (flip_count, total_seconds_observed).
    """
    _, share, dts = _lf_share_series(rows, key=key)
    valid = ~np.isnan(share)
    if valid.sum() < 2:
        return 0, 0.0
    centered = share - 0.5
    flips = 0
    seconds = 0.0
    last = None
    for i in range(len(rows)):
        if not valid[i]:
            last = None
            continue
        if last is not None:
            if centered[last] * centered[i] < 0:
                flips += 1
            dt = dts[i] if not np.isnan(dts[i]) else 0.0
            seconds += dt
        last = i
    return flips, seconds


def plot(rows, save_path=None):
    if not rows:
        print("no rows to plot")
        return

    frames = np.array([r.get('frame', i) for i, r in enumerate(rows)])
    a_raw_mag = np.array([vec_mag(r.get('a_hz_raw')) for r in rows])
    a_filt_mag = np.array([vec_mag(r.get('a_hz_filt')) for r in rows])
    zmp_disp_mag = np.array([vec_mag(r.get('zmp_displacement')) for r in rows])

    lf_share = np.full(len(rows), np.nan)
    rf_share = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        gf = r.get('group_forces') or {}
        lf = gf.get('LF', 0.0) or 0.0
        rf = gf.get('RF', 0.0) or 0.0
        tot = lf + rf
        if tot > 0.5:
            lf_share[i] = lf / tot
            rf_share[i] = rf / tot

    sf_raw = np.array([r.get('support_frac_raw') if r.get('support_frac_raw') is not None
                       else np.nan for r in rows])
    sf_filt = np.array([r.get('support_frac_filt') if r.get('support_frac_filt') is not None
                        else np.nan for r in rows])

    branches = [r.get('branch') for r in rows]

    fig, axes = plt.subplots(5, 1, figsize=(13, 11), sharex=True)

    ax = axes[0]
    ax.plot(frames, a_raw_mag, label='|a_hz| raw', color='#cc3333', lw=1.2)
    ax.plot(frames, a_filt_mag, label='|a_hz| filt', color='#225588', lw=1.2)
    ax.set_ylabel('m/s²')
    ax.set_title('Horizontal CoM acceleration: raw vs filtered')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(frames, zmp_disp_mag, color='#7733aa', lw=1.2)
    ax.set_ylabel('m')
    ax.set_title('|ZMP − CoM_projection|  (inverted-pendulum lever)')
    ax.axhline(0.05, color='gray', ls='--', lw=0.6, alpha=0.6)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(frames, lf_share, label='LF share', color='#3377cc', lw=1.3)
    ax.plot(frames, rf_share, label='RF share', color='#cc7733', lw=1.3)
    ax.axhline(0.5, color='gray', ls='--', lw=0.6, alpha=0.6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('share of foot force')
    ax.set_title('LF / RF force share (only when both feet are candidates)')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    ax = axes[3]
    yvals = np.zeros(len(rows))
    colors = [BRANCH_COLORS.get(b, '#dddddd') for b in branches]
    ax.bar(frames, np.ones(len(rows)), bottom=yvals, color=colors, width=1.0,
           edgecolor='none')
    ax.set_yticks([])
    ax.set_title('Branch (decision path)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=b)
               for b, c in BRANCH_COLORS.items()
               if b in set(branches)]
    ax.legend(handles=handles, loc='upper right', ncol=len(handles), fontsize=8)

    ax = axes[4]
    ax.plot(frames, sf_raw, label='support_frac raw', color='#cc3333', lw=1.0)
    ax.plot(frames, sf_filt, label='support_frac filt', color='#225588', lw=1.0)
    ax.axhline(1.0, color='gray', ls='--', lw=0.6, alpha=0.6)
    ax.set_ylabel('×bodyweight')
    ax.set_xlabel('frame')
    ax.set_title('Support fraction')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110)
        print(f"saved {save_path}")
    else:
        plt.show()


def plot_zmp_trail(rows, save_path=None):
    """ZMP horizontal trajectory over the selected window, with foot positions.

    Color encodes frame index (viridis). Foot rep positions are scattered
    underneath in a paler color. Also draws a faint line connecting the
    foot midpoint to the ZMP each frame, to show whether ZMP leans toward
    one foot or the other.
    """
    pts = []
    feet = {'LF': [], 'RF': []}
    frame_ids = []
    for r in rows:
        zmp = r.get('zmp_approx')
        if not zmp:
            continue
        pts.append(zmp)
        frame_ids.append(r.get('frame', 0))
        rp = r.get('rep_pos_hz') or {}
        for g in ('LF', 'RF'):
            if g in rp:
                feet[g].append(rp[g])

    if len(pts) < 2:
        print("plot_zmp_trail: need at least 2 frames with zmp_approx")
        return

    pts = np.array(pts)
    frame_ids = np.array(frame_ids)

    fig, ax = plt.subplots(figsize=(8, 8))

    for g, color in (('LF', '#3377cc'), ('RF', '#cc7733')):
        if feet[g]:
            arr = np.array(feet[g])
            ax.scatter(arr[:, 0], arr[:, 1], s=8, alpha=0.25, color=color,
                       label=f'{g} reps', zorder=1)

    sc = ax.scatter(pts[:, 0], pts[:, 1], c=frame_ids, cmap='viridis',
                    s=10, alpha=0.85, zorder=3)
    ax.plot(pts[:, 0], pts[:, 1], color='gray', lw=0.4, alpha=0.4, zorder=2)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('frame')

    ax.set_xlabel('horizontal x (m)')
    ax.set_ylabel('horizontal z (m)')
    ax.set_title('ZMP trajectory + foot rep positions  '
                 '(clean walk = double-loop "butterfly")')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110)
        print(f"saved {save_path}")
    else:
        plt.show()


def plot_share_flips(rows, save_path=None):
    """LF share over time + d(share)/dt + flip markers (raw & post-EMA)."""
    frames, share_raw, dts = _lf_share_series(rows, key='group_forces_raw')
    frames, share_sm,  _   = _lf_share_series(rows, key='group_forces')
    valid_raw = ~np.isnan(share_raw)
    if valid_raw.sum() < 2:
        print("plot_share_flips: need at least 2 multi_foot frames with share")
        return
    raw_present = valid_raw.sum() > 0 and not np.allclose(
        np.nan_to_num(share_raw, nan=0.0), np.nan_to_num(share_sm, nan=0.0))

    # First difference per frame (ignoring NaN gaps)
    def _diff(series):
        d = np.full_like(series, np.nan)
        last = None
        for i in range(len(series)):
            if np.isnan(series[i]):
                last = None
                continue
            if last is not None:
                dt = dts[i] if not np.isnan(dts[i]) else 1.0
                d[i] = (series[i] - series[last]) / max(dt, 1e-6)
            last = i
        return d

    def _flip_indices(series):
        centered = series - 0.5
        out = []
        last = None
        for i in range(len(series)):
            if np.isnan(series[i]):
                last = None
                continue
            if last is not None and centered[last] * centered[i] < 0:
                out.append(i)
            last = i
        return out

    dshare_sm = _diff(share_sm)
    dshare_raw = _diff(share_raw) if raw_present else None
    flips_sm = _flip_indices(share_sm)
    flips_raw = _flip_indices(share_raw) if raw_present else []

    n_raw, sec_raw = _count_share_flips(rows, key='group_forces_raw')
    n_sm,  sec_sm  = _count_share_flips(rows, key='group_forces')

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    ax = axes[0]
    if raw_present:
        ax.plot(frames, share_raw, color='#cc3333', lw=0.8, alpha=0.6,
                label='LF share (raw)')
    ax.plot(frames, share_sm, color='#3377cc', lw=1.3, label='LF share (post-EMA)')
    ax.axhline(0.5, color='gray', ls='--', lw=0.6, alpha=0.6)
    if flips_sm:
        ax.plot(frames[flips_sm], share_sm[flips_sm], 'o', color='#225588',
                ms=4, label=f'flip post-EMA ({len(flips_sm)})')
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('LF share')
    title = (f'LF share — raw: {n_raw} flips / {sec_raw:.2f} s '
             f'({n_raw/sec_raw:.2f}/s)' if sec_raw > 0 else 'LF share')
    if sec_sm > 0:
        title += (f'   |   post-EMA: {n_sm} flips / {sec_sm:.2f} s '
                  f'({n_sm/sec_sm:.2f}/s)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    ax = axes[1]
    if raw_present and dshare_raw is not None:
        ax.plot(frames, dshare_raw, color='#cc3333', lw=0.8, alpha=0.5,
                label='raw')
    ax.plot(frames, dshare_sm, color='#7733aa', lw=1.0, label='post-EMA')
    ax.axhline(0.0, color='gray', ls='--', lw=0.6, alpha=0.6)
    ax.set_xlabel('frame')
    ax.set_ylabel('d(LF share) / dt  (1/s)')
    ax.set_title('Rate of change of LF share  '
                 '(biologically plausible bound: roughly ±2 / stride)')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110)
        print(f"saved {save_path}")
    else:
        plt.show()


def plot_root_relative(rows, save_path=None):
    """Per-foot velocity in root-forward direction + signed L−R distance.

    Top panel:  v_rel_root_LF (blue) and v_rel_root_RF (orange).
                Stance reads ~−|com_speed| (foot left behind by body),
                swing reads positive (foot moving forward faster).
                Zero-crossings mark candidate touchdown / liftoff events.

    Bottom panel: signed inter-foot distance d_LR_signed (gray) and its
                  rate v_LR_signed (purple). v_LR sign tells which foot
                  is currently advancing.
    """
    frames = np.array([r.get('frame', i) for i, r in enumerate(rows)], dtype=float)

    def _series(key):
        return np.array([r.get(key) if r.get(key) is not None else np.nan
                         for r in rows], dtype=float)

    v_LF = _series('v_rel_root_LF')
    v_RF = _series('v_rel_root_RF')
    d_LR = _series('d_LR_signed')
    v_LR = _series('v_LR_signed')

    if np.all(np.isnan(v_LF)) and np.all(np.isnan(v_RF)):
        print("plot_root_relative: log has no v_rel_root_* fields "
              "(re-capture with the updated structural stream)")
        return

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)

    ax = axes[0]
    ax.plot(frames, v_LF, color='#3377cc', lw=1.2, label='v_rel_root LF')
    ax.plot(frames, v_RF, color='#cc7733', lw=1.2, label='v_rel_root RF')
    ax.axhline(0.0, color='gray', ls='--', lw=0.8, alpha=0.7)
    # Reference bands for "clear stance" / "clear swing"
    ax.axhspan(-3.0, -0.4, color='#3377cc', alpha=0.05)
    ax.axhspan(+0.4, +3.0, color='#cc7733', alpha=0.05)
    ax.set_ylabel('m/s')
    zc_lf, sec_lf = _count_zero_crossings(rows, 'v_rel_root_LF')
    zc_rf, sec_rf = _count_zero_crossings(rows, 'v_rel_root_RF')
    title = 'Foot velocity in root-forward direction  '
    if sec_lf > 0:
        title += (f'(LF zero-crossings: {zc_lf}/{sec_lf:.1f}s = '
                  f'{zc_lf/sec_lf:.2f}/s,  ')
    if sec_rf > 0:
        title += (f'RF: {zc_rf}/{sec_rf:.1f}s = {zc_rf/sec_rf:.2f}/s)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(frames, d_LR, color='#888888', lw=1.0, label='d_LR_signed (m)')
    ax2.plot(frames, v_LR, color='#7733aa', lw=1.0, label='v_LR_signed (m/s)')
    ax.axhline(0.0, color='gray', ls=':', lw=0.6, alpha=0.6)
    ax.set_xlabel('frame')
    ax.set_ylabel('signed L−R distance (m)', color='#444444')
    ax2.set_ylabel('d/dt(L−R distance) (m/s)', color='#7733aa')
    ax.set_title('Inter-foot signed distance and its rate '
                 '(sign of v_LR = which foot is advancing)')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110)
        print(f"saved {save_path}")
    else:
        plt.show()


def _suffix_save(save_path, suffix):
    if not save_path:
        return None
    if '.' in save_path.rsplit('/', 1)[-1]:
        head, ext = save_path.rsplit('.', 1)
        return f"{head}_{suffix}.{ext}"
    return f"{save_path}_{suffix}"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('log', help='Path to JSONL log file')
    p.add_argument('--save', help='Save figure(s) to PATH instead of displaying. '
                                  'When multiple figures are requested, '
                                  '_overview / _zmp / _flips suffixes are appended.')
    p.add_argument('--frames', help='Frame range as START:END (half-open)')
    p.add_argument('--branch', help='Comma-separated list of branches to keep '
                                    '(e.g. "multi_foot,single_foot")')
    p.add_argument('--summary-only', action='store_true',
                   help='Print summary, skip plot')
    p.add_argument('--no-overview', action='store_true',
                   help='Skip the default 5-panel overview figure')
    p.add_argument('--zmp-trail', action='store_true',
                   help='Render the ZMP horizontal trajectory figure')
    p.add_argument('--share-flips', action='store_true',
                   help='Render the LF-share + d(share)/dt figure with flip markers')
    args = p.parse_args()

    rows = load_jsonl(args.log)
    print(f"loaded {len(rows)} rows from {args.log}")

    branches = set(args.branch.split(',')) if args.branch else None
    frame_range = parse_frame_range(args.frames)
    rows = filter_rows(rows, branches=branches, frame_range=frame_range)
    if branches or frame_range:
        print(f"filtered to {len(rows)} rows")

    summarize(rows)
    if args.summary_only:
        return

    extras = bool(args.zmp_trail or args.share_flips)
    show_overview = not args.no_overview

    if show_overview:
        save = _suffix_save(args.save, 'overview') if extras else args.save
        plot(rows, save_path=save)
    if args.zmp_trail:
        save = _suffix_save(args.save, 'zmp') if (show_overview or args.share_flips) else args.save
        plot_zmp_trail(rows, save_path=save)
    if args.share_flips:
        save = _suffix_save(args.save, 'flips') if (show_overview or args.zmp_trail) else args.save
        plot_share_flips(rows, save_path=save)


if __name__ == '__main__':
    main()
