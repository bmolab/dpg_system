#!/usr/bin/env python3
"""Stream Disagreement Diagnostic for Log-Odds Contact Estimator.

Processes multiple motion files and identifies frames where individual
streams disagree with the consensus of other streams. Outputs:
  1. CSV with per-frame, per-group stream data
  2. Human-readable summary of "disagreement events" for adjudication

Usage:
    python stream_diagnostic.py file1.npz file2.npz ...
    python stream_diagnostic.py /path/to/directory/   # processes all .npz files
"""

import sys
import os
import csv
import glob
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

STREAM_NAMES = ['height', 'velocity', 'trajectory', 'hspeed', 'equilibrium', 'touchdown']


@dataclass
class DisagreementEvent:
    """A contiguous range of frames where a stream disagrees with consensus."""
    file: str
    group: str
    stream: str
    start_frame: int
    end_frame: int
    # Average values during the event
    avg_stream_inc: float
    avg_consensus_inc: float  # sum of OTHER streams
    avg_intensity: float
    avg_height_m: float
    avg_vy: float
    avg_decel: float
    avg_hspeed: float
    avg_straightness: float


def process_file(filepath, output_csv_writer, model_path=None):
    """Process a single .npz file and return disagreement events."""
    from dpg_system.smpl_processor import SMPLProcessor, SMPLProcessingOptions

    data = np.load(filepath, allow_pickle=True)
    fps = float(data.get('mocap_framerate', 120.0))
    poses, trans = data['poses'], data['trans']
    basename = os.path.basename(filepath)

    def reshape(p):
        return p[:72].reshape(1, 24, 3) if p.ndim == 1 and p.size >= 72 else p

    if model_path is None:
        model_path = os.path.dirname(os.path.abspath(__file__))

    opts = SMPLProcessingOptions(
        input_type='axis_angle', input_up_axis='Y', axis_permutation='x,z,-y',
        quat_format='wxyz', dt=1.0/fps, add_gravity=True,
        enable_passive_limits=True, enable_apparent_gravity=True,
        floor_enable=True, floor_height=0.0, floor_tolerance=0.15,
        heel_toe_bias=0.0, contact_method='logodds',
        enable_frame_evaluator=False, balance_mode='raw',
        world_frame_dynamics=True,
        com_pos_min_cutoff=999.0, com_pos_beta=1.0,
        com_vel_min_cutoff=20.0, com_vel_beta=0.1,
        com_acc_min_cutoff=5.0, com_acc_beta=0.8,
        smooth_input_window=0, enable_one_euro_filter=False,
        acc_smooth_window=7, torque_smooth_window=0,
        smooth_contact_forces=False, enable_velocity_gate=False,
        use_s_curve_spine=True,
    )

    proc = SMPLProcessor(
        framerate=fps, total_mass_kg=75.0,
        betas=data['betas'],
        gender=str(data['gender']),
        model_path=model_path,
    )

    # Collect per-frame data
    frame_data = []  # list of dicts
    num_frames = min(poses.shape[0], 10000)  # cap for safety

    print(f"  Processing {basename}: {num_frames} frames at {fps} fps...")

    for f in range(num_frames):
        try:
            res = proc.process_frame(reshape(poses[f]), trans[f:f+1], opts)
        except Exception as e:
            continue

        result = getattr(proc, '_logodds_result', None)
        if result is None:
            continue

        for gname in result.intensity:
            ctx = result.stream_context.get(gname, {})
            ps = result.per_stream.get(gname, {})

            # Per-stream increments (weighted)
            increments = {}
            for sname in STREAM_NAMES:
                increments[sname] = ps.get(sname, 0.0)

            total_inc = sum(increments.values())
            intensity = result.intensity.get(gname, 0.0)

            row = {
                'file': basename,
                'frame': f,
                'group': gname,
                'height_m': ctx.get('height_m', 999.0),
                'vy': ctx.get('vy', 0.0),
                'decel': ctx.get('decel', 0.0),
                'hspeed': ctx.get('hspeed', 0.0),
                'straightness': ctx.get('straightness', 0.0),
                'intensity': intensity,
                'logodds': result.log_odds_state.get(gname, 0.0),
            }
            for sname in STREAM_NAMES:
                row[f'stream_{sname}'] = increments[sname]
            row['total_increment'] = total_inc

            # Determine consensus direction (sign of total excluding each stream)
            for sname in STREAM_NAMES:
                others_total = total_inc - increments[sname]
                row[f'{sname}_others_total'] = others_total

            frame_data.append(row)
            if output_csv_writer:
                output_csv_writer.writerow(row)

    # --- Find disagreement events ---
    events = find_disagreement_events(frame_data, basename)
    return events, len(frame_data)


def find_disagreement_events(frame_data, filename):
    """Find contiguous frame ranges where a stream disagrees with consensus."""
    events = []

    # Group data by (group, stream)
    # A "disagreement" = stream's sign differs from the consensus of others
    # AND the magnitude of the stream's contribution is significant
    MIN_MAGNITUDE = 0.5  # stream must contribute at least this to count

    # Organize by group
    by_group = defaultdict(list)
    for row in frame_data:
        by_group[row['group']].append(row)

    for gname, rows in by_group.items():
        rows.sort(key=lambda r: r['frame'])

        for sname in STREAM_NAMES:
            # Track disagreement runs
            in_run = False
            run_start = 0
            run_data = []

            for row in rows:
                stream_val = row[f'stream_{sname}']
                others_val = row[f'{sname}_others_total']

                # Disagreement: stream pushes one direction, others push opposite,
                # and stream's contribution is significant
                is_disagreement = (
                    stream_val * others_val < 0  # opposite signs
                    and abs(stream_val) > MIN_MAGNITUDE
                )

                if is_disagreement:
                    if not in_run:
                        run_start = row['frame']
                        run_data = []
                        in_run = True
                    run_data.append(row)
                else:
                    if in_run and len(run_data) >= 3:
                        # Close event (require min 3 frames to avoid noise)
                        events.append(_build_event(
                            filename, gname, sname, run_data))
                    in_run = False
                    run_data = []

            # Close final run
            if in_run and len(run_data) >= 3:
                events.append(_build_event(filename, gname, sname, run_data))

    return events


def _build_event(filename, gname, sname, run_data):
    """Build a DisagreementEvent from a run of frames."""
    return DisagreementEvent(
        file=filename,
        group=gname,
        stream=sname,
        start_frame=run_data[0]['frame'],
        end_frame=run_data[-1]['frame'],
        avg_stream_inc=np.mean([r[f'stream_{sname}'] for r in run_data]),
        avg_consensus_inc=np.mean([r[f'{sname}_others_total'] for r in run_data]),
        avg_intensity=np.mean([r['intensity'] for r in run_data]),
        avg_height_m=np.mean([r['height_m'] for r in run_data]),
        avg_vy=np.mean([r['vy'] for r in run_data]),
        avg_decel=np.mean([r['decel'] for r in run_data]),
        avg_hspeed=np.mean([r['hspeed'] for r in run_data]),
        avg_straightness=np.mean([r['straightness'] for r in run_data]),
    )


def print_events_summary(all_events):
    """Print a human-readable summary of disagreement events."""
    if not all_events:
        print("\nNo significant disagreement events found.")
        return

    # Group by stream
    by_stream = defaultdict(list)
    for e in all_events:
        by_stream[e.stream].append(e)

    print("\n" + "=" * 80)
    print("STREAM DISAGREEMENT SUMMARY")
    print("=" * 80)
    print(f"Total events: {len(all_events)}")
    print()

    for sname in STREAM_NAMES:
        events = by_stream.get(sname, [])
        if not events:
            continue

        print(f"--- {sname.upper()} ({len(events)} events) ---")
        print()

        # Sort by duration (longest first — most impactful)
        events.sort(key=lambda e: -(e.end_frame - e.start_frame))

        for i, e in enumerate(events[:10]):  # top 10 per stream
            duration = e.end_frame - e.start_frame + 1
            direction = "ANTI-contact" if e.avg_stream_inc < 0 else "PRO-contact"

            print(f"  [{i+1}] {e.file} | {e.group} | f{e.start_frame}-{e.end_frame} ({duration}f)")
            print(f"      Stream says: {direction} ({e.avg_stream_inc:+.2f})")
            print(f"      Others say:  {'PRO' if e.avg_consensus_inc > 0 else 'ANTI'}-contact ({e.avg_consensus_inc:+.2f})")
            print(f"      Intensity:   {e.avg_intensity:.2f}")
            print(f"      Context:     h={e.avg_height_m:.3f}m  vy={e.avg_vy:+.2f}  "
                  f"decel={e.avg_decel:.2f}  hspd={e.avg_hspeed:.2f}  "
                  f"str={e.avg_straightness:.2f}")

            # Adjudication hint
            if e.avg_height_m < 0.02 and e.avg_stream_inc < 0:
                print(f"      ⚠ LIKELY FALSE NEGATIVE: foot on floor but {sname} says no contact")
            elif e.avg_height_m > 0.10 and e.avg_stream_inc > 0:
                print(f"      ⚠ LIKELY FALSE POSITIVE: foot off floor but {sname} says contact")
            else:
                print(f"      → NEEDS REVIEW: ambiguous ground truth")
            print()

        if len(events) > 10:
            print(f"  ... and {len(events) - 10} more events")
        print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Collect files
    files = []
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            files.extend(sorted(glob.glob(os.path.join(arg, '*.npz'))))
        elif os.path.isfile(arg) and arg.endswith('.npz'):
            files.append(arg)
        else:
            print(f"Warning: skipping {arg}")

    if not files:
        print("No .npz files found.")
        sys.exit(1)

    print(f"Processing {len(files)} file(s)...")

    # Output CSV
    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, 'stream_diagnostic_output.csv')

    fieldnames = [
        'file', 'frame', 'group', 'height_m', 'vy', 'decel',
        'hspeed', 'straightness', 'intensity', 'logodds',
    ]
    for sname in STREAM_NAMES:
        fieldnames.append(f'stream_{sname}')
    fieldnames.append('total_increment')
    for sname in STREAM_NAMES:
        fieldnames.append(f'{sname}_others_total')

    all_events = []
    total_frames = 0

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filepath in files:
            try:
                events, nframes = process_file(filepath, writer)
                all_events.extend(events)
                total_frames += nframes
            except Exception as e:
                print(f"  ERROR processing {filepath}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nProcessed {total_frames} frame-group records across {len(files)} files")
    print(f"CSV written to: {csv_path}")

    # Print summary
    print_events_summary(all_events)


if __name__ == '__main__':
    main()
