#!/usr/bin/env python3
"""Analyze stream_diagnostic_output.csv to show concordance vs disagreement.

Reads the CSV produced by stream_diagnostic.py and produces a clear
summary distinguishing frames/sections where streams agree from those
where they disagree.

Usage:
    python analyze_concordance.py [path/to/stream_diagnostic_output.csv]

If no path given, looks in the same directory as this script.
"""

import sys
import os
import csv
from collections import defaultdict

STREAM_NAMES = ['height', 'velocity', 'trajectory', 'hspeed', 'equilibrium', 'touchdown']

# Thresholds
SIGNIFICANT = 0.3    # stream contribution must exceed this to count
AGREE_LABEL = '✓'
DISAGREE_LABEL = '✗'


def load_csv(path):
    """Load CSV into list of dicts, casting numeric fields."""
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                if key not in ('file', 'group'):
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
            rows.append(row)
    return rows


def classify_frame(row):
    """Classify each stream as agreeing or disagreeing with the others.

    Returns:
        dict with keys:
            'concordance': float 0-1 (1 = perfect agreement)
            'streams': dict of stream_name -> {'sign': +1/-1/0, 'agrees': bool, 'value': float}
            'n_disagree': int count of disagreeing streams
            'outlier_streams': list of stream names that disagree
    """
    # Get stream values
    values = {}
    signs = {}
    for s in STREAM_NAMES:
        v = row.get(f'stream_{s}', 0.0)
        values[s] = v
        if abs(v) < SIGNIFICANT:
            signs[s] = 0  # neutral — doesn't vote
        elif v > 0:
            signs[s] = 1
        else:
            signs[s] = -1

    # Consensus = majority sign among non-neutral streams
    voting = {s: signs[s] for s in STREAM_NAMES if signs[s] != 0}
    if not voting:
        return {
            'concordance': 1.0,
            'streams': {s: {'sign': 0, 'agrees': True, 'value': values[s]}
                        for s in STREAM_NAMES},
            'n_disagree': 0,
            'outlier_streams': [],
        }

    pos = sum(1 for v in voting.values() if v > 0)
    neg = sum(1 for v in voting.values() if v < 0)
    consensus_sign = 1 if pos >= neg else -1

    # Check each stream against consensus
    streams = {}
    outliers = []
    for s in STREAM_NAMES:
        if signs[s] == 0:
            streams[s] = {'sign': 0, 'agrees': True, 'value': values[s]}
        elif signs[s] == consensus_sign:
            streams[s] = {'sign': signs[s], 'agrees': True, 'value': values[s]}
        else:
            streams[s] = {'sign': signs[s], 'agrees': False, 'value': values[s]}
            outliers.append(s)

    n_voting = len(voting)
    n_agree = n_voting - len(outliers)
    concordance = n_agree / n_voting if n_voting > 0 else 1.0

    return {
        'concordance': concordance,
        'streams': streams,
        'n_disagree': len(outliers),
        'outlier_streams': outliers,
    }


def find_sections(classified_frames):
    """Group contiguous frames into concordant/discordant sections.

    Returns list of sections, each a dict:
        type: 'concordant' or 'discordant'
        start_frame, end_frame
        frames: list of (frame_num, classification, row)
    """
    if not classified_frames:
        return []

    sections = []
    current_type = None
    current_frames = []

    for frame_num, classification, row in classified_frames:
        is_discordant = classification['n_disagree'] > 0
        frame_type = 'discordant' if is_discordant else 'concordant'

        if frame_type != current_type:
            if current_frames:
                sections.append({
                    'type': current_type,
                    'start_frame': current_frames[0][0],
                    'end_frame': current_frames[-1][0],
                    'frames': current_frames,
                })
            current_type = frame_type
            current_frames = []

        current_frames.append((frame_num, classification, row))

    if current_frames:
        sections.append({
            'type': current_type,
            'start_frame': current_frames[0][0],
            'end_frame': current_frames[-1][0],
            'frames': current_frames,
        })

    return sections


def format_stream_bar(classification):
    """Create a compact visual bar showing stream agreement."""
    parts = []
    for s in STREAM_NAMES:
        info = classification['streams'][s]
        if info['sign'] == 0:
            parts.append(f'  ·  ')  # neutral
        elif info['agrees']:
            if info['sign'] > 0:
                parts.append(f' +{abs(info["value"]):.1f}')
            else:
                parts.append(f' -{abs(info["value"]):.1f}')
        else:
            # Disagreeing — highlight
            if info['sign'] > 0:
                parts.append(f'▶+{abs(info["value"]):.1f}')
            else:
                parts.append(f'▶-{abs(info["value"]):.1f}')
    return ' │'.join(parts)


def print_file_analysis(file_name, group_name, sections, total_frames):
    """Print analysis for one file+group combination."""
    n_concordant = sum(
        len(s['frames']) for s in sections if s['type'] == 'concordant')
    n_discordant = sum(
        len(s['frames']) for s in sections if s['type'] == 'discordant')

    pct_concordant = 100 * n_concordant / total_frames if total_frames > 0 else 0
    pct_discordant = 100 * n_discordant / total_frames if total_frames > 0 else 0

    print(f"\n{'─' * 90}")
    print(f"  {file_name} │ {group_name}")
    print(f"  {total_frames} frames: "
          f"{n_concordant} concordant ({pct_concordant:.0f}%) │ "
          f"{n_discordant} discordant ({pct_discordant:.0f}%)")
    print(f"{'─' * 90}")

    # Header
    header = '  frame  int  '
    for s in STREAM_NAMES:
        header += f'│ {s[:5]:>5} '
    header += '│ status'
    print(header)
    print(f"  {'─' * 84}")

    for section in sections:
        if section['type'] == 'concordant':
            duration = section['end_frame'] - section['start_frame'] + 1
            # Collapse concordant sections
            if duration <= 3:
                for fnum, cls, row in section['frames']:
                    _print_frame_line(fnum, cls, row, concordant=True)
            else:
                # Show first and last frame, collapse middle
                fnum, cls, row = section['frames'][0]
                _print_frame_line(fnum, cls, row, concordant=True)
                print(f"  {'':>5}  ... {duration - 2} concordant frames ...")
                fnum, cls, row = section['frames'][-1]
                _print_frame_line(fnum, cls, row, concordant=True)
        else:
            # Show all discordant frames
            # Section header
            duration = section['end_frame'] - section['start_frame'] + 1
            # Figure out which streams are outliers across this section
            outlier_counts = defaultdict(int)
            for _, cls, _ in section['frames']:
                for s in cls['outlier_streams']:
                    outlier_counts[s] += 1
            top_outlier = max(outlier_counts, key=outlier_counts.get) if outlier_counts else '?'
            print(f"  ┌─ DISAGREEMENT f{section['start_frame']}-{section['end_frame']} "
                  f"({duration}f) — primary outlier: {top_outlier} ─┐")

            for fnum, cls, row in section['frames']:
                _print_frame_line(fnum, cls, row, concordant=False)

            # Summary line for this disagreement section
            avg_intensity = sum(r['intensity'] for _, _, r in section['frames']) / len(section['frames'])
            avg_height = sum(r['height_m'] for _, _, r in section['frames']) / len(section['frames'])
            print(f"  └─ avg intensity={avg_intensity:.2f}  "
                  f"avg h={avg_height:.3f}m  "
                  f"outliers: {dict(outlier_counts)} ─┘")


def _print_frame_line(fnum, cls, row, concordant=True):
    """Print a single frame line."""
    intensity = row.get('intensity', 0.0)
    prefix = '  ' if concordant else '  │'

    parts = f"{prefix}{int(fnum):>5} {intensity:.2f} "
    for s in STREAM_NAMES:
        info = cls['streams'][s]
        v = info['value']
        if info['sign'] == 0:
            parts += f'│   ·   '
        elif info['agrees']:
            parts += f'│ {v:+5.2f} '
        else:
            parts += f'│▶{v:+5.2f}◀'

    if concordant:
        parts += f'│ {AGREE_LABEL}'
    else:
        outliers = ', '.join(cls['outlier_streams'])
        parts += f'│ {DISAGREE_LABEL} [{outliers}]'
    print(parts)


def print_global_summary(all_data):
    """Print aggregate statistics across all files."""
    # Count per-stream disagreement frequency
    stream_disagree_count = defaultdict(int)
    stream_disagree_neg = defaultdict(int)  # disagreeing AND anti-contact
    stream_disagree_pos = defaultdict(int)  # disagreeing AND pro-contact
    total_frames = 0

    for row in all_data:
        cls = classify_frame(row)
        total_frames += 1
        for s in cls['outlier_streams']:
            stream_disagree_count[s] += 1
            if cls['streams'][s]['sign'] < 0:
                stream_disagree_neg[s] += 1
            else:
                stream_disagree_pos[s] += 1

    print("\n" + "=" * 70)
    print("  GLOBAL DISAGREEMENT FREQUENCY")
    print("=" * 70)
    print(f"  Total frame-group records: {total_frames}")
    print()
    print(f"  {'Stream':<15} {'Disagree':>10} {'% of frames':>12} "
          f"{'Anti-contact':>13} {'Pro-contact':>13}")
    print(f"  {'─' * 63}")

    for s in STREAM_NAMES:
        n = stream_disagree_count.get(s, 0)
        pct = 100 * n / total_frames if total_frames > 0 else 0
        neg = stream_disagree_neg.get(s, 0)
        pos = stream_disagree_pos.get(s, 0)
        print(f"  {s:<15} {n:>10} {pct:>11.1f}% {neg:>13} {pos:>13}")

    # Most common disagreement patterns
    pattern_counts = defaultdict(int)
    for row in all_data:
        cls = classify_frame(row)
        if cls['outlier_streams']:
            key = tuple(sorted(cls['outlier_streams']))
            pattern_counts[key] += 1

    if pattern_counts:
        print()
        print(f"  {'─' * 63}")
        print(f"  Most common disagreement patterns:")
        for pattern, count in sorted(pattern_counts.items(),
                                      key=lambda x: -x[1])[:10]:
            pct = 100 * count / total_frames
            print(f"    {' + '.join(pattern):<40} {count:>6} ({pct:.1f}%)")


def main():
    # Find CSV
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, 'stream_diagnostic_output.csv')

    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        print("Run stream_diagnostic.py first to generate the data.")
        sys.exit(1)

    print(f"Loading {csv_path}...")
    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} records")

    # Global summary to stdout
    print_global_summary(rows)

    # Organize by file → group
    by_file_group = defaultdict(list)
    for row in rows:
        key = (row['file'], row['group'])
        by_file_group[key].append(row)

    # Write per-file timelines to a report file
    report_path = os.path.splitext(csv_path)[0] + '_concordance_report.txt'

    # Temporarily redirect print to report file
    original_stdout = sys.stdout

    with open(report_path, 'w') as report:
        sys.stdout = report

        # Write global summary to report too
        print_global_summary(rows)

        # Process each file+group
        for (file_name, group_name), group_rows in sorted(by_file_group.items()):
            classified = []
            for row in group_rows:
                frame = int(row['frame'])
                cls = classify_frame(row)
                classified.append((frame, cls, row))

            classified.sort(key=lambda x: x[0])
            sections = find_sections(classified)

            n_discordant = sum(
                len(s['frames']) for s in sections if s['type'] == 'discordant')
            if n_discordant == 0:
                continue

            print_file_analysis(file_name, group_name, sections, len(classified))

    sys.stdout = original_stdout
    print(f"\nPer-file concordance timelines written to:\n  {report_path}")
    print(f"Open this file to browse concordant vs discordant sections per group.")


if __name__ == '__main__':
    main()

