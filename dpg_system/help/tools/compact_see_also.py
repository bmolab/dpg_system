"""Restyle a page's SEE ALSO row: the label above, the buttons in a tight row.

The old row put 'SEE ALSO - click to open' at the right-hand end of a widely
spaced row of help_links. David's layout (np.nanmean, 2026-10-18) puts the
label on its own line and the buttons close together under it:

    SEE ALSO - click to open
    [np.mean] [np.nanmedian] [np.nansum] ...

The label takes the row's old position; the buttons go one line lower, GAP
apart, from the label's x; everything else at or below the old row (the back
button, usually) moves down by that line too, so nothing ends up covered.
Text blocks are left where they are.

    python3 dpg_system/help/tools/compact_see_also.py dpg_system/help/*_help.json
    python3 dpg_system/help/tools/compact_see_also.py --dry dpg_system/help/x_help.json
"""
import json, sys

LABEL = 'SEE ALSO - click to open'
GAP = 16          # between buttons
LINE = 34         # label height plus a little air: the buttons sit this far below it


def link_width(stem):
    """A help_link's drawn width, from its text (matches what the app makes)."""
    return int(round(6.3 * len(stem) + 16))


def _label(n):
    return (n.get('init') or n.get('name') or '').strip()


def compact(patch):
    """Restyle the SEE ALSO row of one patch dict in place. Returns True if changed."""
    nodes = list(patch.get('nodes', {}).values())
    label = next((n for n in nodes if _label(n) == 'comment ' + LABEL), None)
    links = [n for n in nodes if _label(n).startswith('help_link ')]
    if label is None or not links:
        return False
    row_y = min(n['position_y'] for n in links)
    row = [n for n in links if abs(n['position_y'] - row_y) < 20]
    if not row:
        return False
    row.sort(key=lambda n: n['position_x'])
    x0 = min(n['position_x'] for n in row)
    # a row laid out this way already (by hand, or by an earlier run) is left alone
    already = 20 <= row_y - label['position_y'] <= LINE + 10 and abs(label['position_x'] - x0) <= 8
    if already:
        return False
    # make room: everything at or below the row moves down a line
    for n in nodes:
        if n is label or n in row or n.get('name') == 'text_block':
            continue
        if n['position_y'] >= row_y - 5:
            n['position_y'] += LINE
    label['position_x'], label['position_y'] = x0, row_y
    x = x0
    for n in row:
        stem = _label(n)[len('help_link '):].split(' ')[0]
        w = link_width(stem)
        n['position_x'], n['position_y'] = x, row_y + LINE
        n['width'] = w
        x += w + GAP
    if 'height' in patch and isinstance(patch['height'], (int, float)):
        patch['height'] += LINE
    return True


def compact_file(path, dry=False):
    raw = open(path).read()
    d = json.loads(raw)
    patches = d['patches'].values() if 'patches' in d else [d]
    changed = sum(compact(p) for p in patches)
    if changed and not dry:
        indent = 4 if raw.startswith('{\n') else None
        with open(path, 'w') as f:
            json.dump(d, f, indent=indent)
    return changed


if __name__ == '__main__':
    dry = '--dry' in sys.argv
    paths = [a for a in sys.argv[1:] if a.endswith('.json')]
    total = sum(bool(compact_file(p, dry=dry)) for p in paths)
    print(f'{total} of {len(paths)} pages restyled' + (' (dry run)' if dry else ''))
