"""np selecting: one page per node.

None any more: the np.[], np.crop, np.where and np.argwhere pages are all
hand-made now (see tools/README.md). The shared pieces below are kept for
reference; nothing calls build().

These were one family page (np_select). Each now has its own page whose demo
shows the source array and the selected part side by side, with a SEE ALSO row
of help_links under it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build


def HM(width, height, count, miny=0.0, maxy=1.0, fmt='%.2f', color='viridis'):
    """Property block for a heat_map that draws a whole array as one picture."""
    return {'color': color, 'width': width, 'height': height, 'sample count': count,
            'min y': miny, 'max y': maxy, 'update_mode': 'heat_map',
            'number format': fmt}


LIST = {'text in': '', 'font size': '24'}


def see_also(y, stems, x=30):
    """A SEE ALSO row: one help_link per related page, left to right."""
    row = []
    for s in stems:
        w = int(len(s) * 7.7) + 30
        row.append({'key': 'sa_' + s, 'init': 'help_link ' + s, 'pos': (x, y),
                    'w': w, 'h': 30})
        x += w + 14
    row.append({'key': 'sa_c', 'comment': True, 'text': 'SEE ALSO - click to open',
                'pos': (x, y)})
    return row


# np.[]: its page is hand-made now (see tools/README.md).

# np.crop: its page is hand-made now (see tools/README.md).

# np.where: its page is hand-made now (see tools/README.md).

# np.argwhere: its page is hand-made now (see tools/README.md).
