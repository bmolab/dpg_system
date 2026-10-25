"""np limits, running totals and targets: one page per node.

None any more: the np.clip, np.min, np.max, np.cumsum, np.cumprod, np.diff,
np.distance_from_target, proximity_trigger and np.proximity_to_target pages
are all hand-made now (see tools/README.md). The shared pieces below are kept
for reference; nothing calls build().

These were three family pages: np_clip (np.clip, np.min, np.max; one class,
NumpyClipNode), np_cumulative (np.cumsum, np.cumprod, np.diff) and np_target
(np.distance_from_target, np.proximity_to_target, proximity_trigger). Each
now has its own page whose demo shows the input and the output, with a SEE
ALSO row under it.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_help import build
from help_common import SIG, PLOT, INT


LIST = {'text in': '', 'font size': '24'}


def see_also(y, stems, x=30, right=580):
    """A SEE ALSO row: one help_link per related page, left to right, wrapping
    onto a second row rather than running under the prose."""
    row = []
    x0 = x
    for s in stems:
        w = int(len(s) * 7.7) + 30
        if x > x0 and x + w > right:
            x, y = x0, y + 44
        row.append({'key': 'sa_' + s, 'init': 'help_link ' + s, 'pos': (x, y),
                    'w': w, 'h': 30})
        x += w + 14
    if x + 200 > right:
        x, y = x0, y + 44
    row.append({'key': 'sa_c', 'comment': True, 'text': 'SEE ALSO - click to open',
                'pos': (x, y)})
    return row


def text_h(body):
    """Height of the prose block: 15px per line at text size 30, plus a margin."""
    return (body.count('\n') + 1) * 15 + 30


def set_state(path, node_name, state):
    """Add saved node state (what a node's save_custom writes) to a built page.

    build() has no slot for it; the target nodes keep their targets this way,
    so the demo works the moment the page opens."""
    doc = json.load(open(path))
    patches = list(doc['patches'].values()) if 'patches' in doc else [doc]
    for p in patches:
        for n in p['nodes'].values():
            if n.get('name') == node_name:
                n.update(state)
    json.dump(doc, open(path, 'w'), indent=4)


# --------------------------------------------------------- demo subpatchers
# Inside make_point: two sines at different speeds make an x y point that
# wanders over the square from -1 to 1, coming back near any spot every few
# seconds (the path repeats every 12 seconds).
POINT_SUB = {
    'name': 'make_point',
    'host': 'mp',
    'demo': [
        {'key': 'sx', 'init': 'signal 3.0 sin', 'pos': (40, 40), 'w': 129, 'h': 78,
         'props': SIG('sin', 3.0)},
        {'key': 'sy', 'init': 'signal 4.0 cos', 'pos': (200, 40), 'w': 129, 'h': 78,
         'props': SIG('cos', 4.0)},
        {'key': 'cx', 'init': 'comment x: once every 3 s, y: once every 4 s, both -1 to 1',
         'pos': (40, 135), 'w': 460, 'h': 30},
        {'key': 'pk', 'init': 'pack 2', 'pos': (40, 180), 'w': 130, 'h': 100},
        {'key': 'cp', 'init': 'comment the point as a list: x y',
         'pos': (200, 200), 'w': 260, 'h': 30},
        {'key': 'out', 'init': 'out point', 'pos': (40, 310), 'w': 150, 'h': 80},
    ],
    'links': [('sx', '', 'pk', 'in 1'), ('sy', '', 'pk', 'in 2'),
              ('pk', 'out', 'out', 'point')],
}

TARGETS = [[0.5, 0.7], [-0.5, -0.7], [0.0, 0.0]]


# np.clip, np.min, np.max: their pages are hand-made now (see tools/README.md).

# np.cumsum, np.cumprod, np.diff: their pages are hand-made now (see tools/README.md).

# np.distance_from_target, proximity_trigger, np.proximity_to_target: their
# pages are hand-made now (see tools/README.md).
