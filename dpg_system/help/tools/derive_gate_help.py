"""Derive the change, decreasing and pass help pages from the hand-made increasing_help.json.

    python3 dpg_system/help/tools/derive_gate_help.py

change, increasing, decreasing and pass are one node class (a gate that passes
the value itself when it passes a test). increasing_help.json is laid out by
hand: a signal straight into a plot and into the gate, the gate into a second
plot, and below a number box through a second gate into a number box. The
signal starts by itself (its 'on' is saved true), so there is no load_bang.

This copies that layout into decreasing (the mirror image), change (floor makes
the signal a staircase, and the gate feeds a counter) and pass (a fixed
threshold of 0.5), and writes each page's prose. increasing_help.json keeps its
own hand-edited prose; the text for it here is what it was derived from.

It is deliberately NOT a make_*_help.py: rebuild.py re-flows those. After running
it, use

    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

The template is read by node id, so the ids below must match increasing_help.json.
"""
import copy, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HELP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from derive_arithmetic_help import set_prop, set_comment, set_block, set_number, link

TITLE, TEXT, SIGNAL, PLOT_IN, GATE, PLOT_OUT, NOTE_OUT = 1470, 1481, 1515, 1537, 1588, 1607, 1653
F_IN, NOTE_DRAG, GATE2, F_OUT, CLOSE = 1664, 1700, 1711, 1730, 1763

PAGES = {}


def page_text(stem, title, body):
    PAGES[stem] = (title, body)


# ------------------------------------------------------------------ the prose
GATE_IN = """in:
The value to be tested, and passed on if it succeeds. Receiving data here
triggers the node."""

GATE_OPTIONS = """comparison (the first menu under the node):
The test: !=, ==, >, >=, <, <= or always. The node's name sets it ({op} here);
you can change it by hand, which turns this node into any of its siblings.

operand (under the menu):
What the value is compared with. While self_compare is checked this is the
previous value, filled in after each one arrives - it shows what the next value
will be compared with.

self_compare:
Checked (the default for this node): compare each value with the one before it.
Unchecked: compare with a fixed operand you set yourself, making a threshold gate.

force_int:
When checked, both sides are rounded to whole numbers before the comparison. It
affects only the test - what comes out is the value as it arrived."""

GATE_ARRAYS = """ARRAYS AND TENSORS:
The test passes if ANY element passes it, and then the whole array is sent."""

body = """change passes a value on only when it differs from the one before it.

A stream that repeats the same number frame after frame comes out as one value
per change: 3, 3, 3, 4, 4, 2 comes out as 3, 4, 2. What comes out is the value
itself, not true or false. The first value always passes - there is nothing
before it for it to equal.

In the demo, a ramp is cut into whole-number steps by floor - the staircase on
the plot - so it sends the same number for many frames in a row. change lets one
value through per step, so the counter beside it counts steps, not frames.
Below, the change node has force_int checked: drag the number slowly, and a value
gets through only when its whole-number part changes.

Use it to stop a slow or expensive part of a patch being retriggered by a stream
that has not moved: a menu index, a scene number, a rounded sensor reading.

SYNTAX:
change

EXAMPLE:
change

INPUTS and PARAMETERS:

""" + GATE_IN + """

""" + GATE_OPTIONS.format(op='!=') + """

OUTPUTS:

result:
The input, unchanged, whenever it differs from the value before it.
Otherwise nothing at all is sent.

NOISE:
A fractional sensor stream almost never repeats exactly, so change passes nearly
every frame of it. Round it first, or check force_int, to react only to changes
that matter.

""" + GATE_ARRAYS + """

SEE ALSO:
increasing    only values that have risen
decreasing    only values that have fallen
pass          a gate with a fixed threshold
togedge       react when a toggle turns on or off"""

page_text('change', 'change - pass a value on only when it has moved', body)


def direction_page(name, way, op, seq_out, use, other):
    """increasing and decreasing: mirror images."""
    body = f"""{name} passes a value on only when it is {way.upper()} than the one before it.

1, 3, 3, 2, 5 comes out as {seq_out}. What comes out is the value itself, not true
or false. The first value never passes: with nothing before it, it has not
{'risen' if name == 'increasing' else 'fallen'}. Every value is remembered whether or not it passes, so each is
compared with the one just before it, not with the last one sent.

In the demo, a sine wave goes through {name}. Only the {'rising' if name == 'increasing' else 'falling'} half of each wave
gets through, and the plot joins those halves end to end into a sawtooth that
only {'climbs' if name == 'increasing' else 'falls'}. Below it, drag the number {'up' if name == 'increasing' else 'down'} and the second box follows;
drag it the other way and the second box stays where it was.

{use}

SYNTAX:
{name}

EXAMPLE:
{name}

INPUTS and PARAMETERS:

""" + GATE_IN + """

""" + GATE_OPTIONS.format(op=op) + f"""

OUTPUTS:

result:
The input, unchanged, whenever it is {way} than the value before it.
Otherwise nothing at all is sent.

NOISE:
A noisy stream {'rises' if name == 'increasing' else 'falls'} on about half its frames even when nothing is really
happening. Smooth it first (filter, one_euro_filter), or check force_int, so only
real movement gets through.

""" + GATE_ARRAYS + f"""

SEE ALSO:
{other:<13} the opposite direction
change        any change at all
pass          a gate with a fixed threshold
diff          how much each value differs from the one before"""
    page_text(name, f'{name} - pass a value on only when it has {"risen" if name == "increasing" else "fallen"}', body)


direction_page('increasing', 'larger', '>', '3, 5: 3 rose from 1, the second 3\ndid not rise, 2 fell, and 5 rose',
               """Use it to catch a reading on its way up and ignore it on the way down: a hand
being raised, a sound swelling, a counter moving forward.""", 'decreasing')
direction_page('decreasing', 'smaller', '<', '2: the only value smaller than the one\nbefore it (3, 3 is no fall, and 5 rose)',
               """Use it to catch a reading on its way down and ignore it on the way up: a hand
being lowered, a sound dying away, a level draining.""", 'increasing')


# ------------------------------------------------------------------------ pass
body = """pass lets a value through when it passes a test, and sends the value itself -
not true or false, as a comparison node would.

"> 0.5" says whether a value is above 0.5. "pass > 0.5" sends the value on, but
only when it is above 0.5: 0.7 goes through as 0.7, and 0.3 sends nothing.

With no arguments pass lets everything through - useful as a tidy junction in a
patch. With only a comparison ("pass !=") it compares each value with the one
before it, like change.

In the demo a sine wave goes through "pass > 0.5". Only the tops of the waves
get through, and the plot joins them up. Drag the operand on the node to raise
or lower the cut. Below, drag the number across 0.5: under it nothing gets
through, so the second box keeps the last value that was above 0.5.

SYNTAX:
pass
pass <comparison>
pass <comparison> <operand: int or float>

EXAMPLES:
pass > 0.5      only values above 0.5
pass != 0       anything but zero
pass            everything

INPUTS and PARAMETERS:

in:
The value to be tested, and passed on if it succeeds. Receiving data here
triggers the node.

operand:
The number on the node: what the value is compared with. Drag it, or send a new
one to its inlet. Setting it sends nothing.

comparison:
The menu on the node: !=, ==, >, >=, <, <= or always (the default with no
arguments).

self_compare:
When checked, the operand is replaced by each value after it arrives, so the
node compares each value with the one before it. Given a comparison but no
operand, pass turns this on for you.

force_int:
When checked, both sides are rounded to whole numbers before the comparison.
It affects only the test - what comes out is the value as it arrived.

OUTPUTS:

result:
The input, unchanged, whenever it passes the test. Otherwise nothing is sent.

""" + GATE_ARRAYS + """

SEE ALSO:
change, increasing, decreasing    pass compared with the previous value
>, <, ==, ...                     true or false instead of the value
clamp                             hold a value inside a range rather than drop it"""

page_text('pass', 'pass - let a value through when it passes a test', body)


def gate(node, label, op, init=None, props=None):
    """Turn a template gate into another member of the family."""
    node['name'] = label
    if init:
        node['init'] = init
    else:
        node.pop('init', None)
    node['properties'] = {str(i): {'name': k, 'value': v, 'value_type': type(v).__name__ if not isinstance(v, str) else 'string'}
                          for i, (k, v) in enumerate((props or {'comparison_property': op, 'operand_property': 0.0,
                                                                 'self_compare': True, 'force_int': False}).items())}


def retitle(d, nodes, stem):
    title, body = PAGES[stem]
    d['name'] = stem + '_help'
    set_comment(nodes[TITLE], title)
    set_block(nodes[TEXT], body, '30')


def plain_node(name, node_id, sid, x, y, w, h, props=None):
    return {'name': name, 'id': node_id, 'sid': sid, 'position_x': x, 'position_y': y, 'width': w,
            'height': h, 'visibility': 'show_all', 'draggable': True, 'presentation_state': 'show_all',
            'properties': {str(i): {'name': k, 'value': v, 'value_type': type(v).__name__ if not isinstance(v, str) else 'string'}
                           for i, (k, v) in enumerate((props or {}).items())}}


def derive(template, stem):
    d = copy.deepcopy(template)
    nodes = {n['id']: n for n in d['nodes'].values()}
    links = list(d['links'].values())
    next_id = max(nodes) + 1
    next_sid = max(n.get('sid', 0) for n in nodes.values()) + 1
    retitle(d, nodes, stem)

    if stem == 'decreasing':
        gate(nodes[GATE], 'decreasing', '<')
        gate(nodes[GATE2], 'decreasing', '<')
        set_comment(nodes[NOTE_OUT], 'only the falling sections get through')
        set_comment(nodes[NOTE_DRAG], 'drag this down, then the other way')
        set_number(nodes[F_IN], 0.9)
        set_number(nodes[F_OUT], 0.37)

    elif stem == 'change':
        # a ramp cut into steps, so the same value repeats for many frames
        set_prop(nodes[SIGNAL], 'shape', 'saw')
        set_prop(nodes[SIGNAL], 'period', 4.0)
        set_prop(nodes[SIGNAL], 'range', 5.0)
        set_prop(nodes[SIGNAL], 'bipolar', False)
        set_prop(nodes[PLOT_IN], 'min y', -0.5)
        set_prop(nodes[PLOT_IN], 'max y', 5.0)
        fl = plain_node('floor', next_id, next_sid, 40, 190, 45, 42)
        nodes[fl['id']] = fl
        for l in links:
            if l['source_node'] == SIGNAL:
                l['source_node'], l['source_node_name'], l['source_output_name'] = fl['id'], 'floor', 'result'
        links.append(link(SIGNAL, 'signal', 0, '', fl['id'], 'floor', 0, 'in'))
        gate(nodes[GATE], 'change', '!=')
        # the gate counts steps: a counter and an int box where the second plot was
        old = nodes.pop(PLOT_OUT)
        cnt = plain_node('counter', PLOT_OUT, old['sid'], old['position_x'], old['position_y'], 123, 84, {'step': 1})
        out = plain_node('int', next_id + 1, next_sid + 1, old['position_x'], old['position_y'] + 100, 130, 42,
                         {'': 0, 'format': '%d', 'width': 100, 'font size': '24'})
        nodes[PLOT_OUT], nodes[out['id']] = cnt, out
        for l in links:
            if l['dest_node'] == PLOT_OUT:
                l['dest_node_name'], l['dest_input_name'] = 'counter', 'input'
        links.append(link(PLOT_OUT, 'counter', 0, 'count out', out['id'], 'int', 0, ''))
        set_comment(nodes[NOTE_OUT], 'counts steps, not frames')
        gate(nodes[GATE2], 'change', '!=', props={'comparison_property': '!=', 'operand_property': 1.0,
                                                    'self_compare': True, 'force_int': True})
        set_comment(nodes[NOTE_DRAG], 'drag slowly: only a new whole number gets through')
        set_number(nodes[F_IN], 1.5)
        set_number(nodes[F_OUT], 1.5)

    elif stem == 'pass':
        props = {'###operand_property': 0.5, 'comparison_property': '>', 'self_compare': False,
                 'force_int': False}
        gate(nodes[GATE], 'pass', '>', init='pass > 0.5', props=props)
        gate(nodes[GATE2], 'pass', '>', init='pass > 0.5', props=props)
        set_comment(nodes[NOTE_OUT], 'only the tops of the waves get through')
        set_comment(nodes[NOTE_DRAG], 'drag this across 0.5')
        set_number(nodes[F_IN], 0.9)
        set_number(nodes[F_OUT], 0.9)

    for l in links:
        for end in ('source', 'dest'):
            if l[end + '_node'] in (GATE, GATE2):
                l[end + '_node_name'] = nodes[l[end + '_node']]['name']

    d['nodes'] = {str(i): n for i, n in enumerate(nodes.values())}
    d['links'] = {str(i): l for i, l in enumerate(links)}
    return d


def main():
    with open(os.path.join(HELP, 'increasing_help.json')) as f:
        template = json.load(f)
    for stem in ('decreasing', 'change', 'pass'):
        path = os.path.join(HELP, stem + '_help.json')
        with open(path, 'w') as f:
            json.dump(derive(template, stem), f, indent=4)
        print(path)


if __name__ == '__main__':
    main()
