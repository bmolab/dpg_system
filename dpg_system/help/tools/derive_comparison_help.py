"""Derive one help patch per comparison operator from the hand-made arithmetic_help.json.

    python3 dpg_system/help/tools/derive_comparison_help.py

The comparison nodes share the arithmetic nodes' shape - data at the left, an
operand at the right - so their pages reuse the same hand layout: a sine wave
tested against the operand and plotted, a number box tested into a toggle, and
two random arrays compared into a heat-map mask. comparison_help.json remains
the family overview. After running this, use

    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

Two things differ from arithmetic, both handled below:
- the operand inlet is named '' when the node is given an argument and
  'operand' when it is not, so cords into it are named per node;
- the result is true/false by default, which a plot or heat map cannot draw,
  so those demos set output_type to int or float.
"""
import copy, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HELP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from derive_arithmetic_help import (
    TITLE, TEXT, SIGNAL, SIG_OP, PLOT, SIG_NOTE, INT_IN, INT_OP, INT_OUT,
    OPERAND_FLOAT, RAND_A, RAND_B, MAP_A, MAP_B, MAP_OUT, NP_NOTE, NP_OP,
    HOLD, page, set_prop, set_comment, set_block, set_number, link)

OPS_FN = {'>': lambda a, b: a > b, '>=': lambda a, b: a >= b, '<': lambda a, b: a < b,
          '<=': lambda a, b: a <= b, '==': lambda a, b: a == b, '!=': lambda a, b: a != b}

ARRAYS = """Arrays and tensors are compared element by element, giving an array of results the
same shape - a MASK, 1 wherever the test passes. In the NumPy example at the bottom
left, two random arrays are compared cell by cell."""

ARRAYS_WHOLE = """Arrays and tensors are compared element by element, giving an array of results the 
same shape - a MASK, 1 wherever the test passes. In the NumPy example at the bottom 
left, two arrays of random whole numbers (0, 1 or 2) are compared cell by cell, 
so about {n}."""

OPERAND = """The value to compare against. \n""" + HOLD + """
Give it as an argument ("{e}") and this inlet becomes a number you can drag.
Leave the argument out and it is a plain inlet that must be fed from elsewhere -
as in the NumPy example, where the operand is a whole array."""

OUTPUT_TYPE = """output_type:
What the answer looks like: "bool" gives true or false (the default), "int" 1 or 0,
"float" 1.0 or 0.0. A plot, a heat map or an arithmetic node needs a number, which
is why the sine and NumPy demos are set to int and float."""

EVERY_INPUT = """A result is sent for EVERY input, not only when the answer changes. To react only
at the moment the test flips, follow it with change."""

SMOOTH = """On a smooth signal {a} and {b} look the same - they differ only at exactly the
operand, which a sine wave almost never hits. The difference matters with whole
numbers: the number box starts at 10, the operand itself, where {a} 10 is {ra}
but {b} 10 is {rb}."""


def comparison_page(label, intro, examples, in_text, result_text, notes, see_also):
    return page(intro, label + ' <operand: int or float>', examples, in_text,
                OPERAND.format(e=label + (' 3' if label in ('==', '!=') else ' 0.5')),
                result_text + '\n\n' + OUTPUT_TYPE, notes, see_also)


OPS = [
    dict(stem='greater', label='>', title='> (greater than)',
         sig=dict(operand=0.5, note='1 while the sine wave is above 0.5\ndrag the operand to change the pulse width'),
         int=dict(value=10, operand=10),
         text=comparison_page('>', """
> tests whether each value arriving at its left inlet is GREATER than the operand,
and sends out the answer.

7 into "> 5" is true; 5 into "> 5" is false - equal is not greater.
In the demo a sine wave is tested against 0.5, and the plot shows 1 while the wave
is above it. Raise the operand and the pulses get narrower; lower it and they widen.

Use it as a threshold: is the hand above the head, is the sound loud enough, has
the count passed 100. Feed the answer to a gate or a toggle.

""" + ARRAYS, """
> 0.5     is a 0 to 1 control past halfway?
> 100     has the count passed 100?""",
             'The value to test. Receiving data here triggers the comparison.',
             'True if the input is greater than the operand.',
             [SMOOTH.format(a='>', b='>=', ra='false', rb='true'), EVERY_INPUT],
             """
>=      greater than or equal to
<       less than
change  react only when the answer changes""")),

    dict(stem='greater_equal', label='>=', title='>= (greater than or equal)',
         sig=dict(operand=0.5, note='1 while the sine wave is at or above 0.5\ndrag the operand to change the pulse width'),
         int=dict(value=10, operand=10),
         text=comparison_page('>=', """
>= tests whether each value arriving at its left inlet is GREATER THAN OR EQUAL TO
the operand, and sends out the answer.

7 into ">= 5" is true, and so is 5 - unlike >, the operand itself passes.
In the demo a sine wave is tested against 0.5, and the plot shows 1 while the wave
is at or above it. Raise the operand and the pulses get narrower.

Use it when reaching the threshold counts: "at least 3 people", "level 10 or more".
With whole numbers ">= 6" is the same test as "> 5"; choose the one that says what
you mean.

""" + ARRAYS, """
>= 0.0    is the value zero or positive?
>= 3      at least three""",
             'The value to test. Receiving data here triggers the comparison.',
             'True if the input is greater than or equal to the operand.',
             [SMOOTH.format(a='>', b='>=', ra='false', rb='true'), EVERY_INPUT],
             """
>       strictly greater than
<       less than - the exact opposite of >=
change  react only when the answer changes""")),

    dict(stem='less', label='<', title='< (less than)',
         sig=dict(operand=0.5, note='1 while the sine wave is below 0.5\ndrag the operand to change the pulse width'),
         int=dict(value=10, operand=10),
         text=comparison_page('<', """
< tests whether each value arriving at its left inlet is LESS than the operand,
and sends out the answer.

3 into "< 5" is true; 5 into "< 5" is false - equal is not less.
In the demo a sine wave is tested against 0.5, and the plot shows 1 while the wave
is below it. Raise the operand and the pulses get wider; lower it and they narrow.

Use it to notice when something drops away: the body has gone still (speed < 0.05),
the sound has gone quiet, the dancer has left the stage area.

""" + ARRAYS, """
< 0.05    has the movement all but stopped?
< 0       is the value negative?""",
             'The value to test. Receiving data here triggers the comparison.',
             'True if the input is less than the operand.',
             [SMOOTH.format(a='<', b='<=', ra='false', rb='true'), EVERY_INPUT],
             """
<=      less than or equal to
>=      greater than or equal - the exact opposite of <
change  react only when the answer changes""")),

    dict(stem='less_equal', label='<=', title='<= (less than or equal)',
         sig=dict(operand=0.5, note='1 while the sine wave is at or below 0.5\ndrag the operand to change the pulse width'),
         int=dict(value=10, operand=10),
         text=comparison_page('<=', """
<= tests whether each value arriving at its left inlet is LESS THAN OR EQUAL TO
the operand, and sends out the answer.

3 into "<= 5" is true, and so is 5 - unlike <, the operand itself passes.
In the demo a sine wave is tested against 0.5, and the plot shows 1 while the wave
is at or below it.

Use it when the limit itself is allowed: "no more than 4 voices", "at most 1.0".
With whole numbers "<= 4" is the same test as "< 5"; choose the one that says what
you mean.

""" + ARRAYS, """
<= 4      no more than four
<= 0.0    zero or below""",
             'The value to test. Receiving data here triggers the comparison.',
             'True if the input is less than or equal to the operand.',
             [SMOOTH.format(a='<', b='<=', ra='false', rb='true'), EVERY_INPUT],
             """
<       strictly less than
>       greater than - the exact opposite of <=
change  react only when the answer changes""")),

    dict(stem='equal', label='==', title='== (equal to)',
         sig=dict(operand=2, pre='// 0.25', box='int',
                  note='a sine wave counted in steps of 0.25 (-4 to 3), tested == 2\n1 only while the count is exactly 2'),
         int=dict(value=10, operand=10),
         whole=True,
         text=comparison_page('==', """
== tests whether each value arriving at its left inlet is EXACTLY EQUAL to the
operand, and sends out the answer.

7 into "== 7" is true; 7.001 is false.
Use it to pick out one particular value: the third item in a menu, step 5 of a
sequence, the moment a counter reaches 16.

EXACTLY means exactly. Fractional numbers are rarely exactly equal - 0.1 + 0.2 is
not == 0.3 - and a smooth signal sampled 60 times a second essentially never lands
on a given value. So the demo first counts the sine wave in whole steps with
"// 0.25", and tests the count. Compare whole numbers, round first, or test a range
with >= and < instead.

""" + ARRAYS_WHOLE.format(n='one cell in three matches'), """
== 3      is the third item selected?
== 0      is it exactly zero?""",
             'The value to test. Receiving data here triggers the comparison.',
             'True if the input is exactly equal to the operand.',
             [EVERY_INPUT],
             """
!=      the opposite test - anything but the operand
select  a separate outlet for each of several values
change  react only when the answer changes""")),

    dict(stem='not_equal', label='!=', title='!= (not equal to)',
         sig=dict(operand=2, pre='// 0.25', box='int',
                  note='a sine wave counted in steps of 0.25 (-4 to 3), tested != 2\n1 except while the count is exactly 2'),
         int=dict(value=10, operand=10),
         whole=True,
         text=comparison_page('!=', """
!= tests whether each value arriving at its left inlet is NOT EQUAL to the operand,
and sends out the answer. It is always the opposite of ==.

7 into "!= 7" is false; anything else is true.
Use it for "anything but": anything but the resting value, anything but the first
item, anything but zero ("!= 0" is a common way to ask whether something is happening).

Because it is the opposite of ==, the same caution applies the other way round:
fractional numbers are almost never exactly equal, so on a smooth signal != is
almost always true. The demo counts the sine wave in whole steps first, and the
plot dips to 0 only while the count is exactly 2.

""" + ARRAYS_WHOLE.format(n='two cells in three differ'), """
!= 0      is anything happening at all?
!= 1      anything but the first item""",
             'The value to test. Receiving data here triggers the comparison.',
             'True if the input differs from the operand.',
             [EVERY_INPUT],
             """
==      the opposite test - exactly the operand
change  react only when the answer changes""")),
]


def output_type(node, value):
    """A comparison node saves no operand (it comes from the init), only output_type."""
    node['properties'] = {'0': {'name': 'output_type', 'value': value, 'value_type': 'string'}}


def derive(template, op):
    d = copy.deepcopy(template)
    d['name'] = op['stem'] + '_help'
    nodes = {n['id']: n for n in d['nodes'].values()}
    links = list(d['links'].values())
    next_id = max(nodes) + 1
    next_sid = max(n.get('sid', 0) for n in nodes.values()) + 1
    label, sig, num = op['label'], op['sig'], op['int']

    set_comment(nodes[TITLE], op['title'])
    set_block(nodes[TEXT], op['text'], '30')
    nodes[TEXT]['position_x'], nodes[TEXT]['position_y'] = 729, 45

    # sine demo: tested against the operand; int output so the plot can draw it
    sop = nodes[SIG_OP]
    sop['name'], sop['init'] = label, f'{label} {sig["operand"]}'
    output_type(sop, 'int')
    set_prop(nodes[PLOT], 'min y', -0.2)
    set_prop(nodes[PLOT], 'max y', 1.2)
    set_block(nodes[SIG_NOTE], sig['note'], '24')
    if sig.get('box') == 'int':
        # a whole-number operand wants an int box to drive it
        box = copy.deepcopy(nodes[INT_IN])
        box['id'], box['sid'] = OPERAND_FLOAT, nodes[OPERAND_FLOAT]['sid']
        box['position_x'] = nodes[OPERAND_FLOAT]['position_x']
        box['position_y'] = nodes[OPERAND_FLOAT]['position_y']
        nodes[OPERAND_FLOAT] = box
        for l in links:
            if l['source_node'] == OPERAND_FLOAT:
                l['source_node_name'], l['source_output_name'] = 'int', 'int out'
    set_number(nodes[OPERAND_FLOAT], sig['operand'])
    if 'pre' in sig:
        # signal -> pre -> test, so == and != see whole-number steps
        pre = copy.deepcopy(nodes[SIG_OP])
        pre['id'], pre['sid'] = next_id, next_sid
        pre['name'], pre['init'] = sig['pre'].split()[0], sig['pre']
        pre['properties'] = {'0': {'name': 'operand', 'value': float(sig['pre'].split()[1]),
                                   'value_type': 'float'}}
        pre['position_x'], pre['position_y'] = 333, 92
        sop['position_y'] = 165
        nodes[pre['id']] = pre
        for l in links:
            if l['source_node'] == SIGNAL and l['dest_node'] == SIG_OP:
                l['dest_node'], l['dest_node_name'] = pre['id'], pre['name']
        links.append(link(pre['id'], pre['name'], 0, 'result', SIG_OP, label, 0, 'in'))

    # number demo: the answer shown as a toggle
    iop = nodes[INT_OP]
    iop['name'], iop['init'] = label, f'{label} {num["operand"]}'
    output_type(iop, 'bool')
    set_number(nodes[INT_IN], num['value'])
    old = nodes[INT_OUT]
    nodes[INT_OUT] = {
        'name': 'toggle', 'id': INT_OUT, 'sid': old['sid'],
        'position_x': old['position_x'], 'position_y': old['position_y'],
        'width': 45, 'height': 42, 'visibility': 'show_all', 'draggable': True,
        'presentation_state': 'show_all',
        'properties': {'0': {'name': '', 'value': bool(OPS_FN[label](num['value'], num['operand'])),
                             'value_type': 'bool'}}}
    for l in links:
        if l['dest_node'] == INT_OUT:
            l['dest_node_name'], l['dest_input_name'] = 'toggle', ''

    # NumPy demo: two arrays compared into a 0/1 mask
    nop = nodes[NP_OP]
    nop['name'] = label
    nop.pop('init', None)
    output_type(nop, 'float')
    set_comment(nodes[NP_NOTE], 'comparing two arrays cell by cell gives a mask: 1 where the test is true')
    in_max = 1.0
    if op.get('whole'):
        # random float arrays would never match: use whole numbers 0, 1, 2
        in_max = 2.0
        for r in (RAND_A, RAND_B):
            set_prop(nodes[r], 'dtype', 'int64')
            set_prop(nodes[r], 'min', 0.0)
            set_prop(nodes[r], 'max', 2.0)
    for m in (MAP_A, MAP_B):
        set_prop(nodes[m], 'min y', 0.0)
        set_prop(nodes[m], 'max y', in_max)
    set_prop(nodes[MAP_OUT], 'min y', 0.0)
    set_prop(nodes[MAP_OUT], 'max y', 1.0)

    # cords into a comparison node: its operand inlet is '' with an argument,
    # 'operand' without one
    for l in links:
        for end in ('source', 'dest'):
            if l[end + '_node'] in (SIG_OP, INT_OP, NP_OP):
                l[end + '_node_name'] = label
        if l['dest_node'] in (SIG_OP, INT_OP, NP_OP) and l['dest_input_index'] == 1:
            l['dest_input_name'] = 'operand' if l['dest_node'] == NP_OP else ''

    d['nodes'] = {str(i): n for i, n in enumerate(nodes.values())}
    d['links'] = {str(i): l for i, l in enumerate(links)}
    return d


def main():
    with open(os.path.join(HELP, 'arithmetic_help.json')) as f:
        template = json.load(f)
    for op in OPS:
        path = os.path.join(HELP, op['stem'] + '_help.json')
        with open(path, 'w') as f:
            json.dump(derive(template, op), f, indent=4)
        print(path)


if __name__ == '__main__':
    main()
