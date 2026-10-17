"""Derive one help patch per colour conversion from the hand-made color_convert_help.json.

    python3 dpg_system/help/tools/derive_color_convert_help.py

color_convert_help.json is laid out by hand and is the overview of the family.
This copies its layout into a page per named conversion (rgb_to_hsl_help.json,
cmy_to_hsv_help.json, ...): a saw sweeps one component of a colour in the FROM
space, pack completes it, the node converts it, and vectors show both ends.

The colour swatch always shows the colour itself, which needs rgb: it hangs off
the converter when the page converts TO rgb, off the input when it converts FROM
rgb, and otherwise off a second, small <from>_to_rgb node beside the input.

It is deliberately NOT a make_*_help.py: rebuild.py re-flows those. After running
it, use

    python3 dpg_system/help/tools/relayout.py --minimal dpg_system/help/<stem>_help.json

The template is read by node id, so the ids below must match color_convert_help.json.
"""
import colorsys, copy, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HELP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from derive_arithmetic_help import set_prop, set_comment, set_block, link

# node ids in the template
TITLE, TEXT, SIGNAL, CONV, NOTE_OUT, CLOSE = 1414, 1425, 1479, 1539, 1600, 1617
SWATCH, PACK, VEC_IN, VEC_OUT, NOTE_IN, CONVERT_NOTE = 1651, 1687, 1713, 1898, 2080, 2105

WORDS = {'rgb': 'red, green, blue', 'cmy': 'cyan, magenta, yellow',
         'hsl': 'hue, saturation, lightness', 'hsv': 'hue, saturation, value'}

SPACE = {
    'rgb': """rgb   red, green, blue - how a screen makes light.
      Natural for output, awkward for adjustment: there is no component
      that means "brightness" or "how vivid".""",
    'cmy': """cmy   cyan, magenta, yellow - the inverse of rgb, how inks subtract light.
      Each value is simply one minus the rgb one.""",
    'hsl': """hsl   hue, saturation, lightness. Hue is the colour itself as an angle
      round the wheel, saturation is how vivid, lightness runs black to white
      with the pure colour at the middle.""",
    'hsv': """hsv   hue, saturation, value. Like hsl, but value runs black to the pure
      colour rather than through to white. Handy when you think of it as
      "how much light is on this". Design tools often call it hsb.""",
}

# one orange, written in each space, for the worked example
ORANGE = {'rgb': [1.0, 0.5, 0.0], 'cmy': [0.0, 0.5, 1.0], 'hsl': [30, 1.0, 0.5], 'hsv': [30, 1.0, 1.0]}

# the demo in each FROM space: the saw sweeps the first component over (0, range); the
# other two are held at pack's arguments; the note above it says so
SWEEP = {
    'hsl': (360.0, (0.8, 0.5), 'This creates a colour of saturation 0.8 and lightness 0.5\n'
                               'with the hue sweeping around the 360 degrees of the colour wheel'),
    'hsv': (360.0, (0.8, 0.9), 'This creates a colour of saturation 0.8 and value 0.9\n'
                               'with the hue sweeping around the 360 degrees of the colour wheel'),
    'rgb': (1.0, (0.4, 0.8), 'This creates a colour of green 0.4 and blue 0.8\n'
                             'with red sweeping from 0 to 1'),
    'cmy': (1.0, (0.2, 0.8), 'This creates a colour of magenta 0.2 and yellow 0.8\n'
                             'with cyan sweeping from 0 to 1'),
}

# per conversion: what to watch in the output vector, and what it is for
PAIRS = {
    ('hsl', 'rgb'): ('red, green and blue rise and fall in turn\nset out scale to 0-255 for byte values',
                     'Use it to build colours by hue - sweep round the wheel, drain the saturation,\n'
                     'fade the lightness - and then send them to a screen, a light or a color node.'),
    ('hsv', 'rgb'): ('red, green and blue rise and fall in turn\nset out scale to 0-255 for byte values',
                     'Use it to build colours by hue - sweep round the wheel, drain the saturation,\n'
                     'dim the value - and then send them to a screen, a light or a color node.'),
    ('hsl', 'cmy'): ('cyan, magenta and yellow rise and fall in turn,\neach the opposite of red, green or blue',
                     'Use it to drive something that mixes colour subtractively - inks, or a light\n'
                     'with cyan, magenta and yellow filters - by hue.'),
    ('hsv', 'cmy'): ('cyan, magenta and yellow rise and fall in turn,\neach the opposite of red, green or blue',
                     'Use it to drive something that mixes colour subtractively - inks, or a light\n'
                     'with cyan, magenta and yellow filters - by hue.'),
    ('hsl', 'hsv'): ('the hue passes through unchanged; the same colour\nis saturation 0.89, value 0.9 in hsv',
                     'Both describe a colour by its hue; they differ in what the other two numbers\n'
                     'mean. Use it when a colour comes as hsl and something wants hsv (or hsb).'),
    ('hsv', 'hsl'): ('the hue passes through unchanged; the same colour\nis saturation 0.78, lightness 0.54 in hsl',
                     'Both describe a colour by its hue; they differ in what the other two numbers\n'
                     'mean. Use it when a colour comes as hsv (or hsb) and something wants hsl.'),
    ('rgb', 'hsl'): ('the hue moves from blue (210) to pink (320);\nsaturation dips to 0.5 in the middle',
                     'Use it to take a colour from a camera, an image or a colour picker and get at\n'
                     'its hue, how vivid it is, and how light - to change one and convert back.'),
    ('rgb', 'hsv'): ('the hue moves from blue (210) to pink (320);\nvalue holds at 0.8 until red passes 0.8',
                     'Use it to take a colour from a camera, an image or a colour picker and get at\n'
                     'its hue, how vivid it is, and how bright - to change one and convert back.'),
    ('rgb', 'cmy'): ('cyan falls as red rises: each is one minus the other\nmagenta and yellow hold at 0.6 and 0.2',
                     'Use it for anything that thinks in inks or subtractive filters. It is the\n'
                     'simplest conversion there is: each value is one minus the rgb one.'),
    ('cmy', 'rgb'): ('red falls as cyan rises: each is one minus the other\ngreen and blue hold at 0.8 and 0.2',
                     'Use it to show or send a cmy colour as light. It is the simplest conversion\n'
                     'there is: each value is one minus the cmy one.'),
    ('cmy', 'hsl'): ('the hue moves from orange (45) to green (135);\nlightness stays near 0.5',
                     'Use it to get at the hue, vividness and lightness of a colour that arrives\n'
                     'as cyan, magenta and yellow.'),
    ('cmy', 'hsv'): ('the hue moves from orange (45) to green (135);\nvalue holds at 0.8 once cyan passes 0.2',
                     'Use it to get at the hue, vividness and brightness of a colour that arrives\n'
                     'as cyan, magenta and yellow.'),
}

SCALES = """in scale / out scale:
What numeric range the values use, INDEPENDENTLY at each end:

  0-1     0.0 to 1.0, the usual choice inside a patch
  0-100   percentages, as design tools tend to use
  0-255   bytes, as image files and many devices use

Because the two are separate, the node also does the rescaling for you:
set in scale to 0-255 and out scale to 0-1 to bring an image's colours into
patch range in the same move as the conversion."""

HUE = """HUE IS ALWAYS IN DEGREES:
The scale settings apply to every component EXCEPT hue, which is always 0 to
360 whatever the scale says. Hue is an angle, not a proportion. An {s} value
like [210, 0.5, 0.4] with a 0-1 scale is correct, not a bug."""

OUT_OF_RANGE = """OUT OF RANGE:
Values are converted through rgb, and red, green and blue are held to 0 to 1
on the way, so a component pushed past its range gives the nearest colour
that exists rather than an error."""


def to_rgb(v, space):
    a, b, c = v
    if space == 'rgb':
        return a, b, c
    if space == 'cmy':
        return 1 - a, 1 - b, 1 - c
    if space == 'hsl':
        return colorsys.hls_to_rgb(a / 360, c, b)
    return colorsys.hsv_to_rgb(a / 360, b, c)


def from_rgb(rgb, space):
    if space == 'rgb':
        return tuple(rgb)
    if space == 'cmy':
        return tuple(1 - x for x in rgb)
    if space == 'hsl':
        h, l, s = colorsys.rgb_to_hls(*rgb)
        return h * 360, s, l
    return (lambda h, s, v: (h * 360, s, v))(*colorsys.rgb_to_hsv(*rgb))


def fmt(v):
    return '[' + ', '.join(f'{round(x, 2):g}' for x in v) + ']'


def prose(src, dst):
    name = f'{src}_to_{dst}'
    example_out = from_rgb(to_rgb(ORANGE[src], src), dst)
    parts = [f"""{name} converts a colour described as {WORDS[src]}
into the same colour described as {WORDS[dst]}.

An orange of {fmt(ORANGE[src])} in {src} comes out as {fmt(example_out)} in {dst}.
In the demo, one component of the {src} colour sweeps steadily while the other two
hold still; the vectors show the colour going in and coming out, and the swatch
shows the colour itself""" + (f' (through a second, small {src}_to_rgb node beside the input, as\n'
                              'a swatch needs red, green and blue).' if 'rgb' not in (src, dst) else '.') + """

""" + PAIRS[(src, dst)][1],
             'THE TWO SPACES:\n\n' + SPACE[src] + '\n' + SPACE[dst],
             f'SYNTAX:\n{name}',
             f'EXAMPLE:\n{name}',
             """INPUTS and PARAMETERS:

in:
Three numbers - a list, or a NumPy array - in the """ + src + """ space. Receiving data
here triggers the conversion.

from / to:
The spaces being converted between: rgb, cmy, hsl or hsv. The node's name sets
them to """ + src + ' and ' + dst + """; the menus can repoint it to any other pair
without replacing the node.

""" + SCALES,
             'OUTPUTS:\n\nout:\nThe converted colour: three numbers (a NumPy array) in the ' + dst
             + ' space and the out scale.']
    if 'hsl' in (src, dst) or 'hsv' in (src, dst):
        parts.append(HUE.format(s=next(s for s in (dst, src) if s in ('hsl', 'hsv'))))
    parts.append(OUT_OF_RANGE)
    others = [f'{dst}_to_{src}    the reverse conversion']
    others += [f'{src}_to_{o}    {src} to {WORDS[o]}' for o in ('rgb', 'cmy', 'hsl', 'hsv')
               if o not in (src, dst)]
    others += ['color_convert    choose both spaces yourself',
               'color            pick a colour, or show one']
    parts.append('SEE ALSO:\n' + '\n'.join(others))
    return '\n\n'.join(parts)


def set_vector(node, values):
    for i, v in enumerate(values):
        set_prop(node, f'[{i}]', float(v))


def converter(node, src, dst):
    node['name'] = f'{src}_to_{dst}'
    node.pop('init', None)
    set_prop(node, 'from', src)
    set_prop(node, 'to', dst)


def derive(template, src, dst):
    d = copy.deepcopy(template)
    name = f'{src}_to_{dst}'
    d['name'] = name + '_help'
    nodes = {n['id']: n for n in d['nodes'].values()}
    links = list(d['links'].values())
    next_id = max(nodes) + 1
    next_sid = max(n.get('sid', 0) for n in nodes.values()) + 1

    set_comment(nodes[TITLE], f'{name} ({WORDS[src]} to {WORDS[dst]})')
    set_block(nodes[TEXT], prose(src, dst), '30')

    rng, held, top = SWEEP[src]
    set_prop(nodes[SIGNAL], 'range', rng)
    nodes[PACK]['init'] = f'pack f {held[0]} {held[1]}'
    set_block(nodes[NOTE_IN], top, '24')

    converter(nodes[CONV], src, dst)
    set_comment(nodes[CONVERT_NOTE], 'convert to ' + dst)
    set_block(nodes[NOTE_OUT], PAIRS[(src, dst)][0], '24')

    # what the boxes show on opening: the colour a third of the way through the sweep
    colour_in = (rng / 3, held[0], held[1])
    rgb = to_rgb(colour_in, src)
    set_vector(nodes[VEC_IN], colour_in)
    set_vector(nodes[VEC_OUT], from_rgb(rgb, dst))
    set_prop(nodes[SWATCH], 'color', [c * 255 for c in rgb] + [255.0])

    # the swatch needs rgb
    links = [l for l in links if l['dest_node'] != SWATCH]
    if dst == 'rgb':
        links.append(link(CONV, name, 0, 'out', SWATCH, 'color', 0, '##color'))
    elif src == 'rgb':
        links.append(link(PACK, 'pack', 0, 'out', SWATCH, 'color', 0, '##color'))
    else:
        helper = copy.deepcopy(nodes[CONV])
        helper['id'], helper['sid'] = next_id, next_sid
        converter(helper, src, 'rgb')
        helper['position_x'], helper['position_y'] = 410, 230
        nodes[helper['id']] = helper
        links.append(link(PACK, 'pack', 0, 'out', helper['id'], helper['name'], 0, 'in'))
        links.append(link(helper['id'], helper['name'], 0, 'out', SWATCH, 'color', 0, '##color'))

    for l in links:
        for end in ('source', 'dest'):
            if l[end + '_node'] == CONV:
                l[end + '_node_name'] = name

    d['nodes'] = {str(i): n for i, n in enumerate(nodes.values())}
    d['links'] = {str(i): l for i, l in enumerate(links)}
    return d


PAIRS_ORDER = [(s, t) for s in ('rgb', 'cmy', 'hsl', 'hsv') for t in ('rgb', 'cmy', 'hsl', 'hsv') if s != t]


def main():
    with open(os.path.join(HELP, 'color_convert_help.json')) as f:
        template = json.load(f)
    for src, dst in PAIRS_ORDER:
        path = os.path.join(HELP, f'{src}_to_{dst}_help.json')
        with open(path, 'w') as f:
            json.dump(derive(template, src, dst), f, indent=4)
        print(path)


if __name__ == '__main__':
    main()
