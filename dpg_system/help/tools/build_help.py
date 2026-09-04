"""Build a dpg_system help patcher from a compact spec.

A help patch is: a title comment (top left), a locked text_block of prose
(right hand column), a small live demo made of real nodes (left column),
and a close button at the bottom left.
"""
import json, os

HELP_DIR = '/Users/drokeby/dpg_system/dpg_system/help'

TEXT_X_PAD = 22          # gap between demo column and text block
TEXT_SIZE = '30'
TITLE_SIZE = '48'

# On-screen metrics of the app's font, measured in the running app: macOS loads
# Inconsolata-g at 24pt with global_font_scale 0.5, so a '24' line is 12px high
# and one glyph advances 6.16px. Needed because a text_block is a FIXED box --
# too narrow and TextEditor re-wraps the text, too short and it scrolls.
#
# '24' is the odd one out and has to stay first in the table: it is the
# text_size option's default, so restoring it is a no-op and the font callback
# never fires (see the load-callback trap), leaving the widget on the app's
# bound font -- which is Inconsolata 24 at scale 0.5 anyway, so the numbers
# agree. The other sizes get their font applied for real.
LINE_H = {'24': 12, '30': 15, '36': 18, '48': 24}
GLYPH_W = {'24': 6.16, '30': 7.68, '36': 9.22, '48': 12.29}
BLOCK_PAD_W = 26         # what TextEditor._wrap_columns reserves for frame
                         # padding and scrollbar, plus a couple of pixels


def annotation_box(text, size='24'):
    """Widget width and height that fit a multi-line annotation exactly.

    The font is monospace, so this is arithmetic rather than a guess: the box
    is as wide as the longest line and as tall as the line count. Measured in
    the app -- three 12px lines need exactly 36px of height, with no padding
    to add.
    """
    lines = text.split('\n')
    glyph = GLYPH_W.get(size, GLYPH_W['24'])
    line_h = LINE_H.get(size, LINE_H['24'])
    width = int(max(len(l) for l in lines) * glyph) + BLOCK_PAD_W
    return width, len(lines) * line_h


def _props(d):
    """dict -> the numbered properties container the loader expects."""
    out = {}
    for i, (k, v) in enumerate(d.items()):
        t = type(v).__name__
        if v is None:
            t = 'NoneType'
        out[str(i)] = {'name': k, 'value': v, 'value_type': t}
    return out


def build(name, title, body, demo, links, demo_width=520,
          text_width=780, text_height=620, out_dir=HELP_DIR, subpatch=None):
    """
    name        file stem, e.g. 'sample_hold'  -> sample_hold_help.json
    title       big comment across the top
    body        the prose (a single string)
    demo        list of dicts: {key, init, pos:(x,y), props:{}, comment:False}
    links       list of (src_key, src_out_name, dst_key, dst_in_name)
    subpatch    optional {'name', 'host': demo key of the p/patcher node,
                'demo': [...], 'links': [...]} -- writes the multi-patch file
                format, so the page can ship a subpatcher that really opens.
    """
    nodes = {}
    ids = {}
    nid = [200]

    def new_id():
        nid[0] += 37
        return nid[0]

    # 0: the origin node every patch carries
    nodes['0'] = {
        'name': '', 'id': new_id(), 'position_x': 0, 'position_y': 0,
        'width': 9, 'height': 30, 'visibility': 'show_all', 'draggable': True,
        'protected': True, 'presentation_state': 'hidden',
        'properties': {'0': {'name': '', 'value': None, 'value_type': 'NoneType'}},
    }
    n = 1

    def add(entry):
        nonlocal n
        nodes[str(n)] = entry
        n += 1

    # title
    add({'init': 'comment ' + title, 'name': 'comment', 'id': new_id(),
         'position_x': 24, 'position_y': -4, 'width': 12 * len(title), 'height': 42,
         'visibility': 'show_all', 'draggable': True, 'presentation_state': 'show_all',
         'properties': _props({'text': title, 'font size': TITLE_SIZE}),
         'comment': title})

    text_x = demo_width + TEXT_X_PAD
    # prose
    add({'name': 'text_block', 'id': new_id(),
         'position_x': text_x, 'position_y': 45,
         'width': text_width + 8, 'height': text_height + 16,
         'visibility': 'show_all', 'draggable': True, 'presentation_state': 'show_all',
         'properties': _props({'block': body, 'lock': True,
                               'width': text_width, 'height': text_height,
                               'text_size': TEXT_SIZE}),
         'text': body})

    # demo nodes
    for d in demo:
        i = new_id()
        ids[d['key']] = i
        if d.get('comment'):
            txt = d['text']
            size = d.get('size', '24')
            if '\n' in txt:
                # A paragraph goes in ONE locked text_block, not a stack of
                # comment nodes. A comment holds a single line -- its text
                # lives in a single-line input_text option that strips returns
                # on the way in -- so a stack was the only way to get a second
                # line, and each node then paid for its own title bar and
                # padding: 30px of node for a 12px line. A text_block keeps the
                # returns, draws just as transparently, and spaces the lines at
                # the font's own line height.
                w, h = annotation_box(txt, size)
                add({'name': 'text_block', 'id': i, 'annotation': True,
                     'position_x': d['pos'][0], 'position_y': d['pos'][1],
                     'width': w + 8, 'height': h + 16, 'visibility': 'show_all',
                     'draggable': True, 'presentation_state': 'show_all',
                     'properties': _props({'block': txt, 'lock': True,
                                           'width': w, 'height': h,
                                           'text_size': size}),
                     'text': txt})
                continue
            add({'init': 'comment ' + txt, 'name': 'comment', 'id': i,
                 'position_x': d['pos'][0], 'position_y': d['pos'][1],
                 'width': 7 * len(txt), 'height': 30, 'visibility': 'show_all',
                 'draggable': True, 'presentation_state': 'show_all',
                 'properties': _props({'text': txt, 'font size': size}),
                 'comment': txt})
            continue
        init = d['init']
        entry = {'name': init.split(' ')[0], 'id': i,
                 'position_x': d['pos'][0], 'position_y': d['pos'][1],
                 'width': d.get('w', 130), 'height': d.get('h', 70),
                 'visibility': 'show_all', 'draggable': True,
                 'presentation_state': 'show_all',
                 'properties': _props(d.get('props', {}))}
        if ' ' in init:
            entry['init'] = init
        add(entry)

    # close button, bottom of the demo column
    close_y = max([d['pos'][1] for d in demo] + [0]) + 110
    add({'name': 'close', 'id': new_id(), 'position_x': 24, 'position_y': close_y,
         'width': 88, 'height': 44, 'visibility': 'show_all', 'draggable': True,
         'presentation_state': 'show_all',
         'properties': _props({'close patch': None})})

    lk = {}
    for j, link in enumerate(links):
        # (src, outlet_name, dst, inlet_name) and optionally the outlet index,
        # then the inlet index. Names are enough for most nodes -- the loader
        # searches by name -- but a node whose outlets are all unnamed
        # (dict_extract, repeat) can only be addressed by index.
        sk, so, dk, di = link[:4]
        so_index = link[4] if len(link) > 4 else 0
        di_index = link[5] if len(link) > 5 else 0
        lk[str(j)] = {
            'source_node': ids[sk], 'source_node_name': '',
            'source_output_index': so_index, 'source_output_name': so,
            'dest_node': ids[dk], 'dest_node_name': '',
            'dest_input_index': di_index, 'dest_input_name': di,
        }

    patch = {
        'height': max(close_y + 90, text_height + 110),
        'width': text_x + text_width + 40,
        'position': [100, 100],
        'id': new_id(),
        'name': name + '_help',
        'path': os.path.join(out_dir, name + '_help.json'),
        'nodes': nodes,
        'links': lk,
    }
    path = os.path.join(out_dir, name + '_help.json')

    if subpatch is None:
        json.dump(patch, open(path, 'w'), indent=4)
        return path

    # A subpatcher is two patches in one file, linked both ways: the parent
    # node's 'patcher id' is the subpatch's id, and the subpatch's
    # 'parent_node_uuid' is the parent node's id.
    host_id = ids[subpatch['host']]
    sub_id = new_id()
    for nc in nodes.values():
        if nc.get('id') == host_id:
            nc['patcher id'] = sub_id
            break

    sub_nodes, sub_ids = {}, {}
    sub_nodes['0'] = {
        'name': '', 'id': new_id(), 'position_x': 0, 'position_y': 0,
        'width': 9, 'height': 30, 'visibility': 'show_all', 'draggable': True,
        'protected': True, 'presentation_state': 'hidden',
        'properties': {'0': {'name': '', 'value': None, 'value_type': 'NoneType'}},
    }
    for i, d in enumerate(subpatch['demo'], start=1):
        nid_ = new_id()
        sub_ids[d['key']] = nid_
        x, y = d['pos']
        sub_nodes[str(i)] = {
            'name': d['init'].split(' ')[0], 'init': d['init'], 'id': nid_,
            'position_x': x, 'position_y': y,
            'width': d.get('w', 140), 'height': d.get('h', 60),
            'visibility': 'show_all', 'draggable': True,
            'presentation_state': 'show_all',
            'properties': _props(d.get('props', {})),
        }
    sub_lk = {}
    for j, link in enumerate(subpatch.get('links', [])):
        sk, so, dk, di = link[:4]
        sub_lk[str(j)] = {
            'source_node': sub_ids[sk], 'source_node_name': '',
            'source_output_index': link[4] if len(link) > 4 else 0,
            'source_output_name': so,
            'dest_node': sub_ids[dk], 'dest_node_name': '',
            'dest_input_index': link[5] if len(link) > 5 else 0,
            'dest_input_name': di,
        }
    sub = {
        'height': max([d['pos'][1] + d.get('h', 60) for d in subpatch['demo']]) + 80,
        'width': max([d['pos'][0] + d.get('w', 140) for d in subpatch['demo']]) + 80,
        'position': [160, 160],
        'id': sub_id,
        'name': subpatch['name'],
        'parent_node_uuid': host_id,
        'nodes': sub_nodes,
        'links': sub_lk,
    }
    doc = {'name': name + '_help', 'path': path,
           'patches': {'0': sub, '1': patch}}
    json.dump(doc, open(path, 'w'), indent=4)
    return path
