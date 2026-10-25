"""Remove the load_bang -> t 1 starters that only switch signal nodes on,
and the toggles that only switch a metro on.

    python3 dpg_system/help/tools/strip_starters.py [--dry-run] [--toggles] dpg_system/help/*_help.json

A signal saved with 'on' true starts by itself when the patch loads, so the
load_bang -> t 1 -> signal 'on' chain is clutter. A t 1 is removed only when
every cord out of it goes to a signal's 'on' inlet; a load_bang only when
everything it fed has been removed. Each signal it was starting is saved with
on = true. A metro saves its own on/off too, so a toggle whose only cord goes
to a metro's 'on' is removed as well (the metro keeps its saved state).
build() applies both rules to generated pages, unless told keep_toggles (the
toggle page, where the toggle is the point).

If the removed nodes sat in a band of their own across the demo, the demo is
moved up to close the gap.
"""
import json, sys

TOP = 40          # above this is the title
GAP = 12          # space kept between what was above the band and the demo


def _label(n):
    return (n.get('init') or n.get('name') or '').split(' ')[0] or n.get('name', '')


def close_gap(keep, removed):
    """If the removed nodes sat in a band of their own across the demo, move
    the demo below that band up to close the gap."""
    top = min(n['position_y'] for n in removed)
    bottom = max(n['position_y'] + n.get('height', 40) for n in removed)
    # the page's prose is the largest text_block; small ones are part of the demo
    blocks = [n for n in keep if _label(n) == 'text_block']
    prose = max(blocks, key=lambda n: n.get('width', 0) * n.get('height', 0)) if blocks else None
    demo = [n for n in keep if n.get('name') and n['position_y'] >= TOP and n is not prose]
    if any(n['position_y'] < bottom and n['position_y'] + n.get('height', 40) > top for n in demo):
        return
    below = [n for n in demo if n['position_y'] >= bottom]
    if not below:
        return
    above = [n['position_y'] + n.get('height', 40) for n in keep
             if n.get('name') and n['position_y'] + n.get('height', 40) <= top]
    shift = min(n['position_y'] for n in below) - (max(above + [TOP]) + GAP)
    if shift > 0:
        for n in below:
            n['position_y'] -= shift


def strip_patch(p):
    """Strip starters from one patch dict in place; returns the labels removed."""
    if 'nodes' not in p:
        return []
    nodes = {n['id']: n for n in p['nodes'].values()}
    links = list(p.get('links', {}).values())
    outs = {}
    for l in links:
        outs.setdefault(l['source_node'], []).append(l)
    ins = {}
    for l in links:
        ins.setdefault(l['dest_node'], []).append(l)

    def is_signal_on(l):
        d = nodes.get(l['dest_node'])
        return d is not None and _label(d) == 'signal' and l['dest_input_name'] == 'on'

    dead = set()
    for nid, n in nodes.items():
        if n.get('init', '').strip() in ('t 1', 'trigger 1') and outs.get(nid) \
                and all(is_signal_on(l) for l in outs[nid]):
            dead.add(nid)
    for nid, n in nodes.items():
        if _label(n) == 'load_bang' and outs.get(nid) and all(l['dest_node'] in dead for l in outs[nid]):
            dead.add(nid)
    # a t whose only source is not a dead load_bang is fed by something else: keep it
    for nid in list(dead):
        if _label(nodes[nid]) == 't' and any(l['source_node'] not in dead for l in ins.get(nid, [])):
            dead.discard(nid)
    if not dead:
        return []

    for l in links:
        if l['source_node'] in dead and is_signal_on(l):
            for pr in nodes[l['dest_node']].get('properties', {}).values():
                if pr.get('name') == 'on':
                    pr['value'], pr['value_type'] = True, 'bool'

    removed = [nodes[i] for i in dead]
    keep = [n for i, n in nodes.items() if i not in dead]
    close_gap(keep, removed)

    p['nodes'] = {str(i): n for i, n in enumerate(keep)}
    p['links'] = {str(i): l for i, l in enumerate(
        l for l in links if l['source_node'] not in dead and l['dest_node'] not in dead)}
    return sorted(n.get('init') or n['name'] for n in removed)


def strip_toggles(p):
    """Remove a toggle whose only cord switches a metro on: a metro saves its
    own on/off and starts by itself. The metro keeps its saved state."""
    if 'nodes' not in p:
        return []
    nodes = {n['id']: n for n in p['nodes'].values()}
    links = list(p.get('links', {}).values())
    dead = set()
    for nid, n in nodes.items():
        if _label(n) != 'toggle' or any(l['dest_node'] == nid for l in links):
            continue
        outs = [l for l in links if l['source_node'] == nid]
        if len(outs) == 1 and _label(nodes[outs[0]['dest_node']]) == 'metro' \
                and outs[0]['dest_input_name'] == 'on':
            dead.add(nid)
    if not dead:
        return []
    keep = [n for i, n in nodes.items() if i not in dead]
    close_gap(keep, [nodes[i] for i in dead])
    p['nodes'] = {str(i): n for i, n in enumerate(keep)}
    p['links'] = {str(i): l for i, l in enumerate(
        l for l in links if l['source_node'] not in dead and l['dest_node'] not in dead)}
    return ['toggle'] * len(dead)


def strip_file(path, dry=False, keep_toggles=True):
    d = json.load(open(path))
    removed = []
    for p in (d['patches'].values() if 'patches' in d else [d]):
        removed += strip_patch(p)
        if not keep_toggles:
            removed += strip_toggles(p)
    if removed and not dry:
        with open(path, 'w') as f:
            json.dump(d, f, indent=4)
    return removed


def main():
    args = sys.argv[1:]
    dry = '--dry-run' in args
    toggles = '--toggles' in args       # opt in: the toggle and metro pages keep theirs
    total = 0
    for path in [a for a in args if not a.startswith('--')]:
        r = strip_file(path, dry, keep_toggles=not toggles)
        if r:
            total += 1
            print(f'{path.split("/")[-1]:40} {r}')
    print(f'{total} patches {"would change" if dry else "changed"}')


if __name__ == '__main__':
    main()
