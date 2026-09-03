"""Check a help patcher against the statically extracted node interfaces.

Catches, without launching the GUI:
  - node names that are not registered node types
  - link endpoints naming an inlet/outlet the node does not have
  - property names that match nothing on the node
  - links pointing at node ids not present in the file
"""
import json, sys, os

IFACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iface.json')


def load_iface():
    return json.load(open(IFACE))


def names(iface, label, kinds, strip=True):
    """Port names for a node.

    strip=True mirrors restore_properties(), which compares labels with a
    leading '#' guard removed. Link resolution in dpg_app does NOT strip, so
    pass strip=False when checking links -- otherwise a cord aimed at
    '###input' by the bare name 'input' validates here and then silently
    fails to connect on load.
    """
    v = iface.get(label)
    if v is None:
        return None
    ks = set(kinds) | {'alias'}   # a port relabelled at runtime is still valid
    return [(e['label'].strip('#') if strip else e['label'])
            for e in v['elements'] if e['kind'] in ks]


def check(path, iface, strict_props=True):
    doc = json.load(open(path))
    # A patcher and its subpatches share one file under 'patches'. Check every
    # patch in it -- a subpatch used to be skipped silently, which is exactly
    # where an unchecked link would hide.
    if 'patches' in doc:
        problems, warn = [], []
        for key, sub in doc['patches'].items():
            pr, wn = check_one(sub, iface, strict_props)
            problems += [f'patch {key}: {x}' for x in (pr or [])]
            warn += [f'patch {key}: {x}' for x in (wn or [])]
        return problems, warn
    return check_one(doc, iface, strict_props)


def check_one(p, iface, strict_props=True):
    if 'nodes' not in p:
        return None, None      # not a patcher (help_index.json, etc.)
    problems = []
    warn = []
    byid = {}
    for k, nc in p['nodes'].items():
        label = nc.get('init', nc.get('name', '')).split(' ')[0] or nc.get('name', '')
        byid[nc['id']] = label
        if label in ('', 'comment', 'close', 'text_block'):
            continue
        if label not in iface:
            problems.append(f"node '{label}' (entry {k}) is not a registered node type")
            continue
        if strict_props:
            valid = set(names(iface, label, ('input', 'property', 'option')) or [])
            dynamic = '<dynamic>' in valid
            for pk, pc in nc.get('properties', {}).items():
                nm = pc.get('name', '').strip('#')
                if nm == '':
                    continue
                if nm not in valid and not dynamic:
                    warn.append(f"node '{label}': property '{nm}' matches no inlet/property/option "
                                f"(have: {sorted(x for x in valid if x)})")
    # geometry: nothing may sit on top of anything else. Node sizes here are
    # the measured ones written by relayout.py -- run that before trusting this.
    boxes = []
    for k, nc in p['nodes'].items():
        if nc.get('name', '') == '':
            continue
        boxes.append((nc.get('init', nc.get('name', '?')),
                      nc.get('position_x', 0), nc.get('position_y', 0),
                      nc.get('width', 0), nc.get('height', 0)))
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            ax, ay, aw, ah = a[1:]
            bx, by, bw, bh = b[1:]
            if not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay):
                warn.append(f"overlap: {a[0]!r} and {b[0]!r} occupy the same space")

    for k, lc in p.get('links', {}).items():
        s, d = lc['source_node'], lc['dest_node']
        if s not in byid:
            problems.append(f"link {k}: source id {s} not in this patch")
            continue
        if d not in byid:
            problems.append(f"link {k}: dest id {d} not in this patch")
            continue
        sl, dl = byid[s], byid[d]
        # exact, not stripped: this is what the loader compares
        so = lc.get('source_output_name', '')
        di = lc.get('dest_input_name', '')
        outs = names(iface, sl, ('output',), strip=False)
        ins = names(iface, dl, ('input',), strip=False)
        if outs is not None and so and so not in outs and '<dynamic>' not in outs:
            problems.append(f"link {k}: '{sl}' has no outlet named '{so}' (has {outs})")
        if ins is not None and di and di not in ins and '<dynamic>' not in ins:
            problems.append(f"link {k}: '{dl}' has no inlet named '{di}' (has {ins})")
        # An empty port name is legal: the loader falls back to the single
        # inlet/outlet when there is only one. Flag it only when the node has
        # SEVERAL, where there is nothing to fall back to.
        if ins is not None and not di and '' not in ins and len(ins) > 1 \
                and '<dynamic>' not in ins:
            problems.append(f"link {k}: '{dl}' inlet not named, and it has no "
                            f"unnamed inlet to fall back to: {ins}")
    return problems, warn


def main():
    iface = load_iface()
    bad = 0
    for path in sys.argv[1:]:
        pr, wn = check(path, iface)
        tag = os.path.basename(path)
        if pr is None:
            continue
        if not pr and not wn:
            print(f"OK    {tag}")
            continue
        bad += bool(pr)
        print(f"{'FAIL ' if pr else 'warn '} {tag}")
        for x in pr:
            print('   ERROR:', x)
        for x in wn:
            print('   warn :', x)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
