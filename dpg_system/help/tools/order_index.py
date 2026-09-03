"""Rewrite help_index.json grouped by source module.

The index is read by Node.load_help_index() as {stem: [node names]}, and the
'_module' keys are separator comments that make it readable by eye. Sorting the
file flat pulls every separator to the top and the grouping is lost, so the
order is generated here rather than maintained by hand.
"""
import json, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
HELP = os.path.dirname(HERE)


def main():
    ix = json.load(open(os.path.join(HELP, 'help_index.json')))
    iface = json.load(open(os.path.join(HERE, 'iface.json')))

    families = {k: v for k, v in ix.items() if not k.startswith('_')}
    notes = {k: v for k, v in ix.items() if k.startswith('_')}

    # a family belongs to the module most of its node names come from
    home, orphans = {}, []
    for stem, labels in families.items():
        mods = collections.Counter(iface[l]['file'] for l in labels if l in iface)
        if mods:
            home[stem] = mods.most_common(1)[0][0]
        else:
            orphans.append(stem)

    out = {}
    if '_comment' in notes:
        out['_comment'] = notes['_comment']
    for mod in sorted(set(home.values())):
        key = '_' + mod[:-3] if mod.endswith('.py') else '_' + mod
        out[key] = notes.get(key, [f'--- {mod} ---'])
        for stem in sorted(s for s, m in home.items() if m == mod):
            out[stem] = families[stem]
    for stem in sorted(orphans):
        out[stem] = families[stem]

    json.dump(out, open(os.path.join(HELP, 'help_index.json'), 'w'), indent=2)
    print(f'{len(out)} entries, {len(set(home.values()))} modules grouped'
          + (f', {len(orphans)} orphaned' if orphans else ''))


if __name__ == '__main__':
    main()
