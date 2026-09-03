"""Report help coverage: which node names have a help patch, which do not.

Mirrors Node.get_help()'s resolution order -- a node's own name first,
then help_index.json, then help_file_name set in the class.
"""
import json, os, re, glob, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
HELP = os.path.dirname(HERE)
SRC = os.path.dirname(HELP)


def resolve():
    iface = json.load(open(os.path.join(HERE, 'iface.json')))
    stems = {os.path.basename(p)[:-len('_help.json')]
             for p in glob.glob(os.path.join(HELP, '*_help.json'))}

    index = {}
    ip = os.path.join(HELP, 'help_index.json')
    if os.path.exists(ip):
        for stem, labels in json.load(open(ip)).items():
            if stem.startswith('_'):
                continue
            for lab in labels:
                index[lab] = stem

    # help_file_name assignments, resolved the way get_help() does.
    # These sit inside a class, so credit the class -- not every node in the
    # module, which would badly overstate coverage.
    import ast
    hfn = {}
    for f in glob.glob(os.path.join(SRC, '*.py')):
        try:
            tree = ast.parse(open(f, encoding='utf-8', errors='replace').read())
        except SyntaxError:
            continue
        for cd in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for n in ast.walk(cd):
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute) \
                        and n.targets[0].attr == 'help_file_name':
                    v = n.value
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        stem = v.value
                        if stem.endswith('_help'):
                            stem = stem[:-len('_help')]
                        hfn[(os.path.basename(f), cd.name)] = stem

    # help_file_name resolved through the class chain by extract_interface
    inherited = {}
    for lab, v in iface.items():
        for e in v['elements']:
            if e['kind'] == 'help_file_name':
                stem = e['label']
                if stem.endswith('_help'):
                    stem = stem[:-len('_help')]
                inherited[lab] = stem

    covered, missing = {}, []
    for lab, v in iface.items():
        if lab in stems:
            covered[lab] = lab
        elif lab in index:
            covered[lab] = index[lab]
        elif (v['file'], v['class']) in hfn and hfn[(v['file'], v['class'])] in stems:
            covered[lab] = hfn[(v['file'], v['class'])]
        elif inherited.get(lab) in stems:
            # help_file_name set on a BASE class reaches every subclass --
            # TorchDistributionNode carries 't.dist_help' for all 21 t.dist.*
            covered[lab] = inherited[lab]
        else:
            missing.append((v['file'], lab))
    return iface, stems, index, covered, missing


def main():
    iface, stems, index, covered, missing = resolve()
    print(f"help patches on disk : {len(stems)}")
    print(f"node names total     : {len(iface)}")
    print(f"      documented     : {len(covered)}")
    print(f"      undocumented   : {len(missing)}")

    # index entries pointing at a file that is not there
    dangling = sorted({s for s in index.values() if s not in stems})
    if dangling:
        print("\nhelp_index.json points at missing files:")
        for s in dangling:
            print('   ', s + '_help.json')
    unknown = sorted(l for l in index if l not in iface)
    if unknown:
        print("\nhelp_index.json lists names that are not registered nodes:")
        for l in unknown:
            print('   ', l)

    print("\nremaining, by module:")
    by = collections.Counter(f for f, _ in missing)
    tot = collections.Counter(v['file'] for v in iface.values())
    for f, n in by.most_common():
        if len(sys.argv) > 1 and sys.argv[1] != f:
            continue
        print(f"   {f:36} {n:>3} of {tot[f]:>3}")
    if len(sys.argv) > 1:
        print(f"\nundocumented in {sys.argv[1]}:")
        for f, l in sorted(missing):
            if f == sys.argv[1]:
                print('   ', l)


if __name__ == '__main__':
    main()
