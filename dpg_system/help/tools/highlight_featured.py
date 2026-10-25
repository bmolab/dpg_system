"""Highlight the node a help page is about, wherever it appears on the page.

    python3 dpg_system/help/tools/highlight_featured.py [page ...]

A page's featured nodes are its own stem plus every node name that resolves to
it (check_coverage's resolution: help_index.json, help_file_name), so the
tensor page highlights its t.to too. Each one gets 'highlight': true, which the
app draws in its highlight theme (Visibility > Toggle Highlight for Selected).

build() and the derive_*_help.py scripts call mark_featured() before they write
a page, so a regenerated page keeps its highlights. Run this on its own to mark
pages made by hand; with no arguments it does every page in help/.

On the pages about comment, text_block, help_link and close, the page's own
furniture of those kinds - the title, the prose, the SEE ALSO row, the close
button - is left alone; only the examples are highlighted.
"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
HELP = os.path.dirname(HERE)
sys.path.insert(0, HERE)

SEE_ALSO = 'SEE ALSO - click to open'
FURNITURE = ('comment', 'text_block', 'help_link', 'close')

_names_by_stem = None


def featured_names(stem):
    global _names_by_stem
    if _names_by_stem is None:
        from check_coverage import resolve
        covered = resolve()[3]
        _names_by_stem = {}
        for label, page in covered.items():
            _names_by_stem.setdefault(page, set()).add(label)
    return {stem} | _names_by_stem.get(stem, set())


def _patches(doc):
    return list(doc['patches'].values()) if 'patches' in doc else [doc]


def _text(node):
    for p in node.get('properties', {}).values():
        if p.get('name') in ('block', 'text'):
            return str(p.get('value') or '')
    return node.get('comment') or ''


def _furniture(nodes):
    """Ids of the page's own title, prose, SEE ALSO row and close button."""
    ids = set()
    comments = [n for n in nodes if n.get('name') == 'comment']
    see_also = [n for n in comments if _text(n) == SEE_ALSO]
    ids |= {n['id'] for n in see_also}
    ids |= {n['id'] for n in comments if n.get('position_y', 0) < 10}   # the title
    blocks = [n for n in nodes if n.get('name') == 'text_block']
    if blocks:
        ids.add(max(blocks, key=lambda n: len(_text(n)))['id'])        # the prose
    if see_also:
        top = min(n['position_y'] for n in see_also)
        ids |= {n['id'] for n in nodes if n.get('name') == 'help_link'
                and top <= n.get('position_y', 0) <= top + 130}
        ids |= {n['id'] for n in nodes if n.get('name') == 'close'
                and n.get('position_y', 0) > top}
    else:
        # no SEE ALSO row: a lone close may be the page's example (close_help)
        closes = [n for n in nodes if n.get('name') == 'close']
        if len(closes) > 1:
            ids.add(max(closes, key=lambda n: n.get('position_y', 0))['id'])
    return ids


def mark_featured(doc, stem, clear=False):
    """Set 'highlight' on every featured node in every patch of doc; returns
    how many are highlighted. clear drops any highlight first - a page derived
    from a template would otherwise keep the template's."""
    names = featured_names(stem)
    count = 0
    for patch in _patches(doc):
        nodes = list(patch.get('nodes', {}).values())
        if clear:
            for n in nodes:
                n.pop('highlight', None)
        skip = _furniture(nodes) if names & set(FURNITURE) else set()
        for n in nodes:
            if n.get('name') in names and n.get('id') not in skip:
                n['highlight'] = True
                count += 1
    return count


def main(paths):
    if not paths:
        paths = sorted(glob.glob(os.path.join(HELP, '*_help.json')))
    changed, none = [], []
    for path in paths:
        raw = open(path).read()
        doc = json.loads(raw)
        stem = os.path.basename(path)[:-len('_help.json')]
        if mark_featured(doc, stem) == 0:
            none.append(stem)
        out = json.dumps(doc, indent=4) + ('\n' if raw.endswith('\n') else '')
        if out != raw:
            open(path, 'w').write(out)
            changed.append(stem)
    print(f'{len(paths)} pages, {len(changed)} changed')
    if none:
        print('no featured node found on:', ' '.join(none))


if __name__ == '__main__':
    main(sys.argv[1:])
