"""Unwrap help prose: one line per paragraph, so the text_block wraps it.

The generators write their prose hard-wrapped at about 72 characters. A
text_block wraps its text to its own width, so whenever the box is narrower
than those lines (or the font larger) every line breaks twice and the
paragraph comes out ragged. Stored as one line per paragraph, the text
follows whatever width the box has.

Kept as they are:
  - blank lines (paragraph breaks)
  - a line ending in ':' that starts a paragraph (a heading such as SYNTAX:
    or 'in:'); a colon at the end of a line inside a sentence is joined
  - the lines under SYNTAX: and EXAMPLE(S): (node boxes, one per line)
  - list items (-, *, 1., 'key: ...', lines ending in ;), lines laid out in
    columns (runs of spaces), and lines with <placeholders> in them
  - a line shorter than FILLED characters: it was not hard-wrapped; its
    author ended it there (one sentence per line, an OR between syntaxes)
  - an ALL-CAPS heading ending in ':', even straight after a paragraph

    python3 dpg_system/help/tools/reflow_prose.py dpg_system/help/*_help.json
    python3 dpg_system/help/tools/reflow_prose.py --dry dpg_system/help/x_help.json

On a page only the main prose block is reflowed (a text_block of more than
MIN_CHARS characters); short annotation blocks in the demo keep their lines.
"""
import json, re, sys

MIN_CHARS = 600
VERBATIM = re.compile(r'^(SYNTAX|EXAMPLES?)\b.*:$')
LISTY = re.compile(r'^\s*([-*•]|\d+[.)])\s')
COLUMNS = re.compile(r'\S {2,}\S')
CAPS_HEADING = re.compile(r"^[A-Z0-9][A-Z0-9 ,()'/&+-]*[A-Z)]( [a-z_.]+)*( [A-Z0-9 ,()'/&+-]+)?:$")
KEY_ITEM = re.compile(r'^[\w.\[\]<>/+-]+( [\w.\[\]<>/+-]+){0,3}: ')   # 'int64: whole numbers'
SYNTAXY = re.compile(r'<[^<>]+>')                                        # 'set <value> ...'
FILLED = 50     # a line this long was hard-wrapped; a shorter one ended where its author meant


def _items(lines):
    """Which lines are list items: a '-', '*' or '1.' line that opens a
    paragraph, follows a line ending in ':', or follows another item. A dash
    that only begins a wrapped line of prose ('- the distance from zero -')
    is not one."""
    items = []
    for i, line in enumerate(lines):
        prev = lines[i - 1] if i else ''
        items.append(bool(LISTY.match(line)) and (
            not prev.strip() or prev.rstrip().endswith(':') or (i and items[i - 1])))
    return items


def _keep_break(line, nxt, starts_para, line_item=False, nxt_item=False):
    """True when the newline between line and nxt must stay."""
    if not line.strip() or not nxt.strip():
        return True
    if line.rstrip().endswith(':') and (starts_para or CAPS_HEADING.match(line.strip())):
        return True     # a heading; a colon at the end of a line mid-sentence is not one
    if CAPS_HEADING.match(nxt.strip()):
        return True     # a heading straight after a paragraph, with no blank line
    if len(line.rstrip()) < FILLED:
        return True     # a short line: its author ended it there (one sentence per line, OR, ...)
    if line.rstrip().endswith(';'):
        return True
    if KEY_ITEM.match(nxt) and line.rstrip()[-1:] in '.:;!?)':
        return True     # a 'key: ...' item, but only after a finished line, not mid-sentence
    if SYNTAXY.search(line) or SYNTAXY.search(nxt):
        return True
    if line_item or nxt_item:
        return True     # list items keep their own lines
    if COLUMNS.search(line) or COLUMNS.search(nxt):
        return True
    if line.startswith(' ') or nxt.startswith(' '):
        return True
    return False


def reflow(text):
    lines = text.split('\n')
    items = _items(lines)
    out = []
    verbatim = False
    for i, line in enumerate(lines):
        if not line.strip():
            verbatim = False
        if VERBATIM.match(line.strip()):
            verbatim = True
            out.append(line)
            out.append('\n')
            continue
        out.append(line)
        if i == len(lines) - 1:
            break
        nxt = lines[i + 1]
        starts_para = i == 0 or not lines[i - 1].strip() or lines[i - 1].rstrip().endswith(':') \
            and (i < 2 or not lines[i - 2].strip())
        if verbatim or _keep_break(line, nxt, starts_para, items[i], items[i + 1]):
            out.append('\n')
        else:
            out.append(' ')
    return ''.join(out)


def _blocks(d):
    patches = d['patches'].values() if 'patches' in d else [d]
    for p in patches:
        for n in p.get('nodes', {}).values():
            if n.get('name') != 'text_block':
                continue
            for pp in n.get('properties', {}).values():
                if pp.get('name') == 'block' and isinstance(pp.get('value'), str) \
                        and len(pp['value']) > MIN_CHARS:
                    yield n, pp


def reflow_file(path, dry=False):
    raw = open(path).read()
    d = json.loads(raw)
    changed = 0
    for n, pp in _blocks(d):
        new = reflow(pp['value'])
        if new != pp['value']:
            changed += 1
            pp['value'] = new
            if 'text' in n:
                n['text'] = new      # the copy the app shows; must match 'block'
    if changed and not dry:
        indent = 4 if raw.startswith('{\n') else None
        with open(path, 'w') as f:
            json.dump(d, f, indent=indent)
    return changed


if __name__ == '__main__':
    dry = '--dry' in sys.argv
    paths = [a for a in sys.argv[1:] if a.endswith('.json')]
    total = 0
    for path in paths:
        c = reflow_file(path, dry=dry)
        total += bool(c)
    print(f'{total} of {len(paths)} pages reflowed' + (' (dry run)' if dry else ''))
