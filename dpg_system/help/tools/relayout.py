"""Measure a help patch for real, then place its comments so nothing overlaps.

Guessing node sizes does not work — a node's width depends on the widgets
inside it, so hand-picked comment positions end up under a plot. This loads the
patch headlessly, reads the ACTUAL rendered size of every node, writes those
sizes back, and moves each comment into a clear gutter beside the demo column,
keeping it level with the row it annotates.

Run it in the project's conda env, from the repo root:

    python3 dpg_system/help/tools/relayout.py dpg_system/help/*.json
"""
import sys, os, json, faulthandler

REPO = '/Users/drokeby/dpg_system'
GUTTER = 26        # gap between the demo column and the comment gutter
TEXT_GAP = 30      # gap between the comments and the text block
V_GAP = 10         # minimum vertical gap between two comments
TITLE_Y = 40       # a comment above this y is the patch title, left alone



def _start_watchdog(seconds=180):
    """Force-exit if a load blocks, and say where it stopped.

    Some nodes open a NATIVE modal dialog (gl_text's font property is one), and
    a headless run then waits on it forever with no output.

    This uses faulthandler, NOT a threading.Timer, and the difference matters.
    A Timer callback is Python: it needs the GIL to run. A C extension that
    blocks WITHOUT releasing the GIL therefore freezes the watchdog along with
    everything else -- exactly what NDIlib_find_destroy did here, wedging a run
    for fifty minutes with the timer never firing. faulthandler's countdown
    lives in a C thread that needs no GIL, so it still fires, and it dumps a
    Python traceback for every thread on the way out, which names the blocking
    call instead of leaving you guessing.
    """
    faulthandler.dump_traceback_later(seconds, exit=True)
    return seconds


def _pet(timer, seconds=180):
    """Restart the watchdog after each patch."""
    faulthandler.cancel_dump_traceback_later()
    return _start_watchdog(seconds)


def measure(paths):
    """label/size for every node in each patch, by loading them for real."""
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    import dearpygui.dearpygui as dpg
    from dpg_system.dpg_app import App

    app = App()
    app.register_nodes()
    app.start()
    for _ in range(3):
        dpg.render_dearpygui_frame()

    out = {}
    dog = _start_watchdog()
    for n, path in enumerate(paths, 1):
        dog = _pet(dog)
        # Progress, flushed. Without this the loop is silent until every patch
        # has been measured, so a patch that blocks is invisible -- you get no
        # output at all and no way to tell which one it was.
        print(f'  [{n}/{len(paths)}] {os.path.basename(path)}', flush=True)
        app.fresh_patcher = True
        app.load_from_file(os.path.abspath(path))
        # Render until the sizes STOP CHANGING, rather than for a fixed count.
        # Some nodes are not their final size after a few frames -- fader_out~
        # measured 220x220 early and 70x308 once settled -- and a close button
        # placed against the transient lands on top of the node it was meant to
        # sit below. Snapshot, render more, compare; give up after a while and
        # take what we have.
        _prev = None
        for _ in range(12):
            for _ in range(6):
                dpg.render_dearpygui_frame()
            snap = {}
            for _id, _n in app.created_nodes.items():
                if _n is None:
                    continue
                try:
                    snap[_id] = dpg.get_item_rect_size(_n.uuid)
                except Exception:
                    pass
            if snap and snap == _prev:
                break
            _prev = snap
        # app.created_nodes maps the id SAVED IN THE FILE to the node it
        # built, and survives the load. node.loaded_uuid does not: paste()
        # ends with clear_loaded_uuids(), which sets every one of them to -1.
        # Reading it here collapsed all the measurements onto the key -1, so
        # nothing ever matched a node id and no size was written back -- the
        # measure pass silently did nothing.
        sizes = {}
        for loaded, n in app.created_nodes.items():
            if n is None:
                continue
            try:
                w, h = dpg.get_item_rect_size(n.uuid)
            except Exception:
                continue
            if w and h:
                sizes[loaded] = (int(w), int(h))
        out[path] = sizes

        # Close the patch before loading the next. Without this every patch
        # stays open in its own tab for the life of the process -- 169 of them
        # by the end of a full rebuild -- which slows every later load down.
        try:
            app.close_current_node_editor()
        except Exception as e:
            print('   could not close patch:', e)
    faulthandler.cancel_dump_traceback_later()
    return out


def is_title(nc):
    if nc.get('name') != 'comment':
        return False
    if nc.get('position_y', 0) < TITLE_Y:
        return True
    for p in nc.get('properties', {}).values():
        if p.get('name') == 'font size' and str(p.get('value')) == '48':
            return True
    return False


def overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def resolve_minimal(movable, anchors, gap=10):
    """Fix only what is actually broken.

    For a hand-placed patch, re-flowing everything would throw away the
    author's layout. This leaves every item exactly where it is unless it
    genuinely overlaps something, and then moves it the shortest distance that
    clears -- down, or left out from under an anchor, whichever is nearer.
    """
    placed = list(anchors)          # text block and title do not move
    moved = []
    for nc in sorted(movable, key=lambda c: (c.get('position_y', 0), c.get('position_x', 0))):
        b = (nc.get('position_x', 0), nc.get('position_y', 0),
             nc.get('width', 100), nc.get('height', 40))
        if not any(overlaps(b, q) for q in placed):
            placed.append(b)
            continue

        # straight down, below whatever it runs into
        y = b[1]
        for _ in range(40):
            hit = next((q for q in placed if overlaps((b[0], y, b[2], b[3]), q)), None)
            if hit is None:
                break
            y = hit[1] + hit[3] + gap
        options = [(b[0], y, abs(y - b[1]))]

        # or left, clear of the first thing it runs into
        hit = next((q for q in placed if overlaps(b, q)), None)
        if hit is not None:
            x = hit[0] - gap - b[2]
            if x >= 20 and not any(overlaps((x, b[1], b[2], b[3]), q) for q in placed):
                options.append((x, b[1], abs(x - b[0])))

        nx, ny, _ = min(options, key=lambda o: o[2])
        label = nc.get('init', nc.get('name', '?'))
        moved.append((label, (b[0], b[1]), (nx, ny)))
        nc['position_x'], nc['position_y'] = nx, ny
        placed.append((nx, ny, b[2], b[3]))
    return moved


def relayout(path, sizes, minimal=False, dry=False):
    doc = json.load(open(path))
    # A file may hold several patches (a patcher and its subpatches) under
    # 'patches'. Lay out the top one -- the patch with no parent -- and leave
    # the subpatches as authored: they are small, hand-placed, and have no
    # title or text block for the gutter logic to work against.
    if 'patches' in doc:
        tops = [q for q in doc['patches'].values() if q.get('parent_node_uuid') is None]
        p = tops[0] if tops else list(doc['patches'].values())[0]
    else:
        p = doc
    nodes = p['nodes']

    # 1. write the measured sizes back, so width/height stop being guesses
    for nc in nodes.values():
        s = sizes.get(nc.get('id'))
        if s:
            nc['width'], nc['height'] = s

    def box(nc):
        return (nc.get('position_x', 0), nc.get('position_y', 0),
                nc.get('width', 100), nc.get('height', 40))

    text_block = None
    title = None
    close = None
    demo, comments = [], []
    for nc in nodes.values():
        nm = nc.get('name', '')
        if nm == '':
            continue                      # origin
        if nm == 'text_block':
            # A multi-line annotation is a locked text_block, not a comment
            # node, and belongs in the gutter with the other annotations.
            # build_help marks those; anything unmarked is either the prose
            # block or a demo of the node itself.
            if nc.get('annotation'):
                comments.append(nc)
                continue
            # A patch may contain more than one (text_block_help demonstrates
            # the node itself). The documentation block is the largest; the
            # rest are part of the demo and may be moved.
            if text_block is None:
                text_block = nc
            elif nc['width'] * nc['height'] > text_block['width'] * text_block['height']:
                demo.append(text_block)
                text_block = nc
            else:
                demo.append(nc)
        elif nm == 'close':
            close = nc
        elif is_title(nc):
            title = nc
        elif nm == 'comment':
            comments.append(nc)
        else:
            demo.append(nc)

    if not demo and not minimal:
        return None

    if minimal:
        moved = []
        # A long title can clip the top corner of the text block. Move the
        # block down rather than the title -- the title is the thing the reader
        # looks for first, and the block has room to give.
        if text_block is not None and title is not None:
            tb = (text_block['position_x'], text_block['position_y'],
                  text_block['width'], text_block['height'])
            ti = (title['position_x'], title['position_y'],
                  title['width'], title['height'])
            if overlaps(tb, ti):
                was = (text_block['position_x'], text_block['position_y'])
                text_block['position_y'] = ti[1] + ti[3] + 8
                moved.append(('text_block (cleared the title)', was,
                              (text_block['position_x'], text_block['position_y'])))
        anchors = [(n['position_x'], n['position_y'], n['width'], n['height'])
                   for n in (text_block, title) if n]
        moved += resolve_minimal(demo + comments + ([close] if close else []), anchors)
        everything = demo + comments + [n for n in (text_block, title, close) if n]
        p['width'] = max(nc['position_x'] + nc['width'] for nc in everything) + 40
        p['height'] = max(nc['position_y'] + nc['height'] for nc in everything) + 60
        if not dry:
            json.dump(doc, open(path, 'w'), indent=4)
        return ('minimal', moved)

    # 2. nothing starts above the title block
    if title is not None and demo:
        floor = title.get('position_y', 0) + title.get('height', 42) + 12
        top = min(nc['position_y'] for nc in demo)
        if top < floor:
            shift = floor - top
            for nc in demo + comments:
                nc['position_y'] += shift

    # 3. the close button sits below the demo, out of everyone's way
    if close is not None and demo:
        close['position_x'] = 24
        close['position_y'] = max(nc['position_y'] + nc['height'] for nc in demo) + 30

    # 4. comments go in a gutter clear of every demo node
    right = max(nc['position_x'] + nc['width'] for nc in demo)
    gutter_x = right + GUTTER

    placed = []
    moved = 0
    for nc in sorted(comments, key=lambda c: (c.get('position_y', 0), c.get('position_x', 0))):
        before = (nc['position_x'], nc['position_y'])
        nc['position_x'] = gutter_x
        y = nc['position_y']
        # keep it level with its row, but never on top of another comment
        for q in placed:
            while overlaps((gutter_x, y, nc['width'], nc['height']), q):
                y = q[1] + q[3] + V_GAP
        nc['position_y'] = y
        placed.append((gutter_x, y, nc['width'], nc['height']))
        if (nc['position_x'], nc['position_y']) != before:
            moved += 1

    # 5. the text block clears the comment gutter
    widest_comment = max([c['width'] for c in comments], default=0)
    if text_block is not None:
        text_block['position_x'] = gutter_x + widest_comment + TEXT_GAP

    # 6. patch size follows the content
    everything = demo + comments + [n for n in (text_block, title, close) if n]
    p['width'] = max(nc['position_x'] + nc['width'] for nc in everything) + 40
    p['height'] = max(nc['position_y'] + nc['height'] for nc in everything) + 60

    # 7. report any demo nodes still colliding — those need a spec fix
    collisions = []
    checked = demo + ([title] if title else []) + ([close] if close else [])
    for i, a in enumerate(checked):
        for b in checked[i + 1:]:
            if overlaps(box(a), box(b)):
                collisions.append((a.get('init', a['name']), b.get('init', b['name'])))

    if not dry:
        json.dump(doc, open(path, 'w'), indent=4)
    return moved, collisions


def main():
    args = sys.argv[1:]
    minimal = '--minimal' in args
    dry = '--dry-run' in args
    paths = [a for a in args if a.endswith('.json')
             and os.path.basename(a) != 'help_index.json']
    sizes = measure(paths)
    print()
    if minimal:
        print('  minimal mode: only items that actually overlap are moved'
              + ('  (DRY RUN, nothing written)' if dry else ''))
    total = 0
    for path in paths:
        r = relayout(path, sizes.get(path, {}), minimal=minimal, dry=dry)
        if r is None:
            print(f"  {os.path.basename(path):34} no demo nodes, skipped")
            continue
        if r[0] == 'minimal':
            moved = r[1]
            total += len(moved)
            if not moved:
                print(f"  {os.path.basename(path):34} already clear")
            else:
                print(f"  {os.path.basename(path):34} {len(moved)} moved")
                for label, was, now in moved:
                    if was is None:
                        print(f"        {label}")
                    else:
                        print(f"        {str(label)[:44]:46} {was} -> {now}")
            continue
        moved, collisions = r
        note = f"{moved} comments placed"
        if collisions:
            note += f"  |  NODE COLLISIONS: {collisions}"
        print(f"  {os.path.basename(path):34} {note}")
    if minimal:
        print(f"\n  {total} items moved in total")


if __name__ == '__main__':
    main()
    # The work is written by the time main() returns. Tearing the app down --
    # GL context, audio, MIDI threads -- does not reliably complete headlessly,
    # and the process then hangs forever having already done its job. Exit now.
    sys.stdout.flush()
    os._exit(0)
