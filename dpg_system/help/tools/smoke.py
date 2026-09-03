"""Run each help patch for a moment and report demos that do nothing.

A help patch whose demo sits at zero teaches nothing, and the failure is silent
-- the patch loads, validates, and looks right. This loads each one, drives the
frame tasks, and taps every outlet, then flags any node that never sent
anything or only ever sent zeros.

That is exactly the shape of the '* 0.5 behaves as * 0' bug: the file was
correct, the node was wired correctly, and the output was flat.

    python3 dpg_system/help/tools/smoke.py dpg_system/help/*.json
"""
import sys, os, time

REPO = '/Users/drokeby/dpg_system'
# Long enough to cover a full cycle of the slowest signal node used in a demo
# (periods here run to 4 seconds). Too short a window and a sine that happens to
# be in its negative half reads as ALWAYS ZERO -- a false alarm.
FRAMES = 220
FRAME_SLEEP = 0.02
# Outlets that legitimately stay quiet: a toggle or button only sends when
# clicked, and a plot's outlet is not part of the demonstration.
SKIP = {'comment', 'text_block', 'close', 'load_bang', 'print', 'plot',
        'toggle', 'button', 'heat_map', ''}



def _start_watchdog(seconds=180):
    """Force-exit if a load blocks.

    Some nodes open a NATIVE modal dialog (gl_text's font property is one), and
    a headless run then waits on it forever with no output. Nothing here can
    dismiss that dialog, so the only safe behaviour is to die loudly rather
    than hang a build.
    """
    import threading, os as _os

    def _bark():
        print(f"\n  WATCHDOG: no progress for {seconds}s -- a patch is probably "
              f"blocking on a modal dialog. Aborting.", flush=True)
        _os._exit(2)

    t = threading.Timer(seconds, _bark)
    t.daemon = True
    t.start()
    return t


def _pet(timer, seconds=180):
    """Restart the watchdog after each patch."""
    try:
        timer.cancel()
    except Exception:
        pass
    return _start_watchdog(seconds)


def is_zeroish(v):
    try:
        if v is None:
            return True
        if isinstance(v, (int, float, bool)):
            return v == 0
        if hasattr(v, 'any'):
            return not bool(v.any())
    except Exception:
        pass
    return False


def main():
    # Many demos are event-driven on purpose -- dicts, replace, repeat and
    # tracing are not streams. --click bangs every button node partway through
    # so those chains actually get exercised.
    click = '--click' in sys.argv
    paths = [a for a in sys.argv[1:] if a.endswith('.json')
             and os.path.basename(a) != 'help_index.json']
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    import dearpygui.dearpygui as dpg
    from dpg_system.dpg_app import App

    app = App()
    app.register_nodes()
    app.start()
    for _ in range(3):
        dpg.render_dearpygui_frame()

    print()
    dog = _start_watchdog()
    for path in paths:
        dog = _pet(dog)
        app.fresh_patcher = True
        try:
            app.load_from_file(os.path.abspath(path))
        except Exception as e:
            print(f"  {os.path.basename(path):34} LOAD FAILED: {e}")
            continue
        editor = app.get_current_editor()
        seen = {}
        for n in editor._nodes:
            label = getattr(n, 'label', '')
            if label in SKIP:
                continue
            for out in getattr(n, 'outputs', []):
                # A ~ node's signal outlets carry samples through the compiled
                # DSP graph, not messages -- there is nothing to tap, and
                # headlessly the audio callback is not running at all, so every
                # one of them would read as silent. Counting them as failures
                # buries the real findings.
                if getattr(out, 'synth_signal', None) is not None:
                    continue
                key = (id(n), label, out.get_label())
                seen[key] = []
                def tap(v, _k=key, _orig=out.send, *a, **k):
                    seen[_k].append(v)
                    return _orig(v, *a, **k)
                out.send = tap

        buttons = [n for n in editor._nodes if getattr(n, 'label', '') == 'button']
        frames = 40 if click else FRAMES
        for i in range(frames):
            if click and i in (5, 20) and buttons:
                for b in buttons:
                    for out in getattr(b, 'outputs', []):
                        try:
                            out.send('bang')
                        except Exception as e:
                            print('   button bang failed:', e)
            for t in list(app.frame_tasks):
                try:
                    t.frame_task()
                except Exception:
                    pass
            dpg.render_dearpygui_frame()
            time.sleep(FRAME_SLEEP)

        # A patch often holds several nodes of the same label. Tagging by label
        # alone merged them, so one silent instance out of three reported the
        # whole label as silent -- ambiguous, and twice now it read as a bug in
        # a demo that was correct. Number the instances when there is more than
        # one of a label.
        counts = {}
        for (nid, label, oname) in seen:
            counts[label] = counts.get(label, 0) + 1
        seen_index = {}

        silent, flat = [], []
        for (nid, label, oname), vals in seen.items():
            if counts.get(label, 0) > 1:
                k = (nid, label)
                if k not in seen_index:
                    seen_index[k] = len(
                        [x for x in seen_index if x[1] == label]) + 1
                label_shown = f'{label}#{seen_index[k]}'
            else:
                label_shown = label
            tag = f"{label_shown}" + (f".{oname}" if oname else "")
            if not vals:
                silent.append(tag)
            elif all(is_zeroish(v) for v in vals):
                flat.append(tag)
        bits = []
        if flat:
            bits.append(f"ALWAYS ZERO: {sorted(set(flat))}")
        if silent:
            bits.append(f"never sent: {sorted(set(silent))}")
        status = '  |  '.join(bits) if bits else 'alive'
        audio = sum(1 for n in editor._nodes
                    if getattr(n, 'label', '').endswith('~'))
        if audio:
            status += f"   [{audio} audio nodes: signal path not exercised headlessly]"
        print(f"  {os.path.basename(path):34} {status}")

        # Close the patch before loading the next. Without this every patch
        # stays open in its own tab for the life of the process, and each
        # later load is slower than the one before.
        try:
            app.close_current_node_editor()
        except Exception as e:
            print('   could not close patch:', e)

    try:
        dog.cancel()      # finished: do not let teardown trip the watchdog
    except Exception:
        pass



if __name__ == '__main__':
    main()
    # The work is written by the time main() returns. Tearing the app down --
    # GL context, audio, MIDI threads -- does not reliably complete headlessly,
    # and the process then hangs forever having already done its job. Exit now.
    sys.stdout.flush()
    os._exit(0)
