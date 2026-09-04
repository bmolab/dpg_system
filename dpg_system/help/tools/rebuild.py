"""Rebuild every generated help patch, in the right order.

    generate  ->  relayout (measure for real, place comments)  ->  validate

Run it in the project's conda env from the repo root:

    python3 dpg_system/help/tools/rebuild.py

Order matters: the generators write provisional positions, so relayout has to
run after them or comments end up on top of nodes. This script exists so that
cannot be got wrong by hand.
"""
import os, sys, glob, subprocess, json

HERE = os.path.dirname(os.path.abspath(__file__))
HELP = os.path.dirname(HERE)
PY = sys.executable


def generated_stems():
    """Which help patches these generators own — everything they print."""
    stems = set()
    for gen in sorted(glob.glob(os.path.join(HERE, 'make_*_help.py'))):
        r = subprocess.run([PY, gen], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAILED {os.path.basename(gen)}\n{r.stderr}")
            continue
        made = [l.strip() for l in r.stdout.splitlines() if l.strip().endswith('.json')]
        print(f"  {os.path.basename(gen):32} {len(made)} patches")
        stems.update(made)
    return sorted(stems)


CHUNK = 30          # patches per relayout process
CHUNK_TIMEOUT = 420  # seconds before a batch is presumed wedged
ONE_TIMEOUT = 240    # ... and before a single patch is


def relayout_in_batches(paths):
    """Lay the patches out a batch at a time, under an external timeout.

    Two things go wrong when all ~210 go to one process, and both were seen:

    - Memory accumulates across loads until the OS kills the process outright
      (exit -9). relayout writes nothing until it has measured every patch it
      was given, so one kill discarded the layout of all 30 patches in flight.
    - A patch can wedge in a way NOTHING INSIDE THE PROCESS can break. Closing
      the ndi_receiver patch calls NDIlib_find_destroy, which joins an ONVIF
      discovery thread that can spin forever -- holding the GIL, so relayout's
      own watchdog thread never gets to run. Only an outside process can kill
      that, which is why the timeout lives here and not in relayout.py.

    A wedged batch is retried one patch at a time, so a single bad patch costs
    itself and not the 29 around it.
    """
    reported = 0
    for i in range(0, len(paths), CHUNK):
        batch = paths[i:i + CHUNK]
        try:
            r = subprocess.run([PY, os.path.join(HERE, 'relayout.py')] + batch,
                               capture_output=True, text=True, timeout=CHUNK_TIMEOUT)
            lines = [l for l in r.stdout.splitlines() if '_help.json' in l]
            if r.returncode == 0 and len(lines) >= len(batch):
                for l in lines:
                    print(l)
                reported += len(lines)
                continue
            print(f'  batch {i // CHUNK}: exit {r.returncode}, '
                  f'{len(lines)}/{len(batch)} laid out -- retrying singly')
        except subprocess.TimeoutExpired:
            print(f'  batch {i // CHUNK}: TIMED OUT -- retrying singly')
        for one in batch:
            name = os.path.basename(one)
            try:
                r1 = subprocess.run([PY, os.path.join(HERE, 'relayout.py'), one],
                                    capture_output=True, text=True, timeout=ONE_TIMEOUT)
                lines = [l for l in r1.stdout.splitlines() if '_help.json' in l]
                if lines:
                    for l in lines:
                        print(l)
                    reported += len(lines)
                else:
                    print(f'  {name:34} NO LAYOUT (exit {r1.returncode})')
            except subprocess.TimeoutExpired:
                print(f'  {name:34} HANGS -- skipped, layout NOT applied')
    return reported


def main():
    print('generating:')
    paths = generated_stems()
    if not paths:
        print('nothing generated'); return

    print('\nrelayout (loads each patch and measures it for real):')
    reported = relayout_in_batches(paths)
    # Never swallow a failure here. relayout printing nothing means the layout
    # pass did not happen, and every generated patch is left with provisional
    # positions -- which validates as overlapping and gets worse each rebuild.
    if reported == 0:
        print(f'  RELAYOUT PRODUCED NOTHING for {len(paths)} paths '
              f'-- layout NOT applied')

    print('\nvalidate:')
    r = subprocess.run([PY, os.path.join(HERE, 'validate_help.py')] + paths,
                       capture_output=True, text=True)
    print(r.stdout.rstrip())

    print('\nindex:')
    subprocess.run([PY, os.path.join(HERE, 'order_index.py')])

    print()
    subprocess.run([PY, os.path.join(HERE, 'check_coverage.py')])


if __name__ == '__main__':
    main()
