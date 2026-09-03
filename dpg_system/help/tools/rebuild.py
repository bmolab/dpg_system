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


def main():
    print('generating:')
    paths = generated_stems()
    if not paths:
        print('nothing generated'); return

    print('\nrelayout (loads each patch and measures it for real):')
    r = subprocess.run([PY, os.path.join(HERE, 'relayout.py')] + paths,
                       capture_output=True, text=True)
    reported = 0
    for line in r.stdout.splitlines():
        if '_help.json' in line:
            print(line)
            reported += 1
    # Never swallow a failure here. relayout printing nothing means the layout
    # pass did not happen, and every generated patch is left with provisional
    # positions -- which validates as overlapping and gets worse each rebuild.
    if r.returncode != 0 or reported == 0:
        print(f'  RELAYOUT PRODUCED NOTHING (exit {r.returncode}) for '
              f'{len(paths)} paths -- layout NOT applied')
        print('  --- stdout tail ---')
        print('\n'.join(r.stdout.splitlines()[-15:]))
        print('  --- stderr tail ---')
        print('\n'.join(r.stderr.splitlines()[-15:]))

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
