"""Capture 45 s from both Erae ports and log note on/off per channel with the held-note count.
Run it, then touch the pad; it shows whether several fingers on one key arrive as separate voices."""
import mido, time, collections, os, threading
threading.Timer(90, lambda: os._exit(3)).start()
mido.set_backend('mido.backends.rtmidi')
log = collections.defaultdict(list); clocks = collections.Counter()
def mk(port):
    def cb(msg):
        if msg.type in ('clock', 'active_sensing'):
            clocks[port] += 1; return
        log[port].append((time.time(), msg))
    return cb
ins = [mido.open_input(p, callback=mk(p)) for p in ('Erae 2 MIDI', 'Erae 2 MIDI (MPE)')]
print('capturing 45 s', flush=True); time.sleep(45); print('clock+sensing per port:', dict(clocks), flush=True)
for p in ('Erae 2 MIDI', 'Erae 2 MIDI (MPE)'):
    msgs = log[p]; print('=====', p, len(msgs), 'msgs')
    kinds = collections.Counter((m.type, getattr(m, 'channel', -1), getattr(m, 'control', None)) for _, m in msgs)
    for k, n in sorted(kinds.items(), key=lambda kv: -kv[1])[:16]: print('  ', k, n)
    held = collections.defaultdict(set); peak = 0
    t0 = msgs[0][0] if msgs else 0
    print('  note events:')
    for t, m in msgs:
        if m.type == 'note_on' and m.velocity > 0:
            held[m.channel].add(m.note); n = sum(len(v) for v in held.values()); peak = max(peak, n)
            print('   %6.2f ON  ch%-2d note %d vel %d   held now %d' % (t-t0, m.channel+1, m.note, m.velocity, n))
        elif m.type == 'note_off' or (m.type == 'note_on' and m.velocity == 0):
            held[m.channel].discard(m.note); n = sum(len(v) for v in held.values())
            print('   %6.2f OFF ch%-2d note %d            held now %d' % (t-t0, m.channel+1, m.note, n))
    print('  peak simultaneous notes:', peak, ' channels used:', sorted(c+1 for c in {getattr(m,'channel',-1) for _,m in msgs} if c >= 0))
os._exit(0)
