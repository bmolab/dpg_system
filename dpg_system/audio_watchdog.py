"""Diagnose audio dropouts: who held the interpreter when a block was late.

The audio callback is Python, so it cannot run while another thread holds
the GIL. When a block arrives late, the thread at fault has already let go
by the time the callback could look. faulthandler's timer is a C thread that
needs no GIL, so re-arming it every block makes it dump every thread's stack
at the moment a block is overdue, which shows the offender in the act.

Enable with DPG_AUDIO_WATCHDOG=1. Output goes to audio_watchdog.log in the
working directory.
"""
import faulthandler
import gc
import os
import time

ENABLED = os.environ.get('DPG_AUDIO_WATCHDOG', '') not in ('', '0')

_log = None
_last = None
_late_factor = 2.5
_gc_start = 0.0


def _write(text):
    _log.write(text + '\n')
    _log.flush()


def _gc_timer(phase, info):
    global _gc_start
    if phase == 'start':
        _gc_start = time.perf_counter()
    else:
        took = (time.perf_counter() - _gc_start) * 1000
        if took > 2.0:
            _write(f'{time.strftime("%H:%M:%S")} gc gen {info["generation"]} '
                   f'took {took:.1f} ms (collected {info["collected"]})')


def start():
    global _log
    if not ENABLED or _log is not None:
        return
    _log = open('audio_watchdog.log', 'a')
    _write(f'--- audio watchdog started {time.ctime()}, '
           f'switch interval {1000 * __import__("sys").getswitchinterval():.1f} ms')
    gc.callbacks.append(_gc_timer)


def tick(frames, sample_rate, status):
    """Call first thing in the audio callback."""
    global _last
    if _log is None:
        return
    now = time.perf_counter()
    period = frames / sample_rate
    if status.output_underflow:
        _write(f'{time.strftime("%H:%M:%S")} output underflow')
    if _last is not None and now - _last > period * _late_factor:
        _write(f'{time.strftime("%H:%M:%S")} callback late: '
               f'{1000 * (now - _last):.1f} ms gap for a {1000 * period:.1f} ms block')
    _last = now
    # Fires (once) only if the next block is overdue.
    faulthandler.dump_traceback_later(period * _late_factor, repeat=False,
                                      file=_log)
