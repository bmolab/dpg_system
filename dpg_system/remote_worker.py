"""Run an object in a worker process, used as if it were local.

Why: a node's heavy per-frame work (smpl_torque's processor, smpl_ragdoll's
physics) is Python, so wherever it runs it holds the GIL -- and the render
loop holds it too.  When a capture thread delivers frames at 100 Hz and the
render loop is busy, the two share one core between them and the capture
falls behind without bound.  In a worker process the work has a core and a
GIL of its own; the calling thread only waits on a socket, which releases
the GIL.

    proxy = RemoteObject('dpg_system.smpl_processor:SMPLProcessor', framerate=100.0)
    res = proxy.process_frame(pose, trans, options)     # runs in the worker
    offsets = proxy.skeleton_offsets                     # fetched from the worker

Attribute reads, writes and method calls are forwarded.  Every forward is a
round trip (tens of microseconds), so a subclass names its per-frame method
in `snapshot_after`, with the attributes (and zero-argument methods) the
caller reads after it: they come back with the result in the same trip and
are served locally until anything else is forwarded.

The worker is started with subprocess and a private socketpair, not
multiprocessing: spawn would re-import the app's main script in the child,
and fork would copy a process holding GL contexts and running threads.  It
exits when its socket closes, so it does not outlive the app.

If the worker dies, every later use raises WorkerDied; a node responds by
building a new one.
"""

import os
import socket
import subprocess
import sys
import threading
import traceback
import importlib
from multiprocessing.connection import Connection

_MISSING = '__remote_worker_missing__'


class WorkerDied(RuntimeError):
    pass


def _resolve(path):
    module, _, name = path.partition(':')
    obj = importlib.import_module(module)
    for part in name.split('.'):
        obj = getattr(obj, part)
    return obj


class RemoteObject:
    # method name -> (attribute names, zero-argument method names) to return
    # with its result
    snapshot_after = {}

    def __init__(self, factory, *args, **kwargs):
        """factory: 'module:callable' that builds the object in the worker.
        Construction starts now and is waited for on first use."""
        # A thread waiting on the worker gets the GIL back, when the reply
        # comes, only once the thread holding it yields -- by default after
        # up to 5 ms.  With the render loop busy that wait recurs at every
        # reply, and at 100 Hz it was enough to put the capture thread behind
        # again (measured: latency growing past a second with the render loop
        # busy 13 of every 16.7 ms; steady at 6 ms with 0.5 ms here).
        if sys.getswitchinterval() > 0.0005:
            sys.setswitchinterval(0.0005)
        parent_sock, child_sock = socket.socketpair()
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        env['PYTHONPATH'] = package_root + os.pathsep + env.get('PYTHONPATH', '')
        d = self.__dict__
        d['_proc'] = subprocess.Popen(
            [sys.executable, '-m', 'dpg_system.remote_worker', str(child_sock.fileno())],
            pass_fds=(child_sock.fileno(),), env=env)
        child_sock.close()
        d['_conn'] = Connection(parent_sock.detach())
        d['_lock'] = threading.Lock()
        d['_snapshot'] = None
        d['_callables'] = set()
        d['_dead'] = False
        d['_pending'] = 1                    # the create reply, collected on first use
        d['_factory'] = factory
        self._send(('create', factory, args, kwargs))

    # -- transport

    def _send(self, msg):
        try:
            self._conn.send(msg)
        except (OSError, EOFError) as e:
            self.__dict__['_dead'] = True
            raise WorkerDied(f'worker process is gone ({e})') from None

    def _recv(self):
        try:
            status, value, tb = self._conn.recv()
        except (OSError, EOFError) as e:
            self.__dict__['_dead'] = True
            raise WorkerDied(f'worker process exited (code {self._proc.poll()})') from None
        if status == 'err':
            # The worker's traceback, since the exception alone says little.
            if not isinstance(value, BaseException):
                value = RuntimeError(value)
            value.remote_traceback = tb
            raise value
        return value

    def _drain(self):
        while self._pending:
            self.__dict__['_pending'] -= 1
            self._recv()

    def _rpc(self, msg):
        if self._dead:
            raise WorkerDied('worker process is gone')
        with self._lock:
            self._drain()
            self._send(msg)
            return self._recv()

    def call(self, method, *args, **kwargs):
        """Forward a method call; any snapshot is dropped as possibly stale."""
        self.__dict__['_snapshot'] = None
        snap = self.snapshot_after.get(method)
        if snap is None:
            return self._rpc(('call', method, args, kwargs))
        result, snapshot = self._rpc(('call_snap', method, args, kwargs, snap))
        self.__dict__['_snapshot'] = snapshot
        return result

    # -- attribute protocol

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        snapshot = self.__dict__.get('_snapshot')
        if snapshot is not None and name in snapshot:
            kind, value = snapshot[name]
            if kind == 'missing':
                raise AttributeError(name)
            if kind == 'called':
                return lambda: value
            return value
        if name in self._callables or name in self.snapshot_after:
            return lambda *a, **k: self.call(name, *a, **k)
        self.__dict__['_snapshot'] = None
        kind, value = self._rpc(('get', name))
        if kind == 'missing':
            raise AttributeError(name)
        if kind == 'callable':
            self._callables.add(name)
            return lambda *a, **k: self.call(name, *a, **k)
        return value

    def __setattr__(self, name, value):
        self.__dict__['_snapshot'] = None
        self._rpc(('set', name, value))

    # -- lifetime

    @property
    def alive(self):
        return not self._dead and self._proc.poll() is None

    def recreate(self, *args, **kwargs):
        """Replace the object with a new one built by the same factory, in the
        same worker -- a rebuild without the second it takes to start one."""
        self.__dict__['_snapshot'] = None
        self._rpc(('create', self._factory, args, kwargs))

    def wait_ready(self):
        """Block until the object is built (re-raising its construction error)."""
        if self._dead:
            raise WorkerDied('worker process is gone')
        with self._lock:
            self._drain()

    def close(self):
        d = self.__dict__
        if d.get('_dead', True):
            return
        d['_dead'] = True
        try:
            self._conn.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Worker side
# ----------------------------------------------------------------------

def _reply(conn, status, value, tb=None):
    try:
        conn.send((status, value, tb))
    except Exception as e:
        # An unpicklable result or exception: send what can be said.
        conn.send(('err', RuntimeError(f'{type(value).__name__}: {value} ({e})'), tb))


def _worker_main(fd):
    conn = Connection(fd)
    obj = None
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        op = msg[0]
        try:
            if op == 'create':
                _, factory, args, kwargs = msg
                old, obj = obj, None
                if old is not None and callable(getattr(old, 'close', None)):
                    old.close()             # e.g. a sim's physics client
                obj = _resolve(factory)(*args, **kwargs)
                result = None
            elif op == 'call':
                _, name, args, kwargs = msg
                result = getattr(obj, name)(*args, **kwargs)
            elif op == 'call_snap':
                _, name, args, kwargs, (attrs, calls) = msg
                result = getattr(obj, name)(*args, **kwargs)
                snapshot = {}
                for a in attrs:
                    v = getattr(obj, a, _MISSING)
                    snapshot[a] = ('missing', None) if v is _MISSING else ('value', v)
                for c in calls:
                    snapshot[c] = ('called', getattr(obj, c)())
                result = (result, snapshot)
            elif op == 'get':
                _, name = msg
                v = getattr(obj, name, _MISSING)
                if v is _MISSING:
                    result = ('missing', None)
                elif callable(v):
                    result = ('callable', None)
                else:
                    result = ('value', v)
            elif op == 'set':
                _, name, value = msg
                setattr(obj, name, value)
                result = None
            else:
                raise ValueError(f'unknown op {op!r}')
        except BaseException as e:
            _reply(conn, 'err', e, traceback.format_exc())
            continue
        _reply(conn, 'ok', result)


if __name__ == '__main__':
    _worker_main(int(sys.argv[1]))
