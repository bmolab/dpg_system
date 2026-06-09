"""
digico_oscquery_proxy.py — OSCQuery proxy for the DiGiCo S21/S31 (S-Series)

The console speaks OSC (UDP) but not OSCQuery, and — unlike ETC Eos — has no
query API to enumerate its capabilities. It does have two things we can use:

  * "Resend All OSC commands" (OSC Commands view) pushes EVERY send-enabled
    command with its current value to the active controller.
  * The OSC command list can be exported to USB as a file.

So this proxy discovers the console by LISTENING. It registers as the
console's OSC controller; when the console transmits (Resend All, or any live
parameter change) the proxy records each address, infers the parameter type
and range from the arguments, and grows an OSCQuery tree that mirrors the
console's actual namespace. String parameters named "name"/"label" become the
DESCRIPTION of their parent (so channels show their console labels). The
learned tree is persisted, so discovery is a one-time "press Resend All".

The tree is served over HTTP and advertised via mDNS as _oscjson._tcp —
the contract dpg_system's OSCQueryBrowser / oscq_browse expects. Any branch
can be searched and instantiated as a complete interface. Client OSC arriving
on the advertised UDP port is forwarded verbatim to the console; console
output updates tree VALUEs and is pushed to peers registered via the HTTP
/subscribe endpoint.

Console setup (S-Series: Extensions > OSC Control):
  - Enable OSC; set the console receive port (--console-port here)
  - Add a controller: IP = this machine, send port = --listen-port here;
    enable its send and receive. NOTE: only ONE controller can be active at a
    time — this proxy must be it (dpg_system talks to the proxy, not the desk).
  - OSC Commands view > "Resend All" to teach the proxy the full namespace.

Usage:
  python digico_oscquery_proxy.py --console-ip 10.0.1.50
  python digico_oscquery_proxy.py --console-ip 10.0.1.50 --console-port 8001 --listen-port 8002
  python digico_oscquery_proxy.py --import-commands "OSC Commands.txt"   # USB export
  python digico_oscquery_proxy.py --mock                                 # no console

Dependencies: python-osc, zeroconf, and oscquery_service.py (from dpg_system)
either importable as dpg_system.oscquery_service or sitting beside this file.
"""

import argparse
import json
import os
import re
import sys
import threading
import time

from pythonosc import dispatcher as osc_dispatcher
from pythonosc import osc_server
from pythonosc.udp_client import SimpleUDPClient

try:
    from dpg_system.oscquery_service import OSCQueryServer
except ImportError:
    from oscquery_service import OSCQueryServer


# ---------------------------------------------------------------------------
# Minimal OSCQuery registry (duck-types what OSCQueryServer needs: .registry)
# (same shape as the one in eos_oscquery_proxy.py — kept local so each proxy
# deploys as just itself + oscquery_service.py)
# ---------------------------------------------------------------------------

class ProxyRegistry:
    def __init__(self, description='DiGiCo OSCQuery Proxy'):
        self.registry = {
            'DESCRIPTION': description,
            'FULL_PATH': '/',
            'CONTENTS': {},
        }
        self.lock = threading.Lock()

    @staticmethod
    def _split(path):
        return [c for c in path.split('/') if c]

    def _container_for(self, components, create=True):
        node = self.registry
        path = ''
        for comp in components:
            path += '/' + comp
            contents = node.setdefault('CONTENTS', {})
            if comp not in contents:
                if not create:
                    return None
                contents[comp] = {'DESCRIPTION': comp, 'FULL_PATH': path, 'CONTENTS': {}}
            node = contents[comp]
        return node

    def ensure_container(self, path, description=None):
        with self.lock:
            node = self._container_for(self._split(path))
            if description is not None:
                node['DESCRIPTION'] = description
            return node

    def add_param(self, path, type_str, access=3, description=None, value=None,
                  range_=None, widget=None):
        components = self._split(path)
        with self.lock:
            node = self._container_for(components)
            node['TYPE'] = type_str
            node['ACCESS'] = access
            node['FULL_PATH'] = '/' + '/'.join(components)
            if description is not None or 'DESCRIPTION' not in node:
                node['DESCRIPTION'] = description if description is not None else components[-1]
            if value is not None:
                node['VALUE'] = value if isinstance(value, list) else [value]
            if range_ is not None:
                node['RANGE'] = range_
            if widget is not None:
                node['WIDGET'] = widget
            return node

    def get_param(self, path):
        with self.lock:
            return self._container_for(self._split(path), create=False)

    def set_value(self, path, value):
        with self.lock:
            node = self._container_for(self._split(path), create=False)
            if node is not None and 'TYPE' in node:
                node['VALUE'] = value if isinstance(value, list) else [value]

    def set_description(self, path, description):
        with self.lock:
            node = self._container_for(self._split(path), create=False)
            if node is not None:
                node['DESCRIPTION'] = description


# ---------------------------------------------------------------------------
# Learning: infer OSCQuery params from observed console traffic
# ---------------------------------------------------------------------------

NAME_LEAF_RE = re.compile(r'(name|label)$', re.IGNORECASE)


class DigicoLearner:
    """Builds and refines the tree from every message the console sends."""

    def __init__(self, registry, cache_path=None):
        self.registry = registry
        self.cache_path = cache_path
        # learned: address -> {'types': str, 'min': [..], 'max': [..], 'value': [..]}
        self.learned = {}
        self.lock = threading.Lock()
        self._dirty = False
        self._save_timer = None

    @staticmethod
    def _type_letter(arg):
        if isinstance(arg, bool):
            return 'T' if arg else 'F'
        if isinstance(arg, int):
            return 'i'
        if isinstance(arg, float):
            return 'f'
        if isinstance(arg, str):
            return 's'
        return 'b'

    def observe(self, address, args):
        """Record one console message; create or refine the tree node."""
        if not address.startswith('/'):
            return
        with self.lock:
            entry = self.learned.get(address)
            types = ''.join(self._type_letter(a) for a in args) or 'N'
            if entry is None:
                entry = {'types': types, 'min': [], 'max': [], 'value': list(args)}
                if types not in ('N', 's') and all(t in 'if' for t in types):
                    entry['min'] = [a for a in args]
                    entry['max'] = [a for a in args]
                self.learned[address] = entry
                created = True
            else:
                created = False
                # promote int->float if we ever see a float (and vice versa keep f)
                if len(types) == len(entry['types']):
                    merged = ''.join(
                        'f' if 'f' in (a, b) else (a if a == b else 's')
                        for a, b in zip(types, entry['types']))
                    # T/F pairs merge to T (toggle)
                    merged = ''.join(
                        'T' if {a, b} <= {'T', 'F'} else m
                        for a, b, m in zip(types, entry['types'], merged))
                    entry['types'] = merged
                entry['value'] = list(args)
                if entry['min'] and all(t in 'if' for t in entry['types']):
                    for i, a in enumerate(args[:len(entry['min'])]):
                        if isinstance(a, (int, float)):
                            entry['min'][i] = min(entry['min'][i], a)
                            entry['max'][i] = max(entry['max'][i], a)
            self._dirty = True
        self._apply(address, self.learned[address])
        if created:
            print(f"DigicoLearner: learned {address} ({self.learned[address]['types']}) "
                  f"= {list(args)}")
        self._schedule_save()

    def _apply(self, address, entry):
        """Write/refresh one learned address into the OSCQuery tree."""
        types = entry['types']
        value = entry['value']
        if types == 'N':
            self.registry.add_param(address, 'N', access=2)
            return
        # normalize T/F sequences to a stored boolean param
        if all(t in 'TF' for t in types):
            self.registry.add_param(address, 'T', access=3,
                                    value=[bool(v) for v in value])
            return
        if types == 's':
            node = self.registry.add_param(address, 's', access=3,
                                           value=[str(value[0]) if value else ''])
            # a "name"/"label" leaf labels its parent container
            components = [c for c in address.split('/') if c]
            if len(components) >= 2 and NAME_LEAF_RE.search(components[-1]) and value:
                parent = '/' + '/'.join(components[:-1])
                if str(value[0]).strip():
                    self.registry.set_description(parent, str(value[0]))
            return
        if all(t in 'if' for t in types):
            range_ = None
            if entry['min']:
                range_ = []
                for i in range(len(types)):
                    lo = entry['min'][i] if i < len(entry['min']) else 0
                    hi = entry['max'][i] if i < len(entry['max']) else 1
                    # observed range is a floor; give controls sensible defaults
                    if types[i] == 'f':
                        lo, hi = min(lo, 0.0), max(hi, 1.0)
                    elif {lo, hi} <= {0, 1}:
                        lo, hi = 0, 1
                    range_.append({'MIN': lo, 'MAX': hi})
            self.registry.add_param(address, types, access=3,
                                    value=list(value), range_=range_)
            return
        # mixed/unknown: expose as message-style string
        self.registry.add_param(address, 's', access=3, widget='message',
                                value=[' '.join(str(v) for v in value)])

    def rebuild_tree(self):
        for address, entry in self.learned.items():
            self._apply(address, entry)

    # -------- persistence --------

    def _schedule_save(self):
        if not self.cache_path:
            return
        if self._save_timer is not None:
            self._save_timer.cancel()
        self._save_timer = threading.Timer(3.0, self.save)
        self._save_timer.daemon = True
        self._save_timer.start()

    def save(self):
        if not self.cache_path:
            return
        with self.lock:
            if not self._dirty:
                return
            data = {'saved': time.time(), 'learned': self.learned}
            self._dirty = False
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(data, f, indent=1)
        except OSError as e:
            print(f"DigicoLearner: could not save cache: {e}")

    def load(self):
        if not self.cache_path or not os.path.exists(self.cache_path):
            return False
        try:
            with open(self.cache_path, 'r') as f:
                data = json.load(f)
            with self.lock:
                self.learned = data.get('learned', {})
            self.rebuild_tree()
            print(f"DigicoLearner: loaded {len(self.learned)} learned commands "
                  f"from cache (saved {time.ctime(data.get('saved', 0))})")
            return True
        except (OSError, ValueError) as e:
            print(f"DigicoLearner: could not load cache: {e}")
            return False

    # -------- command-file import --------

    def import_command_file(self, path):
        """Best-effort import of the console's exported OSC command list.
        Treats the first /-rooted token on each line as an address; parameters
        get a neutral float spec until real traffic refines them."""
        count = 0
        try:
            with open(path, 'r', errors='replace') as f:
                for line in f:
                    token = next((t for t in line.split() if t.startswith('/')), None)
                    if token is None:
                        continue
                    with self.lock:
                        if token in self.learned:
                            continue
                        self.learned[token] = {'types': 'f', 'min': [0.0],
                                               'max': [1.0], 'value': [0.0]}
                        self._dirty = True
                    self._apply(token, self.learned[token])
                    count += 1
        except OSError as e:
            print(f"DigicoLearner: could not read command file '{path}': {e}")
            return
        print(f"DigicoLearner: imported {count} commands from '{path}' "
              f"(types refine as the console sends real values)")
        self._schedule_save()


# ---------------------------------------------------------------------------
# The proxy: console link (UDP both ways) + client relay + OSCQuery server
# ---------------------------------------------------------------------------

SNAPSHOT_SEEDS = [
    ('/digico/snapshots/fire', 'i', 'Fire snapshot by number',
     [{'MIN': 0, 'MAX': 998}]),
    ('/digico/snapshots/fire/next', 'N', 'Fire next snapshot', None),
    ('/digico/snapshots/fire/previous', 'N', 'Fire previous snapshot', None),
]


class DigicoProxy:
    def __init__(self, registry, learner, console_ip=None, console_port=8001,
                 listen_port=8002):
        self.registry = registry
        self.learner = learner
        self.console_ip = console_ip
        self.console_port = console_port
        self.listen_port = listen_port
        self.console_client = None
        self.console_server = None
        self.oscquery_server = None  # set by main()

        if console_ip:
            self.console_client = SimpleUDPClient(console_ip, console_port)

    def seed_tree(self):
        self.registry.ensure_container('/digico', 'DiGiCo S21 console')
        for address, type_str, description, range_ in SNAPSHOT_SEEDS:
            self.registry.add_param(address, type_str, access=2,
                                    description=description, range_=range_)

    # ---- console -> proxy ----

    def start_console_listener(self):
        disp = osc_dispatcher.Dispatcher()
        disp.set_default_handler(self._from_console)
        self.console_server = osc_server.ThreadingOSCUDPServer(
            ('0.0.0.0', self.listen_port), disp)
        threading.Thread(target=self.console_server.serve_forever,
                         daemon=True).start()
        print(f"DigicoProxy: listening for console OSC on UDP {self.listen_port}")

    def _from_console(self, address, *args):
        self.learner.observe(address, list(args))
        if self.oscquery_server is not None:
            self.oscquery_server.notify_subscribers(address, list(args))

    # ---- clients -> console ----

    def from_client(self, client_address, address, *args):
        args = list(args)
        # argument-less bangs on seeded int commands need an explicit 0
        if not args:
            node = self.registry.get_param(address)
            if node is not None and node.get('TYPE') == 'N' and \
                    address.startswith('/digico/snapshots/'):
                args = [0]
        if args:
            self.registry.set_value(address, args)
        if self.console_client is not None:
            self.console_client.send_message(address, args)
        else:
            print(f"DigicoProxy (mock): {address} {args}")

    # ---- mock show ----

    def load_mock_console(self):
        self.registry.ensure_container('/digico', 'DiGiCo S21 (mock, no console)')
        names = {1: 'kick', 2: 'snare', 3: 'hat', 4: 'bass', 5: 'gtr L', 6: 'gtr R',
                 7: 'keys', 8: 'vox lead', 9: 'vox bv 1', 10: 'vox bv 2'}
        for ch in range(1, 17):
            base = f'/digico/channel/{ch}'
            self.learner.observe(f'{base}/fader', [0.0])
            self.learner.observe(f'{base}/mute', [0])
            self.learner.observe(f'{base}/name', [names.get(ch, f'ch {ch}')])
            for aux in (70, 71):
                self.learner.observe(f'{base}/aux/{aux}/send_level', [0.0])
        for cg in range(110, 114):
            self.learner.observe(f'/digico/channel/{cg}/fader', [0.0])
            self.learner.observe(f'/digico/channel/{cg}/mute', [0])
        self.learner.observe('/digico/channel/120/fader', [0.8])
        self.learner.observe('/digico/channel/120/name', ['master'])
        print("DigicoProxy: mock console loaded (addresses are illustrative — "
              "the real S21 namespace is learned from the console itself)")


class ClientRelay:
    """UDP server on the OSCQuery-advertised port; clients (dpg_system) send here."""

    def __init__(self, port, proxy):
        self.port = port
        self.proxy = proxy
        self.server = None

    def start(self):
        disp = osc_dispatcher.Dispatcher()
        disp.set_default_handler(self.proxy.from_client, needs_reply_address=True)
        self.server = osc_server.ThreadingOSCUDPServer(('0.0.0.0', self.port), disp)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"ClientRelay: listening for client OSC on UDP {self.port}")

    def stop(self):
        if self.server:
            self.server.shutdown()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        description='OSCQuery proxy for DiGiCo S21/S31 consoles')
    parser.add_argument('--console-ip', help='console IP address')
    parser.add_argument('--console-port', type=int, default=8001,
                        help='console OSC receive port (default 8001)')
    parser.add_argument('--listen-port', type=int, default=8002,
                        help="this proxy's port for console OSC — the controller "
                             'send port configured on the console (default 8002)')
    parser.add_argument('--osc-port', type=int, default=8766,
                        help='UDP port advertised to OSCQuery clients (default 8766)')
    parser.add_argument('--name', default='digico',
                        help='mDNS service name to advertise (default "digico")')
    parser.add_argument('--cache', default='digico_proxy_cache.json',
                        help='learned-command cache file ("" to disable)')
    parser.add_argument('--import-commands', metavar='FILE',
                        help="import the console's exported OSC command list")
    parser.add_argument('--mock', action='store_true',
                        help='serve a fake console for testing without hardware')
    args = parser.parse_args()

    if not args.mock and not args.console_ip:
        parser.error('--console-ip is required (or use --mock)')

    registry = ProxyRegistry()
    learner = DigicoLearner(registry, cache_path=args.cache or None)
    proxy = DigicoProxy(registry, learner,
                        console_ip=None if args.mock else args.console_ip,
                        console_port=args.console_port,
                        listen_port=args.listen_port)
    proxy.seed_tree()
    learner.load()
    if args.import_commands:
        learner.import_command_file(args.import_commands)

    if args.mock:
        proxy.load_mock_console()
    else:
        proxy.start_console_listener()

    server = OSCQueryServer(registry, default_osc_port=args.osc_port,
                            service_name=args.name)
    server.start()
    proxy.oscquery_server = server

    relay = ClientRelay(args.osc_port, proxy)
    relay.start()

    print(f"digico_oscquery_proxy: serving '{args.name}' "
          f"(HTTP {server.http_port}, OSC UDP {args.osc_port})"
          + (" [MOCK]" if args.mock else
             f" -> console {args.console_ip}:{args.console_port}, "
             f"learning from console on UDP {args.listen_port}"))
    if not args.mock and not learner.learned:
        print("digico_oscquery_proxy: tree is empty — on the console, open "
              "Extensions > OSC Control, add this machine as the controller "
              f"(send port {args.listen_port}, receive port {args.console_port}), "
              "then press 'Resend All' in the OSC Commands view to teach the proxy.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\ndigico_oscquery_proxy: shutting down")
        learner.save()
        relay.stop()
        server.stop()


if __name__ == '__main__':
    main()