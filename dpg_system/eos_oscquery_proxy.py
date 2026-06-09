"""
eos_oscquery_proxy.py — OSCQuery proxy for ETC Eos-family consoles (Gio @5 etc.)

The console speaks OSC but not OSCQuery. This proxy runs on any machine on the
lighting network and:

1. Connects to the console over TCP (port 3032) and enumerates the show's
   contents via the Eos "OSC Get" API (/eos/get/<type>/count + index queries):
   patched channels, cue lists, cues, groups, submasters, macros, presets,
   palettes and snapshots — each with its console label.
2. Builds an OSCQuery JSON tree of real Eos control addresses from that data
   and serves it over HTTP, advertised via mDNS as _oscjson._tcp — exactly the
   contract dpg_system's OSCQueryBrowser / oscq_browse expects. Any branch of
   the tree (/eos/chan, /eos/cue/1, /eos/macro ...) can be instantiated as a
   complete interface.
3. Relays OSC: UDP messages from clients are forwarded to the console over
   TCP; /eos/out/... feedback from the console is pushed to peers that
   registered via the HTTP /subscribe endpoint.
4. Subscribes to console change notifications (/eos/subscribe=1) and
   re-enumerates a branch when /eos/out/notify/... reports edits, so the
   served tree tracks the show file.

Console setup (Eos: Setup > System Settings > Show Control > OSC):
  - OSC RX and OSC TX enabled
  - OSC TCP format: "OSC 1.0" (packet-length framing, the default here)
    or "OSC 1.1" (SLIP framing, use --slip)
  - Third-party clients connect to TCP port 3032

Usage:
  python eos_oscquery_proxy.py --eos-ip 10.101.90.101            # real console
  python eos_oscquery_proxy.py --eos-ip 10.101.90.101 --slip     # OSC 1.1 mode
  python eos_oscquery_proxy.py --mock                            # fake show, no console

Dependencies: python-osc, zeroconf, and oscquery_service.py (from dpg_system)
either importable as dpg_system.oscquery_service or sitting beside this file.
"""

import argparse
import json
import os
import re
import socket
import struct
import sys
import threading
import time
import traceback

from pythonosc.osc_message import OscMessage
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_bundle import OscBundle
from pythonosc import dispatcher as osc_dispatcher
from pythonosc import osc_server
from pythonosc.udp_client import SimpleUDPClient

try:
    from dpg_system.oscquery_service import OSCQueryServer
except ImportError:
    from oscquery_service import OSCQueryServer


# ---------------------------------------------------------------------------
# Minimal OSCQuery registry (duck-types what OSCQueryServer needs: .registry)
# ---------------------------------------------------------------------------

class EosRegistry:
    """Holds the OSCQuery JSON tree. The dict object is mutated in place so the
    HTTP server (which keeps a reference) always serves the current tree."""

    def __init__(self, description='ETC Eos OSCQuery Proxy'):
        self.registry = {
            'DESCRIPTION': description,
            'FULL_PATH': '/',
            'CONTENTS': {},
        }
        self.lock = threading.Lock()

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

    @staticmethod
    def _split(path):
        return [c for c in path.split('/') if c]

    def ensure_container(self, path, description=None):
        with self.lock:
            node = self._container_for(self._split(path))
            if description is not None:
                node['DESCRIPTION'] = description
            return node

    def add_param(self, path, type_str, access=3, description=None, value=None,
                  range_=None, widget=None):
        """Insert a leaf parameter. range_ is a list of {'MIN':..,'MAX':..} or
        {'VALS': [...]} dicts (one per element of type_str)."""
        components = self._split(path)
        with self.lock:
            node = self._container_for(components)
            node['TYPE'] = type_str
            node['ACCESS'] = access
            node['FULL_PATH'] = '/' + '/'.join(components)
            node['DESCRIPTION'] = description if description is not None else components[-1]
            if value is not None:
                node['VALUE'] = value if isinstance(value, list) else [value]
            if range_ is not None:
                node['RANGE'] = range_
            if widget is not None:
                node['WIDGET'] = widget
            return node

    def set_value(self, path, value):
        with self.lock:
            node = self._container_for(self._split(path), create=False)
            if node is not None and 'TYPE' in node:
                node['VALUE'] = value if isinstance(value, list) else [value]

    def clear_branch(self, path):
        """Empty a container's CONTENTS (in place) so it can be rebuilt."""
        with self.lock:
            node = self._container_for(self._split(path))
            node.setdefault('CONTENTS', {}).clear()


# ---------------------------------------------------------------------------
# OSC over TCP — Eos supports OSC 1.0 (4-byte length prefix) and 1.1 (SLIP)
# ---------------------------------------------------------------------------

SLIP_END = 0xC0
SLIP_ESC = 0xDB
SLIP_ESC_END = 0xDC
SLIP_ESC_ESC = 0xDD


def slip_encode(data):
    out = bytearray([SLIP_END])
    for b in data:
        if b == SLIP_END:
            out += bytes([SLIP_ESC, SLIP_ESC_END])
        elif b == SLIP_ESC:
            out += bytes([SLIP_ESC, SLIP_ESC_ESC])
        else:
            out.append(b)
    out.append(SLIP_END)
    return bytes(out)


class EosTCPLink:
    """Persistent TCP OSC connection to the console with auto-reconnect.

    on_message(address, args) is called from the receive thread for every
    OSC message the console sends. on_connect() is called after each
    (re)connection so the owner can re-handshake.
    """

    def __init__(self, ip, port=3032, slip=False, on_message=None, on_connect=None):
        self.ip = ip
        self.port = port
        self.slip = slip
        self.on_message = on_message
        self.on_connect = on_connect
        self.sock = None
        self.send_lock = threading.Lock()
        self.connected = threading.Event()
        self._stop = False
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        self._close()

    def _close(self):
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        self.connected.clear()

    def _run(self):
        backoff = 1.0
        while not self._stop:
            try:
                sock = socket.create_connection((self.ip, self.port), timeout=5)
                sock.settimeout(None)
                self.sock = sock
                self.connected.set()
                backoff = 1.0
                print(f"EosTCPLink: connected to {self.ip}:{self.port} "
                      f"({'SLIP/1.1' if self.slip else 'packet-length/1.0'})")
                if self.on_connect:
                    threading.Thread(target=self.on_connect, daemon=True).start()
                self._receive_loop(sock)
            except OSError as e:
                if not self._stop:
                    print(f"EosTCPLink: connection failed/lost ({e}); retrying in {backoff:.0f}s")
            self._close()
            if self._stop:
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 15.0)

    def _receive_loop(self, sock):
        buffer = bytearray()
        while not self._stop:
            data = sock.recv(65536)
            if not data:
                raise OSError("connection closed by console")
            buffer += data
            for packet in self._extract_packets(buffer):
                self._dispatch_packet(packet)

    def _extract_packets(self, buffer):
        packets = []
        if self.slip:
            while True:
                try:
                    end = buffer.index(SLIP_END)
                except ValueError:
                    break
                frame = bytes(buffer[:end])
                del buffer[:end + 1]
                if frame:
                    frame = frame.replace(bytes([SLIP_ESC, SLIP_ESC_END]), bytes([SLIP_END]))
                    frame = frame.replace(bytes([SLIP_ESC, SLIP_ESC_ESC]), bytes([SLIP_ESC]))
                    packets.append(frame)
        else:
            while len(buffer) >= 4:
                length = struct.unpack('>I', buffer[:4])[0]
                if length > 10_000_000:  # framing desync — drop the buffer
                    print("EosTCPLink: implausible packet length, resyncing")
                    buffer.clear()
                    break
                if len(buffer) < 4 + length:
                    break
                packets.append(bytes(buffer[4:4 + length]))
                del buffer[:4 + length]
        return packets

    def _dispatch_packet(self, packet):
        try:
            if packet[:8] == b'#bundle\x00':
                self._dispatch_bundle(OscBundle(packet))
            else:
                msg = OscMessage(packet)
                if self.on_message:
                    self.on_message(msg.address, list(msg.params))
        except Exception as e:
            print(f"EosTCPLink: failed to parse packet: {e}")

    def _dispatch_bundle(self, bundle):
        for content in bundle:
            if isinstance(content, OscBundle):
                self._dispatch_bundle(content)
            elif self.on_message:
                self.on_message(content.address, list(content.params))

    def send(self, address, args=None):
        if not self.connected.is_set() or self.sock is None:
            return False
        builder = OscMessageBuilder(address=address)
        for arg in (args or []):
            builder.add_arg(arg)
        data = builder.build().dgram
        if self.slip:
            framed = slip_encode(data)
        else:
            framed = struct.pack('>I', len(data)) + data
        try:
            with self.send_lock:
                self.sock.sendall(framed)
            return True
        except OSError as e:
            print(f"EosTCPLink: send failed ({e})")
            self._close()
            return False


# ---------------------------------------------------------------------------
# Eos show-data enumeration and tree building
# ---------------------------------------------------------------------------

NUMERIC_RE = re.compile(r'^\d+(\.\d+)?$')

# Curated keys exposed as buttons under /eos/key/.
# Eos treats an argument-less OSC command as 1.0, so a bang = key press.
EOS_KEYS = [
    'go_0', 'stop', 'update', 'record', 'enter', 'clear_cmdline',
    'at', 'full', 'out', 'last', 'next', 'select_last', 'select_active',
    'blackout', 'highlight', 'label',
]

# type -> (eos get name, tree branch under /eos, human label)
SIMPLE_FIRE_TYPES = [
    ('macro', 'macro', 'Macros'),
    ('preset', 'preset', 'Presets'),
    ('ip', 'ip', 'Intensity Palettes'),
    ('fp', 'fp', 'Focus Palettes'),
    ('cp', 'cp', 'Color Palettes'),
    ('bp', 'bp', 'Beam Palettes'),
    ('snap', 'snap', 'Snapshots'),
]


def numeric_sort_key(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('inf')


class EosProxy:
    def __init__(self, registry, link=None, enum_timeout=10.0, cache_path=None):
        self.registry = registry
        self.link = link
        self.enum_timeout = enum_timeout
        self.cache_path = cache_path
        self.oscquery_server = None  # set by main()

        # show data: type -> {number(str): label}; cues: list# -> {cue#: label}
        self.show = {}
        self.cues = {}
        self.show_name = ''

        # enumeration plumbing
        self._count_events = {}    # request path -> (Event, [count])
        self._collect = None       # active collection dict or None
        self._collect_lock = threading.Lock()

        self._notify_pending = set()
        self._notify_timer = None
        self._enum_lock = threading.Lock()

    # ---------------- console message handling ----------------

    def handle_console_message(self, address, args):
        """Called from the TCP receive thread for every console message."""
        if address.startswith('/eos/out/get/'):
            self._handle_get_reply(address, args)
        elif address.startswith('/eos/out/notify/'):
            self._handle_notify(address)
        elif address == '/eos/out/show/name':
            if args and isinstance(args[0], str):
                if self.show_name and args[0] != self.show_name:
                    print(f"EosProxy: show changed to '{args[0]}', re-enumerating")
                    threading.Thread(target=self.enumerate_all, daemon=True).start()
                self.show_name = args[0]
                self.registry.ensure_container('/eos', f"Eos: {self.show_name}")

        # live feedback worth mirroring into the tree
        if address == '/eos/out/cmd' and args:
            self.registry.set_value('/eos/out/cmd', [str(args[0])])
        elif address == '/eos/out/active/cue/text' and args:
            self.registry.set_value('/eos/out/active/cue/text', [str(args[0])])
        elif address == '/eos/out/pending/cue/text' and args:
            self.registry.set_value('/eos/out/pending/cue/text', [str(args[0])])

        # push all console output to subscribed peers
        if address.startswith('/eos/out/') and self.oscquery_server is not None:
            self.oscquery_server.notify_subscribers(address, args)

    def _handle_get_reply(self, address, args):
        tail = address[len('/eos/out/get/'):]
        components = tail.split('/')

        # count reply: /eos/out/get/<type>[/...]/count
        if components[-1] == 'count':
            request = '/eos/get/' + tail
            entry = self._count_events.pop(request, None)
            if entry is not None and args:
                event, holder = entry
                holder[0] = int(args[0])
                event.set()
            return

        # item reply: /eos/out/get/<type>/<numbers...>/list/<chunk>/<total>
        try:
            list_pos = components.index('list')
        except ValueError:
            return
        if components[list_pos + 1] != '0':
            return  # continuation chunk; label was in chunk 0
        type_name = components[0]
        numbers = components[1:list_pos]
        if not numbers or not all(NUMERIC_RE.match(n) for n in numbers):
            return  # secondary reply such as .../links/list/... — not the item itself
        label = ''
        if len(args) >= 3 and isinstance(args[2], str):
            label = args[2]

        with self._collect_lock:
            collect = self._collect
            if collect is None or collect['type'] != type_name:
                return
            if type_name == 'patch':
                key = numbers[0]  # channel; later parts just repeat the channel
            elif type_name == 'cue':
                if numbers[0] != collect.get('cuelist'):
                    return
                if len(numbers) >= 3 and numbers[2] != '0':
                    return  # cue part, not the base cue
                key = numbers[1]
            else:
                key = numbers[0]
            if key not in collect['items']:
                collect['items'][key] = label
                if len(collect['items']) >= collect['expected']:
                    collect['done'].set()

    def _handle_notify(self, address):
        # /eos/out/notify/<type>/... — debounce, then re-enumerate that type
        components = address[len('/eos/out/notify/'):].split('/')
        if not components:
            return
        self._notify_pending.add(components[0])
        if self._notify_timer is not None:
            self._notify_timer.cancel()
        self._notify_timer = threading.Timer(2.0, self._process_notifies)
        self._notify_timer.daemon = True
        self._notify_timer.start()

    def _process_notifies(self):
        pending, self._notify_pending = self._notify_pending, set()
        print(f"EosProxy: show data changed ({', '.join(sorted(pending))}), re-enumerating")
        for type_name in pending:
            self.enumerate_and_build(type_name)
        self.save_cache()

    # ---------------- enumeration ----------------

    def _query_count(self, request_path):
        event = threading.Event()
        holder = [None]
        self._count_events[request_path] = (event, holder)
        self.link.send(request_path)
        if not event.wait(self.enum_timeout):
            self._count_events.pop(request_path, None)
            print(f"EosProxy: no count reply for {request_path}")
            return None
        return holder[0]

    def _enumerate_target(self, get_path, type_name, cuelist=None):
        """Run a count + index sweep. Returns {number(str): label} or None."""
        count = self._query_count(f'{get_path}/count')
        if count is None:
            return None
        if count == 0:
            return {}
        collect = {
            'type': type_name,
            'cuelist': cuelist,
            'items': {},
            'expected': count,
            'done': threading.Event(),
        }
        with self._collect_lock:
            self._collect = collect
        for start in range(0, count, 50):
            for i in range(start, min(start + 50, count)):
                self.link.send(f'{get_path}/index/{i}')
            time.sleep(0.05)  # don't flood the console
        collect['done'].wait(self.enum_timeout + count * 0.02)
        with self._collect_lock:
            self._collect = None
        got = len(collect['items'])
        if got < count:
            print(f"EosProxy: {get_path}: got {got}/{count} items before timeout")
        return collect['items']

    def enumerate_type(self, type_name):
        """Enumerate one show-data type into self.show / self.cues."""
        if type_name == 'patch':
            items = self._enumerate_target('/eos/get/patch', 'patch')
            if items is not None:
                self.show['patch'] = items
        elif type_name in ('cue', 'cuelist'):
            lists = self._enumerate_target('/eos/get/cuelist', 'cuelist')
            if lists is None:
                return
            self.show['cuelist'] = lists
            self.cues = {}
            for list_number in sorted(lists, key=numeric_sort_key):
                cues = self._enumerate_target(
                    f'/eos/get/cue/{list_number}', 'cue', cuelist=list_number)
                self.cues[list_number] = cues or {}
        else:
            items = self._enumerate_target(f'/eos/get/{type_name}', type_name)
            if items is not None:
                self.show[type_name] = items

    def enumerate_and_build(self, type_name):
        with self._enum_lock:
            self.enumerate_type(type_name)
            self.build_branch(type_name)

    def enumerate_all(self):
        if self.link is not None:
            self.link.connected.wait(10)
        types = ['patch', 'group', 'sub', 'cue'] + [t[0] for t in SIMPLE_FIRE_TYPES]
        print("EosProxy: enumerating show data from console...")
        started = time.time()
        for type_name in types:
            self.enumerate_and_build(type_name)
        counts = {k: len(v) for k, v in self.show.items()}
        counts['cues'] = sum(len(c) for c in self.cues.values())
        print(f"EosProxy: enumeration complete in {time.time() - started:.1f}s — {counts}")
        self.save_cache()

    def handshake(self):
        """Called on every (re)connect to the console."""
        self.link.send('/eos/ping', ['eos_oscquery_proxy'])
        self.link.send('/eos/subscribe', [1])
        self.enumerate_all()

    # ---------------- tree building ----------------

    def build_static_tree(self):
        reg = self.registry
        reg.ensure_container('/eos', 'ETC Eos console')

        reg.add_param('/eos/cmd', 's', access=3, widget='message',
                      description='Command line (terminated — runs immediately)')
        reg.add_param('/eos/newcmd', 's', access=3, widget='message',
                      description='Command line (replaces current command)')

        reg.ensure_container('/eos/key', 'Console keys')
        for key in EOS_KEYS:
            reg.add_param(f'/eos/key/{key}', 'N', access=2, description=f'{key} key')

        reg.ensure_container('/eos/out', 'Console feedback (read-only)')
        reg.add_param('/eos/out/cmd', 's', access=1, value=[''],
                      description='Command line text')
        reg.add_param('/eos/out/active/cue/text', 's', access=1, value=[''],
                      description='Active cue')
        reg.add_param('/eos/out/pending/cue/text', 's', access=1, value=[''],
                      description='Pending cue')

    def build_branch(self, type_name):
        reg = self.registry
        if type_name == 'patch':
            channels = self.show.get('patch', {})
            reg.clear_branch('/eos/chan')
            reg.ensure_container('/eos/chan', f'Channels ({len(channels)} patched)')
            for number in sorted(channels, key=numeric_sort_key):
                label = channels[number] or f'channel {number}'
                reg.add_param(f'/eos/chan/{number}', 'f', access=3, value=[0.0],
                              range_=[{'MIN': 0.0, 'MAX': 100.0}], description=label)

        elif type_name == 'group':
            groups = self.show.get('group', {})
            reg.clear_branch('/eos/group')
            reg.ensure_container('/eos/group', f'Groups ({len(groups)})')
            for number in sorted(groups, key=numeric_sort_key):
                label = groups[number] or f'group {number}'
                reg.add_param(f'/eos/group/{number}', 'f', access=3, value=[0.0],
                              range_=[{'MIN': 0.0, 'MAX': 100.0}], description=label)

        elif type_name == 'sub':
            subs = self.show.get('sub', {})
            reg.clear_branch('/eos/sub')
            reg.ensure_container('/eos/sub', f'Submasters ({len(subs)})')
            for number in sorted(subs, key=numeric_sort_key):
                label = subs[number] or f'sub {number}'
                reg.add_param(f'/eos/sub/{number}', 'f', access=3, value=[0.0],
                              range_=[{'MIN': 0.0, 'MAX': 1.0}], description=label)
                reg.add_param(f'/eos/sub/{number}/fire', 'F', access=2, value=[False],
                              description=f'{label} bump')

        elif type_name in ('cue', 'cuelist'):
            reg.clear_branch('/eos/cue')
            lists = self.show.get('cuelist', {})
            reg.ensure_container('/eos/cue', f'Cue lists ({len(lists)})')
            for list_number in sorted(lists, key=numeric_sort_key):
                list_label = lists[list_number] or f'cue list {list_number}'
                cues = self.cues.get(list_number, {})
                reg.ensure_container(f'/eos/cue/{list_number}',
                                     f'{list_label} ({len(cues)} cues)')
                for cue_number in sorted(cues, key=numeric_sort_key):
                    cue_label = cues[cue_number] or f'cue {cue_number}'
                    reg.add_param(f'/eos/cue/{list_number}/{cue_number}/fire', 'N',
                                  access=2, description=cue_label)

        else:
            for get_name, branch, branch_label in SIMPLE_FIRE_TYPES:
                if get_name != type_name:
                    continue
                items = self.show.get(get_name, {})
                reg.clear_branch(f'/eos/{branch}')
                reg.ensure_container(f'/eos/{branch}', f'{branch_label} ({len(items)})')
                for number in sorted(items, key=numeric_sort_key):
                    label = items[number] or f'{get_name} {number}'
                    reg.add_param(f'/eos/{branch}/{number}/fire', 'N', access=2,
                                  description=label)

    def build_all_branches(self):
        for type_name in ['patch', 'group', 'sub', 'cue'] + [t[0] for t in SIMPLE_FIRE_TYPES]:
            self.build_branch(type_name)

    # ---------------- cache ----------------

    def save_cache(self):
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, 'w') as f:
                json.dump({'show_name': self.show_name, 'saved': time.time(),
                           'show': self.show, 'cues': self.cues}, f, indent=1)
        except OSError as e:
            print(f"EosProxy: could not save cache: {e}")

    def load_cache(self):
        if not self.cache_path or not os.path.exists(self.cache_path):
            return False
        try:
            with open(self.cache_path, 'r') as f:
                data = json.load(f)
            self.show = data.get('show', {})
            self.cues = data.get('cues', {})
            self.show_name = data.get('show_name', '')
            self.build_all_branches()
            print(f"EosProxy: pre-loaded tree from cache "
                  f"('{self.show_name}', saved {time.ctime(data.get('saved', 0))})")
            return True
        except (OSError, ValueError) as e:
            print(f"EosProxy: could not load cache: {e}")
            return False

    # ---------------- mock show ----------------

    def load_mock_show(self):
        self.show_name = 'Mock Show'
        self.registry.ensure_container('/eos', 'Eos: Mock Show (no console)')
        chan_labels = ['house left warm', 'house right warm', 'cyc blue', 'cyc red',
                       'spot 1', 'spot 2', 'blinders', 'movers downstage']
        self.show['patch'] = {str(i + 1): (chan_labels[i] if i < len(chan_labels) else '')
                              for i in range(24)}
        self.show['group'] = {'1': 'all warm', '2': 'cyc', '5': 'movers'}
        self.show['sub'] = {str(i + 1): label for i, label in
                            enumerate(['warms', 'cyc', 'specials', 'fx bumps'])}
        self.show['cuelist'] = {'1': 'main', '2': 'preshow'}
        self.cues = {
            '1': {'1': 'house open', '2': 'blackout', '3': 'act one top',
                  '3.5': 'sneak cyc', '10': 'finale'},
            '2': {'1': 'walk-in loop'},
        }
        self.show['macro'] = {'1': 'channel check', '2': 'reset rig'}
        self.show['preset'] = {'1': 'center special'}
        self.show['ip'] = {'1': 'full'}
        self.show['fp'] = {'1': 'downstage center'}
        self.show['cp'] = {'1': 'amber', '2': 'congo'}
        self.show['bp'] = {'1': 'narrow'}
        self.show['snap'] = {'1': 'programming', '2': 'live'}
        self.build_all_branches()
        print("EosProxy: mock show loaded")


# ---------------------------------------------------------------------------
# UDP relay — clients (dpg_system) send OSC here; we forward to the console
# ---------------------------------------------------------------------------

class UDPRelay:
    def __init__(self, port, proxy):
        self.port = port
        self.proxy = proxy
        self.server = None
        self.thread = None

    def start(self):
        disp = osc_dispatcher.Dispatcher()
        disp.set_default_handler(self._handle, needs_reply_address=True)
        self.server = osc_server.ThreadingOSCUDPServer(('0.0.0.0', self.port), disp)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"UDPRelay: listening for client OSC on UDP {self.port}")

    def stop(self):
        if self.server:
            self.server.shutdown()

    def _handle(self, client_address, address, *args):
        proxy = self.proxy
        # mirror levels into the tree so a fresh fetch shows last-set values
        if re.match(r'^/eos/(chan|group|sub)/[\d.]+$', address) and args:
            try:
                proxy.registry.set_value(address, [float(args[0])])
            except (TypeError, ValueError):
                pass
        if proxy.link is not None:
            if not proxy.link.send(address, list(args)):
                print(f"UDPRelay: console not connected, dropped {address}")
        else:
            print(f"UDPRelay (mock): {address} {list(args)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    # logs go to journals/files when run as a service — don't sit in the buffer
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        description='OSCQuery proxy for ETC Eos-family consoles')
    parser.add_argument('--eos-ip', help='console IP address')
    parser.add_argument('--eos-port', type=int, default=3032,
                        help='console OSC TCP port (default 3032)')
    parser.add_argument('--slip', action='store_true',
                        help='use OSC 1.1 SLIP framing (console "OSC TCP format" setting)')
    parser.add_argument('--osc-port', type=int, default=8765,
                        help='UDP port this proxy listens on for client OSC (default 8765)')
    parser.add_argument('--name', default='eos',
                        help='mDNS service name to advertise (default "eos")')
    parser.add_argument('--cache', default='eos_proxy_cache.json',
                        help='enumeration cache file ("" to disable)')
    parser.add_argument('--mock', action='store_true',
                        help='serve a fake show without connecting to a console')
    args = parser.parse_args()

    if not args.mock and not args.eos_ip:
        parser.error('--eos-ip is required (or use --mock)')

    registry = EosRegistry()
    proxy = EosProxy(registry, cache_path=args.cache or None)
    proxy.build_static_tree()

    if args.mock:
        proxy.load_mock_show()
    else:
        proxy.load_cache()  # serve last-known tree immediately; live data replaces it
        link = EosTCPLink(args.eos_ip, args.eos_port, slip=args.slip,
                          on_message=proxy.handle_console_message,
                          on_connect=proxy.handshake)
        proxy.link = link
        link.start()

    server = OSCQueryServer(registry, default_osc_port=args.osc_port,
                            service_name=args.name)
    server.start()
    proxy.oscquery_server = server

    relay = UDPRelay(args.osc_port, proxy)
    relay.start()

    # keepalive ping so the console keeps the TCP session alive
    def keepalive():
        while True:
            time.sleep(10)
            if proxy.link is not None and proxy.link.connected.is_set():
                proxy.link.send('/eos/ping', ['eos_oscquery_proxy'])
    threading.Thread(target=keepalive, daemon=True).start()

    print(f"eos_oscquery_proxy: serving '{args.name}' "
          f"(HTTP {server.http_port}, OSC UDP {args.osc_port})"
          + (" [MOCK]" if args.mock else f" -> console {args.eos_ip}:{args.eos_port}"))
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\neos_oscquery_proxy: shutting down")
        relay.stop()
        server.stop()
        if proxy.link:
            proxy.link.stop()


if __name__ == '__main__':
    main()