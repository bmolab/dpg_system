"""
dpg_system node for Shelly Gen2+ smart plugs / relays (built against a
Shelly Plug US Gen4, S4PL-00116US, firmware 1.7.99).

    shelly <host> [password]

<host> is the plug's IP (give it a fixed one), a .local name, or the bare
device id (shellyplugusg4-98c37700dc94 -> shellyplugusg4-98c37700dc94.local).
The 'device' list on the node is filled from a network scan (mDNS
_shelly._tcp) when the node is created; 'scan' refreshes it. Picking an entry
retargets the node.

Everything goes through the device's local JSON-RPC endpoint (POST /rpc), so
no cloud account is involved. If the device has a password set (Shelly.SetAuth)
pass it as the second argument; Gen2+ uses SHA-256 digest auth as user 'admin'.

All network calls run on one worker thread per node, serialised through a
queue, and also poll Switch.GetStatus every 'poll interval' seconds so that
presses of the plug's own button show up in the patch. Results are applied
in frame_task() on the main thread.

Messages to the 'in' inlet:
    1 / 0 / on / off / toggle      switch
    on <s> / off <s>               switch, then flip back after <s> seconds
                                   (the plug's own timer: toggle_after)
    bang / refresh                 read status now
    rpc <Method> [json | key value ...]   any RPC call; reply -> 'reply' outlet
    scan                           rescan; device list -> 'reply' outlet
    host <host>                    retarget the node
    led <0-100>                    LED ring brightness (sets the 'led' slider)
"""

import json
import queue
import socket
import threading
import time

import dearpygui.dearpygui as dpg
import requests
from requests.auth import HTTPDigestAuth

from dpg_system.node import Node
from dpg_system.conversion_utils import *


def register_shelly_nodes():
    Node.app.register_node('shelly', ShellyNode.factory)


def normalise_host(host):
    host = (host or '').strip()
    if host.startswith('http://'):
        host = host[7:]
    host = host.rstrip('/')
    # A bare device id (no dots, not an IP) resolves over mDNS as <id>.local.
    if host and '.' not in host and ':' not in host:
        host += '.local'
    return host


def scan_shelly_devices(duration=5.0):
    """Browse mDNS for _shelly._tcp, then ask each device its name.
    Returns [[id, ip, app, gen, name], ...] sorted by ip."""
    from zeroconf import Zeroconf, ServiceBrowser
    found = {}

    class Listener:
        def add_service(self, zc, service_type, name):
            info = zc.get_service_info(service_type, name, 2000)
            if info is None:
                return
            props = {k.decode(): (v.decode() if isinstance(v, bytes) else v)
                     for k, v in info.properties.items()}
            addresses = [socket.inet_ntoa(a) for a in info.addresses if len(a) == 4]
            device_id = name.split('.')[0]
            found[device_id] = [device_id, addresses[0] if addresses else '',
                                props.get('app', ''), props.get('gen', '')]

        def update_service(self, *args):
            pass

        def remove_service(self, *args):
            pass

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, '_shelly._tcp.local.', Listener())
        time.sleep(duration)
    finally:
        zc.close()
    devices = list(found.values())
    for device in devices:
        name = ''
        if device[1]:
            try:
                info = requests.get(f'http://{device[1]}/rpc/Shelly.GetDeviceInfo', timeout=1.5).json()
                name = info.get('name') or ''
            except Exception:                        # noqa: BLE001 - name is optional
                pass
        device.append(name)

    def ip_key(device):
        try:
            return tuple(int(part) for part in device[1].split('.'))
        except ValueError:
            return (999,)
    return sorted(devices, key=ip_key)


def device_label(device):
    """'10.1.1.222  lamp' -- the ip first, so the list sorts and reads by address."""
    _id, ip, _app, _gen, name = device
    return f'{ip or _id}  {name or _id}'


class ShellyRPCError(Exception):
    pass


class ShellyNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = ShellyNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        default_host = ''
        default_password = ''
        if args is not None and len(args) > 0:
            default_host = any_to_string(args[0])
        if args is not None and len(args) > 1:
            default_password = any_to_string(args[1])

        # Plain attributes the worker reads; widget callbacks keep them current.
        self.host = normalise_host(default_host)
        self.password = default_password
        self.poll_interval = 1.0
        self.switch_id = 0

        self.commands = queue.Queue()
        self.events = queue.Queue()
        self.running = False
        self.worker = None
        self.session = requests.Session()
        self.rpc_id = 0
        self.last_output = None
        self.shown_status = None
        self.devices = []
        self.scanning = False
        # LED brightness: the worker sends only the latest slider value, and
        # reads the plug's own value back whenever the node is (re)targeted.
        self.led_target = None
        self.led_applied = None
        self.led_read_needed = True

        # Not labelled 'on': a message whose first word names a widget inlet is
        # routed to that widget, so 'on 3' would set the checkbox and lose the timer.
        self.device_property = self.add_property('device', widget_type='combo', width=260,
                                                 callback=self.device_chosen)
        self.on_input = self.add_input('switch', widget_type='checkbox', callback=self.on_changed)
        # Labelled 'led' on purpose: the message 'led 30' is routed to it.
        self.led_input = self.add_input('led', widget_type='slider_int', default_value=100,
                                        min=0, max=100, callback=self.led_changed)
        self.message_input = self.add_input('in', triggers_execution=True)

        self.status_label = self.add_label('')

        self.state_output = self.add_output('on')
        self.power_output = self.add_output('power (W)')
        self.status_output = self.add_output('status')
        self.reply_output = self.add_output('reply')

        self.host_option = self.add_option('host', widget_type='text_input',
                                           default_value=default_host, callback=self.host_changed)
        self.password_option = self.add_option('password', widget_type='text_input',
                                               default_value=default_password,
                                               callback=self.password_changed)
        self.poll_option = self.add_option('poll interval', widget_type='drag_float',
                                           default_value=1.0, min=0.0, max=60.0,
                                           callback=self.poll_changed)
        self.switch_id_option = self.add_option('switch id', widget_type='input_int',
                                                default_value=0, min=0, max=3,
                                                callback=self.switch_id_changed)
        self.print_debug_option = self.add_option('print debug', widget_type='checkbox',
                                                  default_value=False)

    # -- lifecycle ------------------------------------------------------

    def custom_create(self, from_file):
        self.set_status('no host' if not self.host else 'connecting...')
        self.running = True
        self.worker = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker.start()
        self.add_frame_task()
        self.show_host_in_list()
        self.start_scan(send_reply=False)

    def cleanup(self):
        self.running = False
        self.commands.put(('quit',))
        super().cleanup()

    # -- option callbacks (also fire on patch load, which is what we want:
    #    they only copy values into the worker's attributes) ---------------

    def host_changed(self):
        self.host = normalise_host(any_to_string(self.host_option()))
        self.last_output = None
        self.led_read_needed = True
        self.led_applied = None
        self.commands.put(('poll',))
        self.show_host_in_list()

    def password_changed(self):
        self.password = any_to_string(self.password_option())
        self.commands.put(('poll',))

    def poll_changed(self):
        self.poll_interval = max(0.0, any_to_float(self.poll_option()))
        self.commands.put(('poll',))

    def switch_id_changed(self):
        self.switch_id = any_to_int(self.switch_id_option())
        self.last_output = None
        self.commands.put(('poll',))

    # -- control --------------------------------------------------------

    def on_changed(self):
        # Patch load replays widget callbacks; never switch a real plug just
        # because a patch was opened. The first poll shows the real state.
        if self.in_loading_process:
            return
        self.commands.put(('set', bool(self.on_input()), None))

    def led_changed(self):
        # A saved patch does not push its brightness to the plug; the slider
        # shows the plug's own value once the node has read it.
        if self.in_loading_process:
            return
        self.led_target = max(0, min(100, any_to_int(self.led_input())))
        self.commands.put(('led',))

    def execute(self):
        data = self.message_input()
        if data is None:
            return
        if not isinstance(data, (list, tuple)):
            data = [data]
        data = list(data)
        if len(data) == 0:
            return
        head = data[0]
        rest = data[1:]

        if is_number(head) and not isinstance(head, str):
            self.commands.put(('set', any_to_float(head) != 0, None))
            return
        head = any_to_string(head).strip().lower()
        if head in ('on', 'off', 'true', 'false', '1', '0'):
            on = head in ('on', 'true', '1')
            after = any_to_float(rest[0]) if rest else None
            self.commands.put(('set', on, after if after and after > 0 else None))
        elif head == 'toggle':
            self.commands.put(('toggle',))
        elif head in ('bang', 'refresh', 'status'):
            self.commands.put(('poll',))
        elif head == 'scan':
            self.start_scan(send_reply=True)
        elif head == 'host' and rest:
            self.host_option.set(any_to_string(rest[0]))
            self.host_changed()
        elif head == 'rpc' and rest:
            method = any_to_string(rest[0])
            self.commands.put(('rpc', method, self.parse_params(rest[1:])))
        else:
            self.set_status(f'unknown message: {head}')

    @staticmethod
    def parse_params(tokens):
        """'{"id":0,"on":true}' as JSON, or key value key value ..."""
        if not tokens:
            return {}
        if len(tokens) == 1 and isinstance(tokens[0], dict):
            return tokens[0]
        text = ' '.join(any_to_string(t) for t in tokens).strip()
        if text.startswith('{'):
            try:
                return json.loads(text)
            except ValueError:
                pass
        params = {}
        for key, value in zip(tokens[0::2], tokens[1::2]):
            if isinstance(value, str):
                lowered = value.lower()
                if lowered in ('true', 'false'):
                    value = lowered == 'true'
                elif lowered == 'null':
                    value = None
                else:
                    try:
                        value = json.loads(value)
                    except ValueError:
                        pass
            params[any_to_string(key)] = value
        return params

    # -- worker thread --------------------------------------------------

    def rpc(self, method, params=None, timeout=3.0):
        if not self.host:
            raise ShellyRPCError('no host')
        self.rpc_id += 1
        body = {'id': self.rpc_id, 'method': method}
        if params:
            body['params'] = params
        auth = HTTPDigestAuth('admin', self.password) if self.password else None
        response = self.session.post(f'http://{self.host}/rpc', json=body,
                                     auth=auth, timeout=timeout)
        if response.status_code == 401:
            raise ShellyRPCError('password required' if not self.password else 'wrong password')
        response.raise_for_status()
        reply = response.json()
        if self.print_debug_option():
            print('shelly', method, params, '->', reply)
        if 'error' in reply:
            error = reply['error']
            raise ShellyRPCError(f"{error.get('code')}: {error.get('message')}")
        return reply.get('result')

    def worker_loop(self):
        next_poll = 0.0
        while self.running:
            wait = None
            if self.poll_interval > 0:
                wait = max(0.0, next_poll - time.monotonic())
            try:
                command = self.commands.get(timeout=wait)
            except queue.Empty:
                command = ('poll',)
            kind = command[0]
            if kind == 'quit':
                break
            try:
                if kind == 'set':
                    _, on, after = command
                    params = {'id': self.switch_id, 'on': on}
                    if after:
                        params['toggle_after'] = after
                    self.rpc('Switch.Set', params)
                    kind = 'poll'
                elif kind == 'toggle':
                    self.rpc('Switch.Toggle', {'id': self.switch_id})
                    kind = 'poll'
                elif kind == 'led':
                    target = self.led_target
                    if target is not None and target != self.led_applied:
                        self.set_led_brightness(target)
                        self.led_applied = target
                    kind = None
                elif kind == 'rpc':
                    _, method, params = command
                    self.events.put(('reply', self.rpc(method, params)))
                    kind = 'poll'
                if kind == 'poll' and self.host:
                    status = self.rpc('Switch.GetStatus', {'id': self.switch_id})
                    self.events.put(('status', status))
                    if self.led_read_needed:
                        self.led_read_needed = False
                        brightness = self.read_led_brightness()
                        if brightness is not None:
                            self.led_applied = brightness
                            self.events.put(('led', brightness))
            except Exception as e:                   # noqa: BLE001 - surfaced in the node
                message = str(e) if isinstance(e, ShellyRPCError) else type(e).__name__
                self.events.put(('error', message))
            if kind == 'poll':
                next_poll = time.monotonic() + self.poll_interval
        self.session.close()

    def set_led_brightness(self, brightness):
        """Brightness for both ring modes, so it holds whichever mode the plug
        is in ('power' shows the load; 'switch' shows on/off colours)."""
        key = f'switch:{self.switch_id}'
        colors = {'power': {'brightness': brightness},
                  key: {'on': {'brightness': brightness}, 'off': {'brightness': brightness}}}
        self.rpc('PLUGS_UI.SetConfig', {'config': {'leds': {'colors': colors}}})

    def read_led_brightness(self):
        """The plug's current ring brightness, or None (not a plug, or no LEDs)."""
        try:
            leds = self.rpc('PLUGS_UI.GetConfig')['leds']
            colors = leds['colors']
            if leds.get('mode') == 'switch':
                return int(round(colors[f'switch:{self.switch_id}']['on']['brightness']))
            return int(round(colors['power']['brightness']))
        except Exception:                            # noqa: BLE001 - not every Shelly has a ring
            return None

    # -- device list ----------------------------------------------------

    def start_scan(self, send_reply):
        # Its own thread: a scan listens for 5 s and must not hold up polling.
        if self.scanning:
            return
        self.scanning = True
        if send_reply:
            self.set_status('scanning...')

        def scan():
            try:
                self.events.put(('devices', (scan_shelly_devices(), send_reply)))
            except Exception as e:                   # noqa: BLE001
                self.events.put(('error', f'scan: {type(e).__name__}'))
            finally:
                self.scanning = False
        threading.Thread(target=scan, daemon=True).start()

    def device_chosen(self):
        # The saved 'host' option is the source of truth on load.
        if self.in_loading_process:
            return
        choice = any_to_string(self.device_property()).strip()
        if choice:
            self.retarget(choice.split()[0])

    def retarget(self, host):
        self.host_option.set(host)
        self.host_changed()

    def matches_host(self, device):
        return self.host in (device[1], normalise_host(device[0]))

    def show_host_in_list(self):
        """Fill the device list, keeping the current host in it (and selected)
        even when the scan did not hear it. Main thread only."""
        labels = [device_label(d) for d in self.devices]
        current = ''
        for device, label in zip(self.devices, labels):
            if self.matches_host(device):
                current = label
        if not current and self.host:
            current = self.host
            labels.insert(0, current)
        widget = self.device_property.widget
        widget.combo_items = labels
        dpg.configure_item(widget.uuid, items=labels)
        self.device_property.set(current)

    # -- main thread ----------------------------------------------------

    def set_status(self, text):
        if text != self.shown_status:
            self.shown_status = text
            dpg.set_value(self.status_label.widget.uuid, text)

    def frame_task(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind, payload = event
            if kind == 'status':
                self.apply_status(payload)
            elif kind == 'led':
                self.led_input.set(payload, propagate=False)
            elif kind == 'reply':
                self.reply_output.send(payload)
            elif kind == 'devices':
                devices, send_reply = payload
                self.devices = devices
                if not self.host and devices:
                    self.retarget(devices[0][1] or devices[0][0])
                self.show_host_in_list()
                if send_reply:
                    self.set_status(f'found {len(devices)} device(s)')
                    self.reply_output.send(devices)
            elif kind == 'error':
                self.set_status(payload)

    def apply_status(self, status):
        if not isinstance(status, dict):
            return
        on = bool(status.get('output'))
        power = status.get('apower')
        parts = ['ON' if on else 'off']
        if power is not None:
            parts.append(f'{power:.1f} W')
        if status.get('voltage') is not None:
            parts.append(f"{status['voltage']:.0f} V")
        if status.get('errors'):
            parts.append(' '.join(status['errors']))
        self.set_status('  '.join(parts))

        self.on_input.set(on, propagate=False)
        self.status_output.send(status)
        if power is not None:
            self.power_output.send(float(power))
        if on != self.last_output:
            self.last_output = on
            self.state_output.send(1 if on else 0)
