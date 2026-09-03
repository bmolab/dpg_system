"""
dpg_system node for switching a Crestron DM-NVX-360 KVM between targets.

The node's shape comes from a JSON config, so the number of target buttons is
whatever the file declares:

    {
      "user": "admin",
      "password": "...",
      "receiver": "10.1.1.151",
      "targets": [
        {"name": "mac studio", "host": "10.1.1.153", "multicast": "239.1.0.0"},
        {"name": "linux",      "host": "10.1.1.154", "multicast": "239.1.0.2"}
      ]
    }

Default path is ~/.nvx_kvm.json; pass another as the node's argument:

    nvx_kvm ~/patches/studio_kvm.json

Buttons are created in __init__ from the file, so changing the target list means
re-creating the node. Everything else (which target is live, reachability) is
polled at run time.

All device calls are network-bound and take a second or more, so they run on a
worker thread; results are applied in frame_task() on the main dpg thread.
"""

import threading
import traceback

import dearpygui.dearpygui as dpg

from dpg_system.node import Node
from dpg_system.conversion_utils import *
from dpg_system.nvx_kvm import (NvxError, load_config, load_targets, read_state,
                                resolve_host, switch_receiver, NvxDevice)

POLL_INTERVAL_FRAMES = 300      # roughly every 5 s at 60 fps


def register_nvx_nodes():
    Node.app.register_node('nvx_kvm', NVXKVMNode.factory)


class NVXKVMNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = NVXKVMNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.config_path = None
        if args is not None and len(args) > 0:
            self.config_path = any_to_string(args[0])

        self.targets = []
        self.receiver = None
        self.load_error = ''
        try:
            config = load_config(self.config_path)
            self.targets = load_targets(config)
            receiver = config.get('receiver')
            self.receiver = resolve_host(receiver, config) if receiver else None
            if not self.receiver:
                self.load_error = 'no "receiver" in config'
            elif not self.targets:
                self.load_error = 'no targets in config'
        except NvxError as e:
            self.load_error = str(e)

        # Cross-thread state. The worker only writes these plain attributes;
        # frame_task reads them and touches the widgets.
        self.lock = threading.Lock()
        self.busy = False
        self.pending_status = None       # str to show
        self.pending_active = None       # host that is now live
        self.active_host = None
        self.shown_status = None
        self.shown_active = None
        self.frames_to_poll = 1
        self.hosts_by_mac = {}

        # One button per target, straight from the config.
        self.target_buttons = []
        for target in self.targets:
            button = self.add_input(target['name'], widget_type='button',
                                    callback=self.make_switch_callback(target['host']))
            self.target_buttons.append((target, button))

        # Also switchable from a patch: a name, or a 1-based index.
        self.select_input = self.add_input('select', triggers_execution=True)

        self.status_property = self.add_label('')
        self.active_output = self.add_output('active target')

    # -- creation -------------------------------------------------------

    def custom_create(self, from_file):
        # Widget values are only real once the widgets exist, so anything that
        # reads them belongs here rather than in __init__.
        self.set_status(self.load_error or 'idle')
        if self.receiver:
            self.add_frame_task()

    def cleanup(self):
        if hasattr(self.app, 'remove_frame_task'):
            self.app.remove_frame_task(self)
        super().cleanup()

    # -- ui helpers -----------------------------------------------------

    def set_status(self, text):
        """Main-thread only."""
        if text != self.shown_status:
            self.shown_status = text
            dpg.set_value(self.status_property.widget.uuid, text)

    def refresh_button_labels(self):
        """Mark the live target. Main-thread only."""
        if self.active_host == self.shown_active:
            return
        self.shown_active = self.active_host
        for target, button in self.target_buttons:
            live = (target['host'] == self.active_host)
            dpg.set_item_label(button.widget.uuid,
                               ('* ' if live else '') + target['name'])

    # -- switching ------------------------------------------------------

    def make_switch_callback(self, host):
        def callback(input=None):
            # Loading a patch replays widget callbacks; without this guard the
            # node would fire a real switch every time the patch opens.
            if self.in_loading_process:
                return
            self.switch_to(host)
        return callback

    def switch_to(self, host):
        with self.lock:
            if self.busy:
                return
            self.busy = True
        self.set_status(f'switching to {host}...')
        threading.Thread(target=self.switch_worker, args=(host,), daemon=True).start()

    def switch_worker(self, host):
        try:
            switch_receiver(self.receiver, host, self.hosts_by_mac)
            with self.lock:
                self.pending_status = f'on {self.name_for(host)}'
                self.pending_active = host
        except Exception as e:                       # noqa: BLE001 - surfaced in the node
            with self.lock:
                self.pending_status = f'{type(e).__name__}: {e}'
            if self.app.verbose:
                traceback.print_exc()
        finally:
            with self.lock:
                self.busy = False

    def name_for(self, host):
        for target in self.targets:
            if target['host'] == host:
                return target['name']
        return host

    def execute(self):
        """'select' input: a target name, or a 1-based index."""
        if not self.targets:
            return
        value = self.select_input()
        if value is None:
            return
        host = None
        if is_number(value):
            index = any_to_int(value) - 1
            if 0 <= index < len(self.targets):
                host = self.targets[index]['host']
        else:
            wanted = any_to_string(value).strip().lower()
            for target in self.targets:
                if wanted in (target['name'].lower(), target['host']):
                    host = target['host']
                    break
        if host is None:
            self.set_status(f'no target matching {value!r}')
            return
        self.switch_to(host)

    # -- polling --------------------------------------------------------

    def frame_task(self):
        with self.lock:
            status, active = self.pending_status, self.pending_active
            self.pending_status = self.pending_active = None
            busy = self.busy
        if status is not None:
            self.set_status(status)
        if active is not None:
            self.active_host = active
            self.refresh_button_labels()
            self.active_output.send(self.name_for(active))

        self.frames_to_poll -= 1
        if self.frames_to_poll <= 0:
            self.frames_to_poll = POLL_INTERVAL_FRAMES
            if not busy:
                threading.Thread(target=self.poll_worker, daemon=True).start()

    def ensure_host_map(self):
        """MAC -> host for the targets, so a switch can release the endpoint it
        is leaving. Built once; a stale pairing is left behind without it."""
        if self.hosts_by_mac:
            return
        mapping = {}
        for target in self.targets:
            try:
                device = NvxDevice(target['host'], timeout=8)
                device.login()
                try:
                    mapping[read_state(device)['mac']] = target['host']
                finally:
                    device.close()
            except Exception:                        # noqa: BLE001 - skip unreachable
                continue
        self.hosts_by_mac = mapping

    def poll_worker(self):
        """Read which target the receiver is actually on."""
        try:
            self.ensure_host_map()
            device = NvxDevice(self.receiver, timeout=10)
            device.login()
            try:
                state = read_state(device)
            finally:
                device.close()
            location = state['stream_location'] or ''
            host = location.split('//')[-1].split(':')[0] if location else None
            with self.lock:
                if host and host != self.active_host:
                    self.pending_active = host
                if not self.busy:
                    self.pending_status = (f'on {self.name_for(host)}' if host
                                           else 'no stream')
        except Exception as e:                       # noqa: BLE001
            with self.lock:
                if not self.busy:
                    self.pending_status = f'{type(e).__name__}'
