"""
A minimal KVM switcher window for the DM-NVX-360 setup.

Run a copy on each target computer. Because your keyboard and mouse follow the
switch, whichever machine you are currently driving is the one whose window is
under your cursor — so you can always switch away from where you are.

Standard library only (tkinter), plus nvx_kvm.py beside it. To deploy, copy
nvx_kvm.py and this file to the target machine along with ~/.nvx_kvm.json:

    python3 nvx_switcher.py

Targets come from the "hosts" map in ~/.nvx_kvm.json; anything that reports
itself as a Transmitter becomes a button. The receiver is the "receiver" entry,
or the one host that reports itself as a Receiver.
"""

import queue
import sys
import threading
import tkinter as tk
from tkinter import font as tkfont

try:                                    # installed as part of dpg_system
    from dpg_system.nvx_kvm import (NvxDevice, NvxError, load_config, load_targets,
                                    read_state, resolve_host, switch_receiver)
except ImportError:                     # or just dropped in a folder next to it
    from nvx_kvm import (NvxDevice, NvxError, load_config, load_targets,
                         read_state, resolve_host, switch_receiver)

POLL_SECONDS = 5


def switch_to(receiver_host, target_host, targets_by_mac):
    switch_receiver(receiver_host, target_host, targets_by_mac)
    return f'switched to {target_host}'


class Switcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('KVM')
        self.resizable(False, False)
        self.results = queue.Queue()
        self.buttons = {}
        self.targets = {}          # label -> host
        self.targets_by_mac = {}   # mac -> host
        self.receiver = None
        self.active_host = None
        self.busy = False

        big = tkfont.Font(size=15)
        self.frame = tk.Frame(self, padx=14, pady=12)
        self.frame.pack(fill='both', expand=True)
        self.title_label = tk.Label(self.frame, text='finding endpoints...', font=big)
        self.title_label.pack(anchor='w', pady=(0, 8))
        self.button_box = tk.Frame(self.frame)
        self.button_box.pack(fill='x')
        self.status = tk.Label(self.frame, text='', anchor='w', fg='#666')
        self.status.pack(fill='x', pady=(10, 0))

        self.after(100, self.drain)
        self.run_async(self.discover)

    # -- threading ------------------------------------------------------
    # Every device call is slow enough to freeze the window, so all of them
    # run on a worker thread and post their result back through a queue.

    def run_async(self, fn, *args):
        def work():
            try:
                self.results.put(('ok', fn(*args)))
            except Exception as e:                      # noqa: BLE001 - shown to the user
                self.results.put(('error', f'{type(e).__name__}: {e}'))
        threading.Thread(target=work, daemon=True).start()

    def drain(self):
        try:
            while True:
                kind, payload = self.results.get_nowait()
                if kind == 'error':
                    self.busy = False
                    self.status.config(text=payload, fg='#a00')
                    self.refresh_buttons()
                elif isinstance(payload, dict):         # a state report
                    self.apply_state(payload)
                else:                                   # a status message
                    self.busy = False
                    self.status.config(text=payload or '', fg='#666')
                    self.run_async(self.poll)
        except queue.Empty:
            pass
        self.after(100, self.drain)

    # -- device work ----------------------------------------------------

    def discover(self):
        """Targets and their MACs from the config; the MAC map lets a switch
        release the endpoint it is leaving."""
        config = load_config()
        targets = load_targets(config)
        if not targets:
            raise NvxError('no targets in ~/.nvx_kvm.json')
        receiver = config.get('receiver')
        if not receiver:
            raise NvxError('no "receiver" in ~/.nvx_kvm.json')
        receiver = resolve_host(receiver, config)

        by_mac = {}
        for target in targets:
            device = NvxDevice(target['host'], timeout=8)
            try:
                device.login()
                by_mac[read_state(device)['mac']] = target['host']
            except Exception:                            # unreachable, skip it
                continue
            finally:
                device.close()
        return {'discovered': True, 'receiver': receiver,
                'targets': {t['name']: t['host'] for t in targets},
                'by_mac': by_mac}

    def poll(self):
        device = NvxDevice(self.receiver, timeout=10)
        device.login()
        try:
            state = read_state(device)
        finally:
            device.close()
        location = state['stream_location'] or ''
        host = location.split('//')[-1].split(':')[0] if location else None
        return {'active': host, 'status': state['stream_status']}

    # -- ui -------------------------------------------------------------

    def apply_state(self, payload):
        if payload.get('discovered'):
            self.receiver = payload['receiver']
            self.targets = payload['targets']
            self.targets_by_mac = payload['by_mac']
            self.title_label.config(text=f'receiver {self.receiver}')
            for label, host in sorted(self.targets.items()):
                b = tk.Button(self.button_box, text=label, width=22, pady=6,
                              command=lambda h=host: self.on_click(h))
                b.pack(fill='x', pady=3)
                self.buttons[host] = b
            self.run_async(self.poll)
            return
        self.active_host = payload.get('active')
        if not self.busy:
            self.status.config(text=payload.get('status') or '', fg='#666')
        self.refresh_buttons()
        self.after(POLL_SECONDS * 1000, lambda: self.run_async(self.poll))

    def refresh_buttons(self):
        for host, button in self.buttons.items():
            label = next(k for k, v in self.targets.items() if v == host)
            active = (host == self.active_host)
            button.config(text=('● ' if active else '   ') + label,
                          state='disabled' if (active or self.busy) else 'normal')

    def on_click(self, host):
        if self.busy:
            return
        self.busy = True
        self.status.config(text=f'switching to {host}...', fg='#666')
        self.refresh_buttons()
        self.run_async(switch_to, self.receiver, host, self.targets_by_mac)


if __name__ == '__main__':
    try:
        Switcher().mainloop()
    except NvxError as e:
        print(f'error: {e}', file=sys.stderr)
        sys.exit(1)
