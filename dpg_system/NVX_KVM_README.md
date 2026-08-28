# KVM over IP with DM-NVX-360, no control processor

`nvx_kvm.py` drives Crestron DM-NVX-360 endpoints directly over their CresNext
REST API so one workstation can switch between several target computers. It is
standalone — `requests` is the only dependency and it imports nothing from
dpg_system — so it can be driven from a shell, a hotkey daemon, or wrapped in
dpg nodes later.

Measured against real hardware, 2026-08-13 and 2026-08-17. Where something is
inferred rather than observed, it says so.

## Status

**Working end to end, across two targets.** `switch` moves the workstation
between the Mac Studio and the Linux box — video and mouse both follow, verified
visually at the monitor in both directions.

Measured switch time: **0.8s, 0.9s, 3.5s, 7.7s** over four switches. Usually
about a second; the long tail is the receiver taking its time to report fully
started and paired, not the picture arriving.

All four units are on firmware `7.1.5259.00090`. The NVX switch now has IGMP
snooping enabled, which is a hard prerequisite (see below).

## Topology

Each target computer gets a transmitter: computer HDMI out → NVX HDMI in, and
computer USB → NVX. The workstation gets a receiver: monitor on HDMI out,
keyboard and mouse on USB. Video subscription and USB pairing route
independently, so a switch means re-pointing both.

Current assignment:

```
10.1.1.151   Receiver     workstation (monitor + mouse)     -
10.1.1.152   Transmitter  spare, no source attached         -
10.1.1.153   Transmitter  Mac Studio                        multicast 239.1.0.0
10.1.1.154   Transmitter  Linux workstation                 multicast 239.1.0.2
```

## Config

`~/.nvx_kvm.json`, mode 600. The `targets` list defines how many targets exist
and their settings — it is what the dpg node and the switcher window build
their buttons from, and its order is the button order:

```json
{
  "user": "admin",
  "password": "...",
  "receiver": "10.1.1.151",
  "targets": [
    {"name": "mac studio", "host": "10.1.1.153", "multicast": "239.1.0.0"},
    {"name": "linux",      "host": "10.1.1.154", "multicast": "239.1.0.2"}
  ],
  "hosts": {"mac studio": "10.1.1.153", "linux": "10.1.1.154"}
}
```

`hosts` is a plain name map used by the CLI, so `switch "mac studio"` works. If
`targets` is absent the target list falls back to `hosts` minus the receiver,
which keeps older configs working.

Adding a target is one entry in `targets` — no code change anywhere.

## The recipe that works

Both ends must agree on **all** of these or the receiver sits at `Connecting`
forever with no error reported anywhere:

```
SessionInitiation : Multicast via RTSP
TransportMode     : MPEG2TSRTP
TsPort            : 4570
```

The transmitter additionally needs its **own** multicast group. Two
transmitters sharing an address will collide.

**The final octet of a multicast address must be EVEN.** The device takes
consecutive groups for video and audio, so an odd base is rejected — with the
same featureless "generic error" it returns for everything else, which is why
this took so long to spot. `239.1.0.0` and `239.1.0.2` are fine; `239.1.0.1` is
refused. `start_transmit()` now checks this up front and says so plainly.

Bringing a transmitter up, in order — all three steps are load-bearing:

1. `IsAutomaticInitiationEnabled = False`. While it is true, `Start` writes are
   accepted and silently ignored. The web UI greys out its own Start button for
   the same reason.
2. `Stop = True`, wait a few seconds, then set `TransportMode`,
   `SessionInitiation`, and `MulticastAddress`. **`MulticastAddress` is only
   writable while the stream is stopped** — on a running stream it is refused
   with a generic error that reads exactly like a read-only property.
3. `Stop = False`, `Start = False`, `Start = True`. These are edge-triggered
   momentary flags, not levels; re-posting `Start = True` when it already reads
   `True` does nothing at all.

Then the receiver is a single write of `StreamReceive.Streams[0].StreamLocation`
to `rtsp://<tx-ip>:554/live.sdp`. That URL is predictable, so a receiver can be
pointed at a transmitter before the transmitter has published it.

`start_transmit()` and `align_receiver()` in the module implement all of this.

## Reading the device honestly

**HTTP 200 does not mean success.** A refused write still returns 200; the real
outcome is a per-property `StatusId` inside `Actions[].Results[]`. `0` is OK,
`1` means "accepted, needs a reboot", and negatives are refusals — `-1` value
out of range, `-4` generic, which covers both read-only properties and unmet
preconditions. `NvxDevice.post()` parses this and raises on negatives; positives
surface separately as `_pending`. Ignoring this makes every failed write look
like it worked, which cost hours.

**`NumVideoPacketsRcvd` on the receiver never increments**, even with
`IsStatisticsEnabled` true and video demonstrably flowing. It is a false
negative and must not be used as a success signal. Use instead:

- the receiver's `Bitrate` field, which reports a real measured rate (~716 for
  a 4K60 source)
- the receiver's `HorizontalResolution`/`VerticalResolution`, which only
  populate once it is genuinely decoding
- the transmitter's `NumVideoPacketsTransmitted`, which does increment

**Port 554 is the reliable liveness check.** A transmitter's RTSP server only
listens while its stream is started, so a plain TCP connect to 554 answers
"is this transmitter actually up" faster and more honestly than the API's
`Status` field, which lags. A `412 Encryption failed` reply to a plain RTSP
probe is normal — it means the server is alive and wants Crestron's encrypted
handshake.

**Role changes need a reboot.** `DeviceSpecific.DeviceMode` returns
`StatusId 1`. Budget 2.5–4 minutes; a mode change takes longer than a plain
reboot because it reconfigures the AV pipeline.

## Switching

The per-switch action is a single POST of the receiver's `StreamLocation`.
Transmitters stay up; you do not start them as part of a switch. Measured at
about 2 seconds to `Stream started`.

USB pairing is written explicitly on **both** endpoints —
`Usb.UsbPorts[0].UsbPairing.Layer2.RemoteDevices.Id1`, holding the peer's MAC.
`IsMultipleDeviceSupportEnabled` is false, so it is one partner at a time and a
switch must clear the old pairing rather than just adding a new one. Both ends
report paired within about 5 seconds.

`AvRouting.RouteControl.IsUsbFollowsVideoEnabled` looks like it should collapse
this to one write. It does not — set true on both ends, USB pairing did **not**
follow a video route change. Pairing is written explicitly.

## USB wiring

The unit has three USB jacks. The naming means what it says, read from the
NVX's point of view:

```
DEVICE  USB-C    TRANSMITTER end: the computer connects here. The NVX presents
                 itself as a USB device to that computer.
HOST    Type-A   RECEIVER end: keyboard and mouse connect here. The NVX acts as
                 their USB host.
HID     Type-A   intended for keyboard/mouse with NVX-side HID parsing;
                 untested here, see the hotkey section.
```

Proven working: Mac Studio → **DEVICE** on 153 with a **USB-A to USB-C** cable,
keyboard and mouse → **HOST** on 151.

A caution that cost a lot of time: the first attempt used this same DEVICE jack
with a USB-C-to-USB-C cable and never enumerated — no Crestron device appeared
in the Mac's System Report at all. The jack was right; the cable was almost
certainly charge-only. **If USB does not enumerate, suspect the cable before
the configuration.** `HostEnumerated` staying false on the computer's end is
the signal, and it is upstream of anything settable in the API.

The **HID** port is a separate path where the NVX parses HID reports itself,
which is where hotkey capture would have to live. We never got a keyboard
working on it — but every HID test was confounded (see the hotkey section), so
it is untested rather than broken.

Note that `PairedStatus` reflects stored configuration, not a live link — it
reports `true` on both ends while nothing is physically connected. It is not
evidence that USB works. `HostEnumerated` going true on the computer's end is
the real signal.

Also: USB enumeration only happens when a device is presented to a host, so any
test of USB `Mode` or wiring requires physically re-plugging the cable
afterwards. A round of mode tests without re-plugging proves nothing.

## When everything looks unreachable, check your own machine first

**A connected VPN makes the entire rig look dead.** It captures the route to
the local subnet, so every endpoint fails to answer while the devices are
streaming perfectly — video keeps appearing on the monitor and only switching
"stops working", because switching is the part that needs the API.

This happened, and it was misdiagnosed three times in a row — as a multicast
flood, then a dead switch, then a VLAN change — because "all endpoints
unreachable" matches a real flood we had seen before.

`nvx_kvm net` checks it in one line, and any unreachable-endpoint error now
carries the hint automatically:

```
10.1.1.153: unreachable (TimeoutError)
  hint: the route to 10.1.1.153 leaves via 'utun4', which is a VPN or tunnel
        interface. A connected VPN can capture traffic to the local subnet.
```

The discriminator that would have caught it immediately: **a real flood
recovers within seconds of physically isolating the source.** If unplugging the
transmitters changes nothing, and units that were never touched are also
unreachable, the problem is not traffic — look at routing, the switch, and your
own machine before theorising about the devices.

## Multicast requires IGMP snooping — non-negotiable

A 4K60 stream is roughly 700 Mbps. Before the NVX switch had IGMP snooping,
starting a real multicast stream flooded every port and knocked **all four
units** off the network — the rest of the network was unaffected — and it only
recovered when the transmitter was physically unplugged.

This is now fixed (snooping enabled) and multicast works correctly. But it is
the load-bearing prerequisite: if this setup is ever moved to another switch,
check snooping and a querier **before** starting a transmitter with a real
source attached.

Guard rail worth keeping: when testing anything that touches streaming, ping
two uninvolved units once a second and stop the transmitter automatically if
they drop. That turns a network outage into a three-second blip instead of a
walk to the rack.

## The switcher window

`nvx_switcher.py` is a small tkinter panel: one button per transmitter, the live
one marked and disabled, receiver state polled every few seconds. Standard
library only.

Run a copy **on each target computer**. Your keyboard and mouse follow the
switch, so whichever machine you are currently driving is the one whose window
is under your cursor — that is what makes it always reachable. A single copy on
one target would strand you the moment you switched away from it.

To deploy, copy `nvx_kvm.py`, `nvx_switcher.py`, and `~/.nvx_kvm.json` (mode
600) to the machine, then `python3 nvx_switcher.py`. It imports `nvx_kvm`
either from dpg_system or from the same folder, so a bare directory works.
Linux may need `python3-tk`; macOS ships it.

Roles are rediscovered at launch, so converting another unit to a transmitter
makes it appear as a button with no code change. Note this puts the NVX
credentials in plaintext on every target that runs it — consider a dedicated
account rather than admin if those machines are shared.

Every device call runs on a worker thread and reports back through a queue;
doing them inline freezes the window for the second or more a switch takes.

## The dpg_system node

`nvx_nodes.py` registers **`nvx_kvm`**. It builds one button per entry in the
config's `targets` list, marks the live one with `*`, and exposes:

- a `select` input taking a target name or a 1-based index, so a patch can
  drive switching
- an `active target` output, sent whenever the live target changes
- a status line showing progress and errors

Default config is `~/.nvx_kvm.json`; pass another path as the node argument:

```
nvx_kvm ~/patches/studio_kvm.json
```

Enabled via `"nvx_nodes": true` in `dpg_system_config.json`, and listed in
`optional_import` in `dpg_app.py`.

Two implementation notes, both learned the hard way in this codebase:

Buttons are created in `__init__` from the config, because node structure is
fixed at creation — so **changing the target list means re-creating the node**.
Everything else (which target is live, reachability) is polled at run time.

Every device call is network-bound and takes a second or more, so switching and
polling run on worker threads that only touch plain attributes; `frame_task()`
applies results to widgets on the main dpg thread. The button callbacks also
check `in_loading_process`, or loading a patch would replay them and fire a real
switch on open.

## Native hotkey switching — investigated, not working

The receiver has seven USB pairing slots (`Id1`–`Id7`) and a
`Usb.HotkeyConfig` with a capture-entry key, which together look exactly like
hardware KVM hotkey switching. It could not be made to work. What was learned:

- `CaptureEntryModeKey` accepts only `ScrollLock` and `CustomKeyStroke` out of
  ~30 candidates. With `CustomKeyStroke`, `CaptureEntryModeCustomKey` takes an
  8-byte USB HID report (modifier byte, reserved, then six keycodes) — e.g.
  `05 00 0e 00 00 00 00 00` is Ctrl+Alt+K, `00 00 68 00 00 00 00 00` is F13.
  So the key itself is fully programmable.
- **Enabling `IsMultipleDeviceSupportEnabled` breaks USB.** It clears
  `PairedStatus` to all-false and pairing does not re-establish; keyboard and
  mouse both go dead. Reproduced twice. Since multi-device support is the
  prerequisite for having more than one slot to select between, this blocks the
  feature at its foundation.
- There is no capture-state field anywhere in the tree, and no "select active
  slot" property. The device offers no way to observe whether it is in capture
  mode; a dead keyboard is the only signal, and that is ambiguous.

**A large confounder wasted much of this investigation**: the Mac Extended
keyboard can connect by Bluetooth as well as by cable, and was on Bluetooth for
an unknown portion of the testing. Its keystrokes were going straight to macOS
without ever traversing the NVX, which perfectly mimics "the hotkey is not being
trapped". Before any future hotkey test, prove the keyboard actually routes
through the NVX by switching to a *different* target and confirming the typing
lands there. That test is cheap and unambiguous.

## Commands

```
python -m dpg_system.nvx_kvm net                     # check local routing FIRST
python -m dpg_system.nvx_kvm probe [hosts...]        # log in, list objects
python -m dpg_system.nvx_kvm status [hosts...]       # role/route state, one line each
python -m dpg_system.nvx_kvm dump <host> -o DIR      # whole object tree to JSON
python -m dpg_system.nvx_kvm get <host> <path>       # one CresNext object
python -m dpg_system.nvx_kvm transmit <host> [--multicast 239.1.0.1] [--stop]
python -m dpg_system.nvx_kvm switch <target> [--receiver H] [--dry-run] [--video-only]
python -m dpg_system.nvx_switcher                    # the switcher window
```

## Adding another target

1. Put the computer's HDMI into the NVX's HDMI in, and its USB into the
   **DEVICE** (USB-C) jack — with a cable you know carries data.
2. Set `DeviceSpecific.DeviceMode` to `Transmitter` and USB `Mode` to `Local`,
   then reboot (2.5–4 minutes).
3. Give it an unused **even** multicast group and start it:
   `python -m dpg_system.nvx_kvm transmit <host> --multicast 239.1.0.4`
4. Switch to it: `python -m dpg_system.nvx_kvm switch <host>`

Step 3 may need the web UI. See below.

## Still open

- **`MulticastAddress` writes are unreliable over the API.** With an even
  address and both auto-initiation flags off and the stream stopped, 154 still
  refused every value; it had to be set from the web UI. The same write
  succeeded once on 153. Some precondition remains unidentified, so adding a
  target may need one manual step in the web UI.
- `Edid.CurrentEdid` writes are refused with "value out of range" — it wants
  something other than the display name from `EdidList`. Never shown to matter;
  the 4K60 4:4:4 HDCP source works fine as-is.
- Nothing in the module warns about the multicast/IGMP prerequisite.
- `--video-only` and the `--usb-wait` auto-follow path have not been exercised
  since the working recipe was found.

## Two wrong turns worth remembering

**The firmware misdiagnosis.** Early on, 151 and 153 (then on 2022 firmware)
both refused to transmit while 152 (2023 firmware) streamed happily, and a
controlled experiment — converting the unconnected 151 to a transmitter —
reproduced the failure exactly. That looked conclusive and it was wrong. 152 was
simply found *already streaming*, started by some earlier means; the real cause
was the auto-initiation gate plus the multicast-address precondition, which
affect both firmware versions equally. The one unit that worked had never been
started by us, so it was not evidence about starting at all.

**The `ByReceiver` dead end.** After multicast took the network down,
`SessionInitiation: ByReceiver` looked attractive: it is unicast, so it cannot
flood, and it was the only mode that let a transmitter with a real source
*start*. A long stretch went into it — matching transports, ports, disabling
password protection. It never carried a single packet in any combination tried.
The real fix was IGMP snooping on the switch, after which the original
multicast recipe simply worked.
