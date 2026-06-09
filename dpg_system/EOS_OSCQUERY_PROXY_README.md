# Eos OSCQuery Proxy

Makes an ETC Eos-family console (Gio @5) discoverable and controllable through
OSCQuery, so dpg_system's `oscq_browse` can find it, search its tree, and
instantiate a complete UI for any branch — exactly like any other OSCQuery
device on the network.

The proxy discovers the console's actual show contents itself: it connects to
the console over OSC-TCP and enumerates patched channels, cue lists, cues,
groups, submasters, macros, presets, palettes and snapshots via the Eos
"OSC Get" API, with their console labels. It re-enumerates automatically when
the show file is edited (`/eos/subscribe`).

## Files

- `eos_oscquery_proxy.py` — the whole proxy
- `oscquery_service.py` — reused from dpg_system (HTTP + mDNS server half)

To deploy on another machine, copy **both files into the same directory**.

## Dependencies

```
pip install python-osc zeroconf
```

(Any Python 3.9+. On the dpg machine these are already in the
`dpg_system_2025` env: `/Users/drokeby/miniforge3/envs/dpg_system_2025/bin/python3`.)

## Console setup (once, on the Gio @5)

Setup > System Settings > Show Control > OSC:

- **OSC RX**: enabled
- **OSC TX**: enabled
- **OSC TCP format**: note whether it is *OSC 1.0* (packet-length framing —
  the proxy's default) or *OSC 1.1* (SLIP — run the proxy with `--slip`)

Third-party clients connect to TCP port **3032** on the console; no
console-side IP configuration is needed for TCP.

## Running

```
# real console
python eos_oscquery_proxy.py --eos-ip <console-ip>

# console set to OSC 1.1 (SLIP) TCP framing
python eos_oscquery_proxy.py --eos-ip <console-ip> --slip

# no console needed — serves a fake show for testing dpg_system discovery
python eos_oscquery_proxy.py --mock
```

Options: `--name` (mDNS service name, default `eos`), `--osc-port` (UDP port
the proxy listens on for clients, default 8765), `--eos-port` (default 3032),
`--cache` (enumeration cache file so the tree is served instantly on restart;
`--cache ""` disables).

## What the tree looks like

```
/eos/chan/<n>            float 0–100, DESCRIPTION = patch label
/eos/group/<n>           float 0–100 (group intensity)
/eos/sub/<n>             float 0–1, plus /eos/sub/<n>/fire (bump toggle)
/eos/cue/<list>/<n>/fire button, DESCRIPTION = cue label
/eos/macro/<n>/fire      button, DESCRIPTION = macro label
/eos/preset|ip|fp|cp|bp|snap/<n>/fire   buttons
/eos/cmd, /eos/newcmd    command-line message nodes
/eos/key/<key>           curated console keys (go_0, stop, update, ...)
/eos/out/...             read-only feedback (command line, active/pending cue)
```

All addresses are real Eos OSC addresses — the proxy forwards client UDP
messages to the console verbatim over TCP, so anything not in the tree
(e.g. `/eos/user/...`) also works if sent through the same OSC target.

Console feedback (`/eos/out/...`) is pushed to any peer registered via the
HTTP subscription endpoint: `GET http://<proxy>:<http-port>/subscribe?ip=<ip>&port=<udp-port>`.

## Quick verification

With the proxy running (mock or real), from any machine:

```
curl "http://<proxy-ip>:9000/?HOST_INFO"     # service info (port printed at startup)
curl "http://<proxy-ip>:9000/eos/chan"       # the channel branch
```

In dpg_system, the service appears as `eos` in `oscq_browse`.