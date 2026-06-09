# DiGiCo S21 OSCQuery Proxy

Makes a DiGiCo S21/S31 (S-Series) console discoverable and controllable
through OSCQuery, so dpg_system's `oscq_browse` can find it, search its tree,
and instantiate a complete UI for any branch.

## How discovery works (different from the Eos proxy)

The S21 speaks OSC over UDP but has **no query API** to enumerate its
capabilities. Instead, the proxy **learns by listening**: it registers as the
console's OSC controller, and every message the console sends — a live fader
move, or all of them at once via the console's **"Resend All"** button —
teaches the proxy that address, its argument types, observed range, and
current value. The OSCQuery tree mirrors the console's actual namespace.
String parameters named `name`/`label` become the DESCRIPTION of their parent
node, so channels appear with their console labels.

The learned namespace is cached (`digico_proxy_cache.json`), so "Resend All"
is a one-time teaching step per console setup; the proxy keeps refining types
and values from live traffic forever after.

You can also pre-load the namespace from the console's exported OSC command
list (OSC Commands view > File > save to USB) with `--import-commands FILE`;
imported entries start as generic floats and are refined when real values
arrive.

## Files

- `digico_oscquery_proxy.py` — the whole proxy
- `oscquery_service.py` — reused from dpg_system (HTTP + mDNS server half)

Copy **both files into the same directory** to deploy on another machine.

```
pip install python-osc zeroconf
```

## Console setup (S-Series: Extensions > OSC Control)

1. Enable OSC; set the **console receive port** (proxy's `--console-port`,
   default 8001).
2. **Add a controller**: IP = the proxy machine, **send port** = proxy's
   `--listen-port` (default 8002). Enable both send and receive for it.
3. In **OSC Commands view**, press **"Resend All"** — the proxy prints each
   command as it learns it.

> **Important:** the S-Series allows only **one active OSC controller at a
> time**. The proxy must be that controller — all dpg_system machines then
> talk to the console *through the proxy*, which is the point.

## Running

```
python digico_oscquery_proxy.py --console-ip <console-ip>
python digico_oscquery_proxy.py --console-ip <console-ip> --import-commands "OSC Commands.txt"
python digico_oscquery_proxy.py --mock        # fake console for lab testing
```

Options: `--name` (mDNS service name, default `digico`), `--osc-port`
(UDP port advertised to OSCQuery clients, default 8766), `--console-port`,
`--listen-port`, `--cache` (`""` to disable).

## What the tree looks like

The namespace is whatever the console transmits (learned verbatim), plus a
seeded snapshot branch that is known to work on the S-Series:

```
/digico/snapshots/fire            int — fire snapshot by number
/digico/snapshots/fire/next       button (proxy sends the int 0 the desk expects)
/digico/snapshots/fire/previous   button
<learned>/...                     faders (float sliders), mutes (toggles),
                                  names (strings → parent labels), EQ, dynamics,
                                  aux sends — everything "Resend All" reveals
```

Client OSC sent to the advertised UDP port is forwarded verbatim to the
console; console output updates tree VALUEs and is pushed to peers registered
via `GET http://<proxy>:<http-port>/subscribe?ip=<ip>&port=<udp-port>`.

In dpg_system the service appears as `digico` in `oscq_browse` (alias it to
`audio` via the service alias registry if desired).

## Channel numbering reference (S-Series)

All channels/buses share one OSC number space: Inputs 1–60, Aux 70–79,
Groups 80–93, Matrix 100–107, Control Groups 110–119, Master 120.
(Numbers persist even if a bus type is changed.)
