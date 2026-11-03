#!/usr/bin/env bash
# Force an Orbbec Femto to reconnect by disabling and re-enabling the USB hub
# port it is plugged into (works for direct connections and for the FireNEX
# optical extender alike).
#
# If the Femto is present, the port it is on is toggled. If not, every empty
# external port on the FireNEX remote hub is toggled, since we can't tell which
# one it is on.
#
# This does NOT recover the stuck state behind the FireNEX after a reboot or
# link loss (tested Oct 23 2026; the hub claims per-port power switching but
# the 5 V stays on, so the Femto never sees a real disconnect). For that,
# power-cycle at the remote end without touching the USB cable (tested
# Oct 23 2026): remote FireNEX power off and on, wait for the link, then
# cycle the Femto's power.
#
# Usage: ./femto_reset.sh [off_seconds]   (default 5; asks for sudo)
# Installed root-owned as /usr/local/sbin/femto_reset with a NOPASSWD sudoers
# rule so the femto node can call `sudo -n femto_reset`; re-run
#   sudo install -m 755 -o root femto_reset.sh /usr/local/sbin/femto_reset
# after editing.

set -euo pipefail

OFF_SECONDS=${1:-5}
FEMTO_VID="2bc5"        # Orbbec
HUB_ID="1d5c:5050"      # Fresco Logic hub inside the FireNEX remote unit
SYS=/sys/bus/usb/devices

usb_id() { echo "$(cat "$1/idVendor" 2>/dev/null):$(cat "$1/idProduct" 2>/dev/null)"; }

# sysfs 'disable' file of the hub port that device $1 (e.g. 2-3.4) hangs off.
port_disable_file() {
    local dev=$1
    if [[ $dev == *.* ]]; then
        local parent=${dev%.*}
        echo "$SYS/$parent:1.0/$parent-port${dev##*.}/disable"
    else
        # on a root hub port: 2-3 -> usb2/2-0:1.0/usb2-port3
        local bus=${dev%%-*}
        echo "$SYS/usb$bus/$bus-0:1.0/usb$bus-port${dev#*-}/disable"
    fi
}

files=()
for d in "$SYS"/*; do
    name=$(basename "$d")
    [[ $name == *:* ]] && continue                    # interfaces, not devices
    if [[ $(cat "$d/idVendor" 2>/dev/null) == "$FEMTO_VID" ]]; then
        files+=("$(port_disable_file "$name")")
        echo "Femto found at $name"
    fi
done

if (( ${#files[@]} == 0 )); then
    hub=""
    for d in "$SYS"/*; do
        [[ $(usb_id "$d") == "$HUB_ID" ]] && hub=$(basename "$d") && break
    done
    if [[ -z $hub ]]; then
        echo "No Femto, and no FireNEX remote hub ($HUB_ID) to search: check the" \
             "fiber link (LASER LED), or replug the Femto if it is connected directly." >&2
        exit 1
    fi
    # Port 5 is the FireNEX's own internal device; 1-4 are the external ports.
    for p in 1 2 3 4; do
        [[ $(cat "$SYS/$hub:1.0/$hub-port$p/state") == "not attached" ]] &&
            files+=("$SYS/$hub:1.0/$hub-port$p/disable")
    done
    echo "Femto not present; toggling ${#files[@]} empty port(s) on FireNEX hub $hub"
fi
(( ${#files[@]} > 0 )) || { echo "No ports to toggle." >&2; exit 1; }

# Get sudo up front so we never disable a port and then fail to re-enable it.
sudo -v || { echo "sudo failed; run this from a regular terminal." >&2; exit 1; }

for f in "${files[@]}"; do echo 1 | sudo tee "$f" >/dev/null; done
sleep "$OFF_SECONDS"
for f in "${files[@]}"; do echo 0 | sudo tee "$f" >/dev/null; done

for _ in $(seq 10); do
    sleep 1
    if line=$(lsusb -d "$FEMTO_VID:"); then
        echo "OK: $line"
        exit 0
    fi
done
echo "Femto did not come back. Power-cycle the remote FireNEX, wait for the link, then the Femto." >&2
exit 1
