"""
KVM-over-IP control for Crestron DM-NVX-360 endpoints, with no Crestron control
processor in the picture.

Deliberately standalone: `requests` is the only dependency and nothing here
imports dpg_system, so this can be driven from a shell script, a hotkey daemon,
or wrapped in dpg nodes later.

Authentication is the CresNext web API flow. Credentials are POSTed to
/userlogin.html, the session cookies are kept, and the CREST-XSRF-TOKEN handed
back in the login response is echoed on every subsequent request. Devices ship
with self-signed certificates, so verification is off by default.

Credentials come from, in order: explicit arguments, the NVX_USER / NVX_PASS
environment variables, or ~/.nvx_kvm.json:

    {
      "user": "admin",
      "password": "...",
      "hosts": {"tx-a": "10.1.1.151", "rx": "10.1.1.154"}
    }
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CONFIG_PATH = Path.home() / '.nvx_kvm.json'

# Response header the device hands back at login; we return it prefixed with X-.
XSRF_RESPONSE_HEADER = 'CREST-XSRF-TOKEN'
XSRF_REQUEST_HEADER = 'X-CREST-XSRF-TOKEN'


class NvxError(Exception):
    pass


def load_config(path=None):
    path = Path(path).expanduser() if path else CONFIG_PATH
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise NvxError(f'could not read {path}: {e}')
    return {}


def load_targets(config=None, path=None):
    """The switchable targets, in order, as [{name, host, multicast}, ...].

    Preferred form is an explicit "targets" list, which fixes the order and can
    carry per-target settings:

        {"receiver": "10.1.1.151",
         "targets": [{"name": "mac studio", "host": "10.1.1.153",
                      "multicast": "239.1.0.0"},
                     {"name": "linux", "host": "10.1.1.154",
                      "multicast": "239.1.0.2"}]}

    Falls back to the "hosts" name map, minus whatever the receiver is, so
    older configs keep working.
    """
    config = load_config(path) if config is None else config
    receiver = config.get('receiver')
    targets = []
    for entry in config.get('targets', []):
        host = entry.get('host') or entry.get('address')
        if not host:
            raise NvxError(f'target {entry!r} has no "host"')
        targets.append({'name': entry.get('name', host),
                        'host': host,
                        'multicast': entry.get('multicast')})
    if not targets:
        for name, host in config.get('hosts', {}).items():
            if receiver and host == resolve_host(receiver, config):
                continue
            targets.append({'name': name, 'host': host, 'multicast': None})
    return targets


def resolve_credentials(user=None, password=None, config=None):
    config = load_config() if config is None else config
    user = user or os.environ.get('NVX_USER') or config.get('user')
    password = password or os.environ.get('NVX_PASS') or config.get('password')
    if not user or not password:
        raise NvxError(
            'no credentials: pass --user/--password, set NVX_USER and NVX_PASS, '
            f'or create {CONFIG_PATH}')
    return user, password


def resolve_host(name, config=None):
    """Accept either a literal address or a name from the config's host map."""
    config = load_config() if config is None else config
    return config.get('hosts', {}).get(name, name)


# Interface name prefixes that mean "this is a tunnel, not the local wire".
TUNNEL_PREFIXES = ('utun', 'ipsec', 'ppp', 'tun', 'tap', 'gpd', 'cscotun', 'wg')


def route_interface(host):
    """Which local interface the route to `host` leaves by, or None."""
    try:
        if sys.platform == 'darwin':
            out = subprocess.run(['route', '-n', 'get', host],
                                 capture_output=True, text=True, timeout=4).stdout
            for line in out.splitlines():
                if 'interface:' in line:
                    return line.split(':', 1)[1].strip()
        else:
            out = subprocess.run(['ip', '-o', 'route', 'get', host],
                                 capture_output=True, text=True, timeout=4).stdout
            fields = out.split()
            if 'dev' in fields:
                return fields[fields.index('dev') + 1]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return None


def local_network_warning(host):
    """Explain a local-network problem that is not the device's fault.

    A running VPN is the one that actually bit us: it captured the route to the
    NVX subnet, so every endpoint looked dead while the devices were streaming
    perfectly. Worth saying out loud, because 'unreachable' otherwise reads as
    a device or switch failure and sends you debugging the wrong thing.
    """
    interface = route_interface(host)
    if interface and interface.startswith(TUNNEL_PREFIXES):
        return (f'the route to {host} leaves via {interface!r}, which is a VPN or '
                f'tunnel interface. A connected VPN can capture traffic to the '
                f'local subnet — disconnect it and try again.')
    return None


class NvxDevice:
    """One NVX endpoint's REST session."""

    def __init__(self, host, user=None, password=None, verify=False, timeout=10):
        self.host = host
        self.base = f'https://{host}'
        self.user = user
        self.password = password
        self.verify = verify
        self.timeout = timeout
        self.session = requests.Session()
        self.session.verify = verify
        self.authenticated = False

    # -- session ---------------------------------------------------------

    def login(self):
        """Establish a session. Returns a short string describing how."""
        # Some units have authentication disabled; if a plain read works there
        # is nothing to log into.
        if self._probe_unauthenticated():
            self.authenticated = True
            return 'no authentication required'

        user, password = resolve_credentials(self.user, self.password)

        try:
            # Prime the connection so the device issues its TRACKID cookie.
            self.session.get(f'{self.base}/userlogin.html', timeout=self.timeout)

            r = self.session.post(
                f'{self.base}/userlogin.html',
                data={'login': user, 'passwd': password},
                headers={
                    'Origin': self.base,
                    'Referer': f'{self.base}/userlogin.html',
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                allow_redirects=False,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise self.unreachable(e) from e
        if r.status_code not in (200, 302):
            raise NvxError(f'{self.host}: login returned HTTP {r.status_code}')

        token = r.headers.get(XSRF_RESPONSE_HEADER)
        if token:
            self.session.headers[XSRF_REQUEST_HEADER] = token
        elif 'login' in r.text.lower() and 'passwd' in r.text.lower():
            # Handed the login form back rather than a token: bad credentials.
            raise NvxError(f'{self.host}: login rejected (check credentials)')

        if not self._probe_unauthenticated():
            raise NvxError(
                f'{self.host}: logged in but device reads still fail '
                f'(status {r.status_code}, token {"yes" if token else "no"})')

        self.authenticated = True
        return f'session established{" with XSRF token" if token else ""}'

    def unreachable(self, exc):
        """Build the error for a failed connection, with a local-network hint.

        Worth the extra words: an unreachable endpoint reads as a dead device
        or a dead switch, and a VPN capturing the local subnet produces exactly
        that appearance while everything is actually fine.
        """
        message = f'{self.host}: unreachable ({type(exc).__name__})'
        hint = local_network_warning(self.host)
        if hint:
            message += f'\n  hint: {hint}'
        return NvxError(message)

    def _probe_unauthenticated(self):
        try:
            r = self.session.get(
                f'{self.base}/Device/DeviceInfo',
                headers={'Accept': 'application/json'},
                timeout=self.timeout,
                allow_redirects=False,
            )
        except requests.RequestException:
            return False
        if r.status_code != 200:
            return False
        try:
            r.json()
        except ValueError:
            return False
        return True

    # -- requests --------------------------------------------------------

    def get(self, path):
        """GET a CresNext object, e.g. 'Device/StreamReceive'."""
        path = path.strip('/')
        try:
            r = self.session.get(
                f'{self.base}/{path}',
                headers={'Accept': 'application/json'},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise self.unreachable(e) from e
        if r.status_code != 200:
            raise NvxError(f'{self.host}: GET {path} -> HTTP {r.status_code}')
        try:
            return r.json()
        except ValueError:
            raise NvxError(f'{self.host}: GET {path} returned non-JSON')

    def post(self, payload, path='Device', strict=True):
        """POST a CresNext patch, e.g. {'Device': {'StreamReceive': {...}}}.

        The device answers HTTP 200 even when it refuses the write; the real
        outcome is a per-property StatusId inside the body, where 0 means OK.
        A rejected property raises unless strict is False.
        """
        path = path.strip('/')
        try:
            r = self.session.post(
                f'{self.base}/{path}',
                json=payload,
                headers={'Accept': 'application/json',
                         'Origin': self.base,
                         'Referer': f'{self.base}/'},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise self.unreachable(e) from e
        if r.status_code not in (200, 202):
            raise NvxError(
                f'{self.host}: POST {path} -> HTTP {r.status_code}: {r.text[:200]}')
        try:
            body = r.json()
        except ValueError:
            return {}

        # StatusId 0 is success and 1 means "accepted, needs a reboot to take
        # effect" - neither is a failure. Negative ids are refusals: -1 is
        # value out of range, -4 is generic (which is what read-only
        # properties like MulticastAddress return).
        failures, pending = [], []
        for action in body.get('Actions', []):
            for result in action.get('Results', []):
                status = result.get('StatusId')
                if status in (0, None):
                    continue
                text = (f"{result.get('Path', '')}.{result.get('Property', '')}: "
                        f"{result.get('StatusInfo')} (StatusId {status})")
                (pending if status > 0 else failures).append(text)
        if failures and strict:
            raise NvxError(f'{self.host}: write refused -> ' + '; '.join(failures))
        body['_failures'] = failures
        body['_pending'] = pending
        return body

    def close(self):
        self.session.close()

    def __enter__(self):
        self.login()
        return self

    def __exit__(self, *exc):
        self.close()


def connect(host, user=None, password=None, verbose=True):
    config = load_config()
    device = NvxDevice(resolve_host(host, config), user, password)
    how = device.login()
    if verbose:
        print(f'{device.host}: {how}', file=sys.stderr)
    return device


# -- reconnaissance ------------------------------------------------------

# Objects worth pulling on a first pass. Anything that 404s is skipped, which
# is how we learn what this firmware actually exposes.
PROBE_PATHS = [
    'Device/DeviceInfo',
    'Device/DeviceSpecific',
    'Device/StreamReceive',
    'Device/StreamTransmit',
    'Device/UsbInput',
    'Device/UsbOutput',
    'Device/Usb',
    'Device/Ethernet',
    'Device/AudioVideoInputOutput',
    'Device/DiscoveryAgent',
]


def cmd_net(args):
    """Check the local network path before blaming the devices."""
    config = load_config()
    hosts = args.hosts or [t['host'] for t in load_targets(config)] or [
        '10.1.1.151', '10.1.1.152', '10.1.1.153', '10.1.1.154']
    receiver = config.get('receiver')
    if receiver:
        hosts = [resolve_host(receiver, config)] + [h for h in hosts if h != receiver]
    problems = 0
    for host in hosts:
        host = resolve_host(host, config)
        interface = route_interface(host)
        warning = local_network_warning(host)
        state = 'TUNNEL' if warning else 'ok'
        print(f'  {host:<14} via {interface or "?":<10} {state}')
        if warning:
            problems += 1
    if problems:
        print(f'\n{local_network_warning(hosts[0])}')
    else:
        print('\nlocal routing looks fine; unreachable endpoints are not a VPN.')
    return 0


def cmd_probe(args):
    config = load_config()
    hosts = args.hosts or list(config.get('hosts', {}).values()) or [
        '10.1.1.151', '10.1.1.152', '10.1.1.153', '10.1.1.154']
    warning = local_network_warning(resolve_host(hosts[0], config))
    if warning:
        print(f'WARNING: {warning}\n', file=sys.stderr)
    for host in hosts:
        host = resolve_host(host, config)
        print(f'\n=== {host} ===')
        try:
            device = NvxDevice(host, args.user, args.password)
            print(f'  login: {device.login()}')
        except NvxError as e:
            print(f'  FAILED: {e}')
            continue
        for path in PROBE_PATHS:
            try:
                data = device.get(path)
            except NvxError as e:
                print(f'  {path}: -- ({e.args[0].split("-> ")[-1]})')
                continue
            print(f'  {path}: {_summarize(data)}')
        device.close()


def _summarize(data, depth=0):
    if isinstance(data, dict):
        keys = list(data.keys())
        if depth < 1 and len(keys) == 1:
            return f'{keys[0]} -> {_summarize(data[keys[0]], depth + 1)}'
        return '{' + ', '.join(keys[:12]) + ('...' if len(keys) > 12 else '') + '}'
    if isinstance(data, list):
        return f'[{len(data)} items]'
    return repr(data)


def cmd_dump(args):
    device = connect(args.host, args.user, args.password)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tree = device.get('Device')
    path = out_dir / f'{device.host.replace(".", "_")}.json'
    with open(path, 'w') as f:
        json.dump(tree, f, indent=2, sort_keys=True)
    print(f'wrote {path}')
    device.close()


def cmd_get(args):
    device = connect(args.host, args.user, args.password)
    print(json.dumps(device.get(args.path), indent=2, sort_keys=True))
    device.close()


# -- endpoint state ------------------------------------------------------

# Index 0 is the primary stream on both transmit and receive; the other three
# slots are secondary/J2000 profiles this setup does not use.
PRIMARY = 0

NO_MAC = '00:00:00:00:00:00'


def normalize_mac(mac):
    """DeviceInfo writes MACs dot-separated, USB pairing uses colons."""
    return (mac or '').replace('.', ':').lower()


def read_state(device):
    """Everything the switch logic needs from one endpoint, in four GETs."""
    info = device.get('Device/DeviceInfo')['Device']['DeviceInfo']
    specific = device.get('Device/DeviceSpecific')['Device']['DeviceSpecific']
    usb = device.get('Device/Usb')['Device']['Usb']
    port = usb['UsbPorts'][PRIMARY]
    pairing = port['UsbPairing']
    mode = specific.get('DeviceMode')

    state = {
        'host': device.host,
        'name': info.get('Name'),
        'mac': normalize_mac(info.get('MacAddress')),
        'device_mode': mode,
        'usb_mode': port.get('Mode'),
        'usb_active': port.get('IsActive'),
        'usb_enumerated': port.get('HostEnumerated'),
        'usb_auto_pair': port.get('IsAutoUsbPairingEnabled'),
        'usb_partner': normalize_mac(pairing['Layer2']['RemoteDevices']['Id1']),
        'usb_paired': pairing['PairedStatus']['Id1'],
    }

    obj = 'StreamTransmit' if mode == 'Transmitter' else 'StreamReceive'
    stream = device.get(f'Device/{obj}')['Device'][obj]['Streams'][PRIMARY]
    state['stream_location'] = stream.get('StreamLocation', '')
    state['stream_status'] = stream.get('Status', '')
    state['stream_started'] = stream.get('Start')
    state['multicast'] = stream.get('MulticastAddress', '')
    state['resolution'] = (f"{stream.get('HorizontalResolution')}x"
                           f"{stream.get('VerticalResolution')}")
    return state


def describe(state):
    partner = state['usb_partner']
    partner = '(none)' if partner in ('', NO_MAC) else partner
    return (f"{state['host']:<12} {state['device_mode'] or '?':<12}"
            f" usb={state['usb_mode'] or '?':<7} partner={partner:<18}"
            f" paired={str(state['usb_paired']):<5}"
            f" stream={state['stream_location'] or '(none)'} [{state['stream_status']}]")


def set_usb_partner(device, mac):
    """Point this endpoint's single pairing slot at `mac` ('' clears it)."""
    return device.post({'Device': {'Usb': {'UsbPorts': [
        {'UsbPairing': {'Layer2': {'RemoteDevices': {'Id1': mac or NO_MAC}}}}
    ]}}})


def set_usb_mode(device, mode):
    """'Local' for the endpoint cabled to a computer, 'Remote' for the one
    holding the keyboard and mouse."""
    return device.post({'Device': {'Usb': {'UsbPorts': [{'Mode': mode}]}}})


def set_stream_location(device, location):
    """Subscribe a receiver to a transmitter's stream URL."""
    return device.post({'Device': {'StreamReceive': {'Streams': [
        {'StreamLocation': location}
    ]}}})


# Stream parameters both ends must agree on, or the receiver sits at
# 'Connecting' forever. Measured: 'ByReceiver' lets a transmitter start but
# never carries a packet; 'Multicast via RTSP' with MPEG2TSRTP is what works.
SESSION_INITIATION = 'Multicast via RTSP'
TRANSPORT_MODE = 'MPEG2TSRTP'
TS_PORT = 4570


def start_transmit(device, multicast=None):
    """Bring a transmitter's stream up, and configure it if it is not already.

    Three separate gates have to be cleared, none of them obvious:

    - While IsAutomaticInitiationEnabled is true, Start writes are accepted and
      silently ignored (the web UI greys out its Start button for the same
      reason). It has to be turned off first.
    - MulticastAddress is only writable while the stream is stopped, and in
      multicast mode Start is refused outright without one. Every attempt to
      set it on a running stream comes back as a generic error, which reads
      exactly like a read-only property.
    - Start and Stop are edge-triggered momentary flags, not levels: re-posting
      Start=True when it already reads True does nothing.

    Each transmitter needs its own multicast group; two sharing one address
    will collide. The final octet must be EVEN — the device takes consecutive
    addresses for video and audio, and rejects an odd base with the same
    featureless generic error it uses for everything else.
    """
    if multicast:
        try:
            last = int(multicast.rsplit('.', 1)[1])
        except (IndexError, ValueError):
            raise NvxError(f'{multicast!r} is not an IPv4 address')
        if last % 2:
            raise NvxError(
                f'multicast address {multicast} ends in an odd octet; the NVX '
                f'takes consecutive groups for video and audio and will refuse '
                f'it. Try {multicast.rsplit(".", 1)[0]}.{last - 1} or '
                f'{multicast.rsplit(".", 1)[0]}.{last + 1}')
    stream = device.get('Device/StreamTransmit')['Device']['StreamTransmit']['Streams'][PRIMARY]
    configured = (stream.get('SessionInitiation') == SESSION_INITIATION
                  and stream.get('TransportMode') == TRANSPORT_MODE
                  and stream.get('MulticastAddress')
                  and (multicast is None or stream.get('MulticastAddress') == multicast))
    if configured and 'start' in (stream.get('Status') or '').lower():
        return stream.get('MulticastAddress')

    def post(payload):
        return device.post({'Device': {'StreamTransmit': {'Streams': [payload]}}})

    post({'IsAutomaticInitiationEnabled': False})
    if not configured:
        post({'Stop': True})
        time.sleep(4)   # the encoder needs a moment before it accepts config
        post({'TransportMode': TRANSPORT_MODE})
        post({'SessionInitiation': SESSION_INITIATION})
        if multicast:
            post({'MulticastAddress': multicast})
    post({'Stop': False})
    post({'Start': False})
    post({'Start': True})
    time.sleep(4)
    stream = device.get('Device/StreamTransmit')['Device']['StreamTransmit']['Streams'][PRIMARY]
    if 'start' not in (stream.get('Status') or '').lower():
        raise NvxError(f'{device.host}: transmitter did not start '
                       f'(status {stream.get("Status")!r}, '
                       f'multicast {stream.get("MulticastAddress")!r})')
    return stream.get('MulticastAddress')


def align_receiver(device):
    """Match a receiver's session parameters to the transmitter's.

    A mismatch here leaves the receiver at 'Connecting' indefinitely with no
    error anywhere, which is indistinguishable from a dead transmitter.
    """
    for payload in ({'SessionInitiation': SESSION_INITIATION},
                    {'TransportMode': TRANSPORT_MODE},
                    {'TsPort': TS_PORT}):
        device.post({'Device': {'StreamReceive': {'Streams': [payload]}}})


def switch_receiver(receiver_host, target_host, hosts_by_mac=None,
                    user=None, password=None, video_only=False):
    """Point the receiver at one target: video subscription, then USB pairing.

    Shared by the CLI, the tkinter switcher and the dpg node. Blocking and
    network-bound, so callers with a UI should run it off their main thread.
    `hosts_by_mac` lets the previously paired target be found and released;
    without it a stale pairing is left behind on that endpoint.
    """
    rx = NvxDevice(receiver_host, user, password, timeout=20)
    rx.login()
    tx = NvxDevice(target_host, user, password, timeout=20)
    tx.login()
    try:
        rx_state, tx_state = read_state(rx), read_state(tx)
        if rx_state['device_mode'] != 'Receiver':
            raise NvxError(f'{receiver_host} is a {rx_state["device_mode"]}')
        if tx_state['device_mode'] != 'Transmitter':
            raise NvxError(f'{target_host} is a {tx_state["device_mode"]}')
        if 'start' not in (tx_state['stream_status'] or '').lower():
            start_transmit(tx)
            tx_state = read_state(tx)
        location = tx_state['stream_location'] or f'rtsp://{target_host}:554/live.sdp'

        # Only one USB partner is supported at a time, so the endpoint we are
        # leaving has to be released or it keeps a stale pairing.
        if not video_only:
            previous = rx_state['usb_partner']
            if previous not in ('', NO_MAC) and previous != tx_state['mac']:
                old_host = (hosts_by_mac or {}).get(previous)
                if old_host:
                    old = NvxDevice(old_host, user, password, timeout=20)
                    old.login()
                    try:
                        set_usb_partner(old, '')
                    finally:
                        old.close()

        align_receiver(rx)
        set_stream_location(rx, location)
        if not video_only:
            set_usb_partner(tx, rx_state['mac'])
            set_usb_partner(rx, tx_state['mac'])
        return location
    finally:
        rx.close()
        tx.close()


def stop_transmit(device):
    return device.post({'Device': {'StreamTransmit': {'Streams': [
        {'Stop': True}
    ]}}})


# -- commands ------------------------------------------------------------

def _inventory(config, user, password, hosts=None):
    hosts = hosts or list(config.get('hosts', {}).values()) or [
        '10.1.1.151', '10.1.1.152', '10.1.1.153', '10.1.1.154']
    out = {}
    for host in hosts:
        host = resolve_host(host, config)
        device = NvxDevice(host, user, password)
        device.login()
        out[host] = (device, read_state(device))
    return out


def cmd_status(args):
    config = load_config()
    inv = _inventory(config, args.user, args.password, args.hosts)
    by_mac = {s['mac']: h for h, (_, s) in inv.items()}
    for host, (device, state) in inv.items():
        line = describe(state)
        partner_host = by_mac.get(state['usb_partner'])
        if partner_host:
            line += f'  -> {partner_host}'
        print(line)
        device.close()


def cmd_transmit(args):
    """Bring a transmitter's primary stream up so receivers can subscribe."""
    device = connect(args.host, args.user, args.password)
    state = read_state(device)
    if state['device_mode'] != 'Transmitter':
        raise NvxError(f'{device.host} is a {state["device_mode"]}, not a Transmitter')
    if args.stop:
        stop_transmit(device)
        print(f'{device.host}: stream stopped')
    else:
        multicast = start_transmit(device, args.multicast)
        print(f'{device.host}: streaming on {multicast}')
    time.sleep(2)
    print(describe(read_state(device)))
    device.close()


def cmd_switch(args):
    """Point the workstation receiver at one target: video, then USB."""
    config = load_config()
    rx_host = resolve_host(args.receiver or config.get('receiver') or '', config)
    tx_host = resolve_host(args.target, config)
    if not rx_host:
        raise NvxError('no receiver: pass --receiver or set "receiver" in config')

    rx = NvxDevice(rx_host, args.user, args.password)
    rx.login()
    tx = NvxDevice(tx_host, args.user, args.password)
    tx.login()
    rx_state, tx_state = read_state(rx), read_state(tx)

    if rx_state['device_mode'] != 'Receiver':
        raise NvxError(f'{rx_host} is a {rx_state["device_mode"]}, not a Receiver')
    if tx_state['device_mode'] != 'Transmitter':
        raise NvxError(f'{tx_host} is a {tx_state["device_mode"]}, not a Transmitter')
    # Transmitters normally stay up; bring this one back if it is not.
    if 'start' not in (tx_state['stream_status'] or '').lower():
        if args.dry_run:
            print(f'would start transmitter {tx_host}')
        else:
            print(f'{tx_host}: not streaming, starting it')
            start_transmit(tx)
            tx_state = read_state(tx)
    # The transmitter derives this once running, and it is always predictable.
    location = tx_state['stream_location'] or f'rtsp://{tx_host}:554/live.sdp'

    # The endpoint we are switching away from, so its pairing can be released.
    old_partner = rx_state['usb_partner']
    old_device = None
    if (not args.video_only and old_partner not in ('', NO_MAC)
            and old_partner != tx_state['mac']):
        for host, (dev, st) in _inventory(config, args.user, args.password).items():
            if st['mac'] == old_partner:
                old_device = dev
                break

    if args.dry_run:
        print(f'would set {rx_host} StreamLocation -> {location}')
        if not args.video_only:
            if old_device:
                print(f'would clear USB pairing on {old_device.host}')
            print(f'would pair {rx_host} <-> {tx_host} ({tx_state["mac"]}) '
                  f'if pairing does not follow automatically')
        return

    start = time.monotonic()
    align_receiver(rx)
    set_stream_location(rx, location)
    print(f'{rx_host}: subscribed to {location}')

    if not args.video_only:
        # Measured on this fleet: USB does NOT follow the video route, even
        # with AvRouting.RouteControl.IsUsbFollowsVideoEnabled set on both
        # ends, so pairing is written explicitly by default. Pass a non-zero
        # --usb-wait to give automatic pairing a chance first.
        followed = False
        if args.usb_wait > 0 and rx_state['usb_auto_pair']:
            for _ in range(int(args.usb_wait * 2)):
                time.sleep(0.5)
                if read_state(rx)['usb_partner'] == tx_state['mac']:
                    followed = True
                    break
        if followed:
            print(f'USB pairing followed the video route automatically '
                  f'({time.monotonic() - start:.1f}s)')
        else:
            if old_device is not None:
                set_usb_partner(old_device, '')
                print(f'{old_device.host}: released USB pairing')
            set_usb_partner(tx, rx_state['mac'])
            set_usb_partner(rx, tx_state['mac'])
            print(f'paired {rx_host} <-> {tx_host} manually')

    # Settle: wait for the receiver to report the stream running.
    deadline = time.monotonic() + args.timeout
    final = read_state(rx)
    while time.monotonic() < deadline:
        final = read_state(rx)
        video_up = 'start' in (final['stream_status'] or '').lower()
        usb_up = args.video_only or final['usb_paired']
        if video_up and usb_up:
            break
        time.sleep(0.5)
    else:
        print('warning: did not fully settle within timeout', file=sys.stderr)

    print(describe(final))
    print(f'switch took {time.monotonic() - start:.1f}s')
    for dev in (rx, tx, old_device):
        if dev is not None:
            dev.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--user')
    parser.add_argument('--password')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('probe', help='log into each endpoint and list objects')
    p.add_argument('hosts', nargs='*')
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser('dump', help='write one endpoint\'s whole object tree')
    p.add_argument('host')
    p.add_argument('-o', '--out', default='nvx_dumps')
    p.set_defaults(func=cmd_dump)

    p = sub.add_parser('get', help='print one CresNext object')
    p.add_argument('host')
    p.add_argument('path')
    p.set_defaults(func=cmd_get)

    p = sub.add_parser('status', help='one line of role/route state per endpoint')
    p.add_argument('hosts', nargs='*')
    p.set_defaults(func=cmd_status)

    p = sub.add_parser('net', help='check local routing (a VPN can hide the devices)')
    p.add_argument('hosts', nargs='*')
    p.set_defaults(func=cmd_net)

    p = sub.add_parser('transmit', help='start (or stop) a transmitter\'s stream')
    p.add_argument('host')
    p.add_argument('--multicast',
                   help='multicast group for this transmitter, e.g. 239.1.0.1. '
                        'Each transmitter needs its own or they collide; only '
                        'settable while the stream is stopped')
    p.add_argument('--stop', action='store_true')
    p.set_defaults(func=cmd_transmit)

    p = sub.add_parser('switch', help='point the receiver at one target')
    p.add_argument('target')
    p.add_argument('--receiver')
    p.add_argument('--video-only', action='store_true',
                   help='move the video subscription, leave USB pairing alone')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--usb-wait', type=float, default=0.0,
                   help='seconds to let auto pairing follow before pairing by hand '
                        '(0 = pair explicitly straight away, which is what works here)')
    p.add_argument('--timeout', type=float, default=20.0)
    p.set_defaults(func=cmd_switch)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except NvxError as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
