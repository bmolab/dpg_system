"""
OSCQuery Service Module for dpg_system.

Provides:
- OSCQueryServer: Advertises this dpg_system instance via mDNS and serves
  the OSCQueryRegistry JSON tree over HTTP.
- OSCQueryBrowser: Discovers remote _oscjson._tcp. services on the network.
- ServiceAliasRegistry: Maps friendly names to service names.
"""

import json
import os
import socket
import threading
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

try:
    from zeroconf import Zeroconf, ServiceBrowser, ServiceInfo, ServiceStateChange
    HAS_ZEROCONF = True
except ImportError:
    HAS_ZEROCONF = False
    print("OSCQuery: zeroconf not installed. Service discovery disabled. Install with: pip install zeroconf")

from urllib.request import urlopen
from urllib.error import URLError


# ---------------------------------------------------------------------------
# HTTP handler for serving OSCQuery JSON
# ---------------------------------------------------------------------------

class OSCQueryHTTPHandler(BaseHTTPRequestHandler):
    """Serves the OSCQueryRegistry JSON tree via HTTP."""

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parsed.query
        query_params = dict(p.split('=') for p in query.split('&') if '=' in p) if query else {}

        server_data = self.server.oscquery_data  # set by OSCQueryServer

        # Subscription management
        if path == '/subscribe':
            ip = query_params.get('ip', '')
            port = query_params.get('port', '')
            if ip and port:
                subscribers = server_data.get('subscribers', [])
                entry = (ip, int(port))
                if entry not in subscribers:
                    subscribers.append(entry)
                    server_data['subscribers'] = subscribers
                self._send_json({'status': 'subscribed', 'ip': ip, 'port': int(port)})
            else:
                self.send_error(400, "Missing ip or port parameter")
            return

        if path == '/unsubscribe':
            ip = query_params.get('ip', '')
            port = query_params.get('port', '')
            if ip and port:
                subscribers = server_data.get('subscribers', [])
                entry = (ip, int(port))
                if entry in subscribers:
                    subscribers.remove(entry)
                self._send_json({'status': 'unsubscribed', 'ip': ip, 'port': int(port)})
            else:
                self.send_error(400, "Missing ip or port parameter")
            return

        # HOST_INFO request
        if query == 'HOST_INFO' or path == '/HOST_INFO':
            host_info = server_data.get('HOST_INFO', {})
            self._send_json(host_info)
            return

        # Navigate to the requested path in the registry
        registry = server_data.get('registry', {})
        if path == '/' or path == '':
            result = registry
        else:
            result = self._navigate_to_path(registry, path)

        if result is None:
            self.send_error(404, "Path not found in OSC address space")
            return

        # If a specific attribute is queried (e.g. ?VALUE)
        if query and query in result:
            self._send_json(result[query])
        else:
            self._send_json(result)

    def _navigate_to_path(self, registry, path):
        """Navigate the registry tree to find the node at the given path."""
        components = [c for c in path.split('/') if c]
        current = registry
        for comp in components:
            if 'CONTENTS' in current and comp in current['CONTENTS']:
                current = current['CONTENTS'][comp]
            elif comp in current:
                current = current[comp]
            else:
                return None
        return current

    def _send_json(self, data):
        response = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)


# ---------------------------------------------------------------------------
# OSCQueryServer — advertises this instance
# ---------------------------------------------------------------------------

class OSCQueryServer:
    """
    Starts an HTTP server serving the OSCQueryRegistry JSON and advertises
    it via mDNS as _oscjson._tcp.

    Supports advertising multiple named services from a single instance.
    """

    def __init__(self, registry, default_osc_port=2500, service_name=None):
        """
        Args:
            registry: The OSCQueryRegistry instance to serve.
            default_osc_port: The default OSC UDP port for this instance.
            service_name: Default service name for advertisement.
        """
        self.registry = registry
        self.default_osc_port = default_osc_port
        self.service_name = service_name or 'dpg_system'

        self.http_server = None
        self.http_thread = None
        self.http_port = 0

        self.zeroconf = None
        self.advertised_services = {}  # name -> ServiceInfo

        self._running = False

    def start(self):
        """Start the HTTP server and advertise via mDNS."""
        if self._running:
            return

        # Find a free port for HTTP
        self.http_port = self._find_free_port()
        if self.http_port is None:
            print("OSCQueryServer: Could not find free port for HTTP server")
            return

        # Start HTTP server
        self.http_server = HTTPServer(('0.0.0.0', self.http_port), OSCQueryHTTPHandler)
        self.http_server.oscquery_data = {
            'registry': self.registry.registry,
            'HOST_INFO': {
                'NAME': self.service_name,
                'OSC_PORT': self.default_osc_port,
                'OSC_TRANSPORT': 'UDP',
                'EXTENSIONS': {
                    'ACCESS': True,
                    'VALUE': True,
                    'RANGE': True,
                    'DESCRIPTION': True,
                    'LISTEN': True,
                }
            },
            'subscribers': [],  # list of (ip, port) tuples for peer push
        }
        self.http_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
        self.http_thread.start()

        # Advertise via Zeroconf
        if HAS_ZEROCONF:
            try:
                self.zeroconf = Zeroconf()
                self.advertise_service(self.service_name, self.default_osc_port)
            except Exception as e:
                print(f"OSCQueryServer: Zeroconf advertisement failed: {e}")
                traceback.print_exception(e)

        self._running = True
        print(f"OSCQueryServer: HTTP on port {self.http_port}, advertising as '{self.service_name}'")

    def stop(self):
        """Stop the HTTP server and remove mDNS advertisements."""
        self._running = False

        # Remove all advertised services
        if self.zeroconf:
            for name, info in list(self.advertised_services.items()):
                try:
                    self.zeroconf.unregister_service(info)
                except Exception:
                    pass
            self.advertised_services.clear()
            try:
                self.zeroconf.close()
            except Exception:
                pass
            self.zeroconf = None

        # Stop HTTP server
        if self.http_server:
            self.http_server.shutdown()
            self.http_server = None
        if self.http_thread:
            self.http_thread.join(timeout=2.0)
            self.http_thread = None

    def advertise_service(self, name, osc_port):
        """
        Advertise a named service via mDNS.

        Args:
            name: Service name (e.g. 'speech_to_text', 'eos')
            osc_port: The OSC UDP port for this service
        """
        if not HAS_ZEROCONF or self.zeroconf is None:
            return

        if name in self.advertised_services:
            # Already advertised
            return

        local_ip = self._get_local_ip()
        if local_ip is None:
            print(f"OSCQueryServer: Could not determine local IP for advertising '{name}'")
            return

        try:
            info = ServiceInfo(
                "_oscjson._tcp.local.",
                f"{name}._oscjson._tcp.local.",
                addresses=[socket.inet_aton(local_ip)],
                port=self.http_port,
                properties={
                    'OSC_PORT': str(osc_port),
                    'OSC_TRANSPORT': 'UDP',
                },
                server=f"{socket.gethostname()}.local.",
            )
            self.zeroconf.register_service(info)
            self.advertised_services[name] = info
            print(f"OSCQueryServer: Advertised '{name}' (OSC port {osc_port}, HTTP port {self.http_port})")
        except Exception as e:
            print(f"OSCQueryServer: Failed to advertise '{name}': {e}")

    def unadvertise_service(self, name):
        """Remove a named service from mDNS advertisement."""
        if name in self.advertised_services:
            if self.zeroconf:
                try:
                    self.zeroconf.unregister_service(self.advertised_services[name])
                except Exception:
                    pass
            del self.advertised_services[name]

    def update_host_info(self, key, value):
        """Update a HOST_INFO field on the HTTP server."""
        if self.http_server and hasattr(self.http_server, 'oscquery_data'):
            self.http_server.oscquery_data['HOST_INFO'][key] = value

    def get_subscribers(self):
        """Return list of (ip, port) subscriber tuples."""
        if self.http_server and hasattr(self.http_server, 'oscquery_data'):
            return list(self.http_server.oscquery_data.get('subscribers', []))
        return []

    def notify_subscribers(self, osc_address, value):
        """Push an OSC message to all subscribed peers."""
        subscribers = self.get_subscribers()
        if not subscribers:
            return
        try:
            from pythonosc.udp_client import SimpleUDPClient
        except ImportError:
            return
        for ip, port in subscribers:
            try:
                client = SimpleUDPClient(ip, port)
                if isinstance(value, list):
                    client.send_message(osc_address, value)
                else:
                    client.send_message(osc_address, [value])
            except Exception:
                pass

    def _find_free_port(self, start=9000, end=9099):
        """Find a free TCP port for the HTTP server."""
        for port in range(start, end):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('', port))
                sock.close()
                return port
            except OSError:
                continue
        return None

    def _get_local_ip(self):
        """Get the local IP address of this machine."""
        try:
            # Connect to an external address to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"


# ---------------------------------------------------------------------------
# OSCQueryBrowser — discovers remote services
# ---------------------------------------------------------------------------

class DiscoveredService:
    """Represents a discovered OSCQuery service on the network."""

    def __init__(self, name, ip, http_port, osc_port, properties=None):
        self.name = name
        self.ip = ip
        self.http_port = http_port
        self.osc_port = osc_port
        self.properties = properties or {}
        self.json_tree = None
        self.last_fetch_error = None

    def fetch_json(self, retries=3, delay=0.5):
        """Fetch the full OSCQuery JSON tree from this service, with retries."""
        import time
        
        urls_to_try = [f"http://{self.ip}:{self.http_port}/"]
        # If not already localhost, add a localhost fallback
        if self.ip not in ('127.0.0.1', 'localhost'):
            urls_to_try.append(f"http://127.0.0.1:{self.http_port}/")
        
        last_exception = None
        for url in urls_to_try:
            for attempt in range(retries):
                try:
                    response = urlopen(url, timeout=2)
                    self.json_tree = json.loads(response.read().decode('utf-8'))
                    self.last_fetch_error = None
                    # If localhost worked, remember it for future fetches
                    if '127.0.0.1' in url and self.ip != '127.0.0.1':
                        self.ip = '127.0.0.1'
                    return self.json_tree
                except Exception as e:
                    last_exception = e
                    if attempt < retries - 1:
                        time.sleep(delay)
                    
        self.last_fetch_error = str(last_exception)
        print(f"OSCQueryBrowser: Failed to fetch JSON from {self.name} ({urls_to_try[0]}) after {retries} attempts: {last_exception}")
        return None

    def fetch_param(self, osc_path):
        """Fetch a single parameter's current state via HTTP.
        Returns the param dict (with TYPE, VALUE, etc.) or None."""
        url = f"http://{self.ip}:{self.http_port}{osc_path}"
        try:
            response = urlopen(url, timeout=3)
            param = json.loads(response.read().decode('utf-8'))
            return param
        except Exception as e:
            return None

    def __repr__(self):
        return f"DiscoveredService(name='{self.name}', ip='{self.ip}', osc_port={self.osc_port}, http_port={self.http_port})"


class OSCQueryBrowser:
    """
    Discovers remote _oscjson._tcp. services via mDNS/Zeroconf.
    Maintains a live catalog of discovered services.
    """

    def __init__(self):
        self.services = {}  # name -> DiscoveredService
        self.zeroconf = None
        self.browser = None
        self._lock = threading.Lock()
        self._callbacks = []  # list of callables: callback(event, service_name)
        self._running = False

    def start(self):
        """Start browsing for _oscjson._tcp. services."""
        if not HAS_ZEROCONF:
            print("OSCQueryBrowser: zeroconf not available, discovery disabled")
            return
        if self._running:
            return

        try:
            self.zeroconf = Zeroconf()
            self.browser = ServiceBrowser(
                self.zeroconf,
                "_oscjson._tcp.local.",
                handlers=[self._on_service_state_change]
            )
            self._running = True
            print("OSCQueryBrowser: Started browsing for _oscjson._tcp. services")
        except Exception as e:
            print(f"OSCQueryBrowser: Failed to start: {e}")
            traceback.print_exception(e)

    def stop(self):
        """Stop browsing."""
        self._running = False
        if self.browser:
            self.browser.cancel()
            self.browser = None
        if self.zeroconf:
            try:
                self.zeroconf.close()
            except Exception:
                pass
            self.zeroconf = None

    def add_callback(self, callback):
        """Add a callback for service events. callback(event, service_name) where event is 'added' or 'removed'."""
        self._callbacks.append(callback)

    def remove_callback(self, callback):
        """Remove a previously added callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def get_service_names(self):
        """Return list of discovered service names."""
        with self._lock:
            return list(self.services.keys())

    def get_service(self, name):
        """Get a DiscoveredService by name. Returns None if not found."""
        with self._lock:
            return self.services.get(name)

    def refresh_service(self, name):
        """Re-fetch the JSON tree for a specific service."""
        service = self.get_service(name)
        if service:
            service.fetch_json()
            return service.json_tree
        return None

    def search_param(self, query, case_insensitive=True):
        """
        Search for a parameter name across all discovered services.

        Returns a list of tuples: [(service_name, full_osc_path, param_dict), ...]
        """
        results = []
        with self._lock:
            for svc_name, svc in self.services.items():
                if svc.json_tree is None:
                    continue
                matches = self._search_tree(svc.json_tree, query, '', case_insensitive)
                for path, param_dict in matches:
                    results.append((svc_name, path, param_dict))
        return results

    def _search_tree(self, node, query, current_path, case_insensitive):
        """Recursively search a JSON tree for parameters matching query."""
        results = []
        if not isinstance(node, dict):
            return results

        query_lower = query.lower() if case_insensitive else query

        if 'CONTENTS' in node:
            for key, child in node['CONTENTS'].items():
                child_path = current_path + '/' + key
                key_compare = key.lower() if case_insensitive else key

                # Check if this key matches the query
                if query_lower in key_compare:
                    # If it's a leaf (has TYPE), add it
                    if 'TYPE' in child:
                        results.append((child_path, child))
                    # If it's a container, add all its leaf children
                    elif 'CONTENTS' in child:
                        leaves = self._collect_leaves(child, child_path)
                        results.extend(leaves)

                # Recurse into children regardless
                results.extend(self._search_tree(child, query, child_path, case_insensitive))

        return results

    def _collect_leaves(self, node, current_path):
        """Collect all leaf parameters (nodes with TYPE) under a tree node."""
        results = []
        if not isinstance(node, dict):
            return results
        if 'TYPE' in node:
            results.append((current_path, node))
        if 'CONTENTS' in node:
            for key, child in node['CONTENTS'].items():
                child_path = current_path + '/' + key
                results.extend(self._collect_leaves(child, child_path))
        return results

    def _on_service_state_change(self, zeroconf, service_type, name, state_change):
        """Callback from Zeroconf ServiceBrowser."""
        # Extract the service name from the full mDNS name
        # e.g. "speech_to_text._oscjson._tcp.local." -> "speech_to_text"
        service_name = name.replace(f".{service_type}", "").strip('.')

        if state_change == ServiceStateChange.Added or state_change == ServiceStateChange.Updated:
            # Fetch service info in a background thread to avoid blocking
            threading.Thread(
                target=self._resolve_and_add,
                args=(zeroconf, service_type, name, service_name),
                daemon=True
            ).start()

        elif state_change == ServiceStateChange.Removed:
            with self._lock:
                if service_name in self.services:
                    del self.services[service_name]
                    print(f"OSCQueryBrowser: Service removed: '{service_name}'")
            self._notify_callbacks('removed', service_name)

    def _resolve_and_add(self, zeroconf, service_type, name, service_name):
        """Resolve service info and add to catalog."""
        try:
            info = zeroconf.get_service_info(service_type, name, timeout=5000)
            if info is None:
                return

            # Get IP address
            addresses = info.parsed_addresses()
            if not addresses:
                return
            ip = addresses[0]

            # If the discovered IP is our own machine, use 127.0.0.1 for reliable HTTP access
            local_ips = self._get_local_ips()
            if ip in local_ips:
                ip = '127.0.0.1'

            # Get HTTP port (the port advertised by the service)
            http_port = info.port

            # Get OSC port from TXT record
            properties = {}
            if info.properties:
                for key, val in info.properties.items():
                    if isinstance(key, bytes):
                        key = key.decode('utf-8')
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    properties[key] = val

            osc_port = int(properties.get('OSC_PORT', 0))

            service = DiscoveredService(
                name=service_name,
                ip=ip,
                http_port=http_port,
                osc_port=osc_port,
                properties=properties
            )

            # Fetch the JSON tree
            service.fetch_json()

            with self._lock:
                self.services[service_name] = service

            print(f"OSCQueryBrowser: Discovered '{service_name}' at {ip}:{osc_port} (HTTP:{http_port})")
            self._notify_callbacks('added', service_name)

        except Exception as e:
            print(f"OSCQueryBrowser: Failed to resolve '{service_name}': {e}")

    @staticmethod
    def _get_local_ips():
        """Return a set of this machine's IP addresses."""
        import socket
        local_ips = {'127.0.0.1'}
        try:
            hostname = socket.gethostname()
            for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
                local_ips.add(info[4][0])
        except Exception:
            pass
        # Also try the "connect to external" trick
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            local_ips.add(s.getsockname()[0])
            s.close()
        except Exception:
            pass
        return local_ips

    def _notify_callbacks(self, event, service_name):
        """Notify all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(event, service_name)
            except Exception as e:
                print(f"OSCQueryBrowser callback error: {e}")


# ---------------------------------------------------------------------------
# ServiceAliasRegistry — maps friendly names to service names
# ---------------------------------------------------------------------------

class ServiceAliasRegistry:
    """
    Maps friendly alias names to actual mDNS service names.
    e.g. {"lights": "eos", "lighting": "eos", "audio": "digico"}

    Stored as a JSON file for persistence.
    """

    def __init__(self, config_path=None):
        self.aliases = {}
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), 'oscquery_aliases.json'
        )
        self.load()

    def load(self):
        """Load aliases from the config file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.aliases = json.load(f)
            except Exception as e:
                print(f"ServiceAliasRegistry: Failed to load {self.config_path}: {e}")
                self.aliases = {}

    def save(self):
        """Save aliases to the config file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.aliases, f, indent=2)
        except Exception as e:
            print(f"ServiceAliasRegistry: Failed to save {self.config_path}: {e}")

    def resolve(self, name):
        """
        Resolve a name through aliases.
        Returns the canonical service name, or the input name if no alias exists.
        """
        # Case-insensitive lookup
        name_lower = name.lower()
        for alias, target in self.aliases.items():
            if alias.lower() == name_lower:
                return target
        return name

    def add_alias(self, alias, service_name):
        """Add or update an alias mapping."""
        self.aliases[alias.lower()] = service_name
        self.save()

    def remove_alias(self, alias):
        """Remove an alias."""
        alias_lower = alias.lower()
        if alias_lower in self.aliases:
            del self.aliases[alias_lower]
            self.save()

    def get_aliases_for(self, service_name):
        """Get all aliases that map to a given service name."""
        return [alias for alias, target in self.aliases.items() if target == service_name]

    def get_all(self):
        """Return a copy of all alias mappings."""
        return dict(self.aliases)
