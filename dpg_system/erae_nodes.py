"""Embodme Erae 2 (and Erae Touch) over the Erae API - SysEx on the 'Erae 2 MIDI' port.

The Erae speaks ordinary MIDI / MPE on its own, and midi_note_in etc. already
handle that. What this node adds is the API layer, which the plain MIDI path
cannot give you:

  * the finger stream - every touch on an API Zone as a float x, y, z with a
    persistent finger identity, rather than a note number and a bent pitch
  * drawing on the pad - pixels, rectangles and whole images into an API Zone

API Zones are laid out on the device with Embodme's Erae Lab; each has an index
0..127.  A zone that is not in the current layout answers a boundary request
with size (127, 127), and sends no fingers.

Protocol: Embodme "ERAE API V2 - Rev 004 - 2025/04/24" and the reference code
at gitlab.com/embodme/erae_api_sysex.  All multi-byte payloads (finger id,
xyz floats, image RGB) travel 7-bitized: every 7 data bytes are preceded by one
byte carrying their top bits, and an XOR checksum of the encoded bytes follows.
"""
import struct
import time
import threading
import numpy as np
import mido

from dpg_system.node import Node
from dpg_system.conversion_utils import *
from dpg_system.midi_nodes import MidiDeviceNode, MidiIn, MidiOut


def register_erae_nodes():
    Node.app.register_node('erae', EraeNode.factory)
    Node.app.register_node('finger_zones', FingerZonesNode.factory)


# ----------------------------------------------------------------- protocol
EMBODME_ID = [0x00, 0x21, 0x50]
ERAE_FAMILY = [0x00, 0x01]
PRODUCT_MEMBER = {'Erae 2': [0x00, 0x02], 'Erae Touch': [0x00, 0x01]}
NETWORK_ID = 0x01
SERVICE = 0x01
API_SERVICE = 0x04

CMD_API_ENABLE = 0x01
CMD_API_DISABLE = 0x02
CMD_ZONE_BOUNDARY = 0x10
CMD_CLEAR_ZONE = 0x20
CMD_DRAW_PIXEL = 0x21
CMD_DRAW_RECT = 0x22
CMD_DRAW_IMAGE = 0x23
CMD_VERSION = 0x7F

REPLY_NON_FINGER = 0x7F
REPLY_ZONE_BOUNDARY = 0x01
REPLY_VERSION = 0x02

ACTION_DOWN, ACTION_MOVE, ACTION_UP = 0, 1, 2
ACTION_NAMES = ['down', 'move', 'up']

# The spec's own advice: keep draw_image payloads at 32 pixels or fewer so the
# OS MIDI driver never has to split a SysEx.
IMAGE_CHUNK_PIXELS = 32

# Finger x, y arrive in zone pixel units; the LED grid sits half a pixel left of and
# below the touch grid, so this is added to the pixel coordinates before the halos
# are drawn. Tuned on the pad. The node exposes it as 'halo offset'.
PIXEL_CENTRE = 0.5


def bitize7(data):
    """8-bit bytes -> 7-bit stream, XOR checksum appended (spec 'bitize7chksum')."""
    out = []
    for i in range(0, len(data), 7):
        block = data[i:i + 7]
        top = 0
        for j, el in enumerate(block):
            top |= (int(el) & 0x80) >> (j + 1)
        out.append(top)
        out.extend(int(el) & 0x7F for el in block)
    chk = 0
    for b in out:
        chk ^= b
    return out + [chk]


def unbitize7(data):
    """7-bit stream (without its checksum) -> 8-bit bytes."""
    out = []
    n = len(data)
    i = 0
    while i < n:
        top = data[i]
        for j in range(7):
            k = i + j + 1
            if k >= n:
                break
            out.append(((top << (j + 1)) & 0x80) | data[k])
        i += 8
    return out


def xor_checksum(data):
    chk = 0
    for b in data:
        chk ^= b
    return chk


def bitized7size(length):
    return length // 7 * 8 + ((1 + length % 7) if (length % 7 > 0) else 0)


FINGER_ID_LEN = bitized7size(8)    # 10 bytes on the wire
XYZ_LEN = bitized7size(12)         # 14 bytes on the wire


def _byte(v, lo=0, hi=127):
    try:
        v = int(round(float(v)))
    except (TypeError, ValueError):
        v = lo
    return max(lo, min(hi, v))


def _rgb7(values):
    """Colour components given 0..255 (or 0..1 floats) -> the 7-bit bytes the pixel/rect commands take."""
    out = []
    for v in values:
        v = float(v)
        if 0.0 <= v <= 1.0:
            v = v * 255.0
        out.append(_byte(v, 0, 255) >> 1)
    return out


class EraeNode(MidiDeviceNode):
    """erae - Embodme Erae 2 finger stream and display, via the Erae API."""

    @staticmethod
    def factory(name, data, args=None):
        node = EraeNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        self.receiver_prefix = [0x45, 0x52, 0x41]      # 'E' 'R' 'A' - any 1..16 bytes; tags replies as ours
        self.api_enabled = False
        self.zone_sizes = {}
        self.finger_slots = {}          # 64-bit finger id -> small slot index
        self.slot_free = []
        self.next_slot = 0
        # live fingers for the halo display: id -> (zone, x, y, z); written on the MIDI thread
        self.fingers = {}
        self.fingers_lock = threading.Lock()
        self.display_dirty = False
        self.last_canvas = None
        self.last_frame_time = 0.0
        self.last_boundary_request = 0.0
        self.background = None          # float rgb, row 0 = bottom, drawn under the halos

        port_name = 'Erae 2 MIDI'
        if args:
            val, t = decode_arg(args, 0)
            if t == str:
                port_name = val
        super().__init__(label, data, [port_name])

        self.product = self.add_option('product', widget_type='combo', default_value='Erae 2', callback=self.product_changed)
        self.product.widget.combo_items = list(PRODUCT_MEMBER.keys())
        self.named_actions = self.add_option('action as name', widget_type='checkbox', default_value=False)
        self.image_x = self.add_option('image x', widget_type='input_int', default_value=0, min=0, max=127)
        self.image_y = self.add_option('image y', widget_type='input_int', default_value=0, min=0, max=127)
        self.debug = self.add_option('debug', widget_type='checkbox', default_value=False)

        # halo display: a ring round each finger, radius following pressure, like the
        # 'slide' look of Erae Lab's key element - drawn by the node into the API zone
        self.show_touches = self.add_option('show touches', widget_type='checkbox', default_value=False, callback=self.show_touches_changed)
        self.halo_color = self.add_option('halo color', widget_type='color_picker', default_value=[1.0, 0.55, 0.1, 1.0])
        self.back_color = self.add_option('background', widget_type='color_picker', default_value=[0.03, 0.03, 0.1, 1.0])
        self.halo_min = self.add_option('halo min radius', widget_type='drag_float', default_value=1.0, min=0.0, max=20.0)
        self.halo_max = self.add_option('halo max radius', widget_type='drag_float', default_value=5.0, min=0.0, max=30.0)
        self.halo_width = self.add_option('halo width', widget_type='drag_float', default_value=1.0, min=0.3, max=6.0)
        self.pressure_range = self.add_option('pressure range', widget_type='drag_float', default_value=1.0, min=0.01, max=1000.0)
        self.display_fps = self.add_option('display fps', widget_type='drag_float', default_value=30.0, min=1.0, max=100.0)
        self.halo_offset = self.add_option('halo offset', widget_type='drag_float', default_value=PIXEL_CENTRE, min=-3.0, max=3.0)

    def create_properties_inputs_and_outputs(self):
        self.api_mode = self.add_input('api mode', widget_type='checkbox', default_value=True, callback=self.api_mode_changed)
        self.zone = self.add_input('zone', widget_type='input_int', default_value=0, min=0, max=127)
        self.query_in = self.add_input('query', widget_type='button', callback=self.query_zone)
        self.clear_in = self.add_input('clear', widget_type='button', callback=self.clear_zone)
        self.pixel_in = self.add_input('pixel', triggers_execution=True)
        self.rect_in = self.add_input('rect', triggers_execution=True)
        self.image_in = self.add_input('image', triggers_execution=True)
        self.backdrop_in = self.add_input('backdrop', triggers_execution=True)

        self.touch_out = self.add_output('touch')
        self.zone_size_out = self.add_output('zone size')
        self.version_out = self.add_output('api version')
        self.midi_out_raw = self.add_output('midi received')

    def custom_create(self, from_file):
        if self.api_mode():
            self.enable_api()

    def custom_cleanup(self):
        self.remove_frame_tasks()
        if self.last_canvas is not None:
            self.clear_zone()
        self.disable_api()
        super().custom_cleanup()

    # ------------------------------------------------------------ transport
    def header(self):
        member = PRODUCT_MEMBER.get(self.product(), PRODUCT_MEMBER['Erae 2'])
        return EMBODME_ID + ERAE_FAMILY + member + [NETWORK_ID, SERVICE, API_SERVICE]

    def send_api(self, body):
        if self.out_port is None or getattr(self.out_port, 'port', None) is None:
            return
        data = self.header() + body
        for b in data:
            if b < 0 or b > 127:
                print('erae: refusing to send byte outside 0..127:', data)
                return
        if self.debug():
            print('erae ->', ' '.join('%02X' % b for b in data))
        try:
            self.out_port.send(mido.Message('sysex', data=data))
        except Exception as e:
            print('erae: send failed:', e)

    def enable_api(self):
        # Spec: disable before every enable, so a stale session never doubles up.
        self.send_api([CMD_API_DISABLE])
        self.send_api([CMD_API_ENABLE] + self.receiver_prefix)
        self.api_enabled = True
        self.finger_slots.clear()
        self.slot_free.clear()
        self.next_slot = 0
        with self.fingers_lock:
            self.fingers.clear()
            self.display_dirty = True
        self.last_canvas = None
        self.send_api([CMD_VERSION] + self.receiver_prefix)
        self.send_api([CMD_ZONE_BOUNDARY, _byte(self.zone())])

    def disable_api(self):
        if self.api_enabled:
            self.send_api([CMD_API_DISABLE])
        self.api_enabled = False

    def api_mode_changed(self):
        if self.api_mode():
            self.enable_api()
        else:
            self.disable_api()

    def product_changed(self):
        if self.api_enabled:
            self.enable_api()

    def port_changed(self):
        was_enabled = self.api_enabled
        if was_enabled:
            self.disable_api()
        super().port_changed()
        if was_enabled:
            self.enable_api()

    # ------------------------------------------------------------- commands
    def query_zone(self):
        self.send_api([CMD_ZONE_BOUNDARY, _byte(self.zone())])

    def clear_zone(self):
        self.send_api([CMD_CLEAR_ZONE, _byte(self.zone())])

    def draw_pixel(self, values):
        # x y r g b   (colour 0..255, or 0..1 floats)
        if len(values) < 5:
            print('erae: pixel wants x y r g b')
            return
        x, y = _byte(values[0]), _byte(values[1])
        self.send_api([CMD_DRAW_PIXEL, _byte(self.zone()), x, y] + _rgb7(values[2:5]))

    def draw_rect(self, values):
        # x y w h r g b
        if len(values) < 7:
            print('erae: rect wants x y w h r g b')
            return
        x, y, w, h = (_byte(v) for v in values[:4])
        self.send_api([CMD_DRAW_RECT, _byte(self.zone()), x, y, w, h] + _rgb7(values[4:7]))

    def draw_image(self, image):
        """image rows top-to-bottom as any picture; the Erae's origin is bottom-left, so it is flipped on the way out."""
        img = any_to_array(image)
        if img is None or img.ndim < 2:
            print('erae: image wants an array of shape (height, width, 3) or (height, width)')
            return
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
        if np.issubdtype(img.dtype, np.floating):
            top = float(img.max()) if img.size else 0.0
            if top <= 1.0:
                img = img * 255.0
        rgb = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        rgb = rgb[::-1, :, :]            # row 0 becomes the bottom row on the device
        height, width = rgb.shape[:2]
        width = min(width, 128)
        height = min(height, 128)
        zone = _byte(self.zone())
        x0, y0 = _byte(self.image_x()), _byte(self.image_y())
        # one message per run of <= 32 pixels; rows split when they are wider than that
        for row in range(height):
            y = y0 + row
            if y > 127:
                break
            col = 0
            while col < width:
                run = min(IMAGE_CHUNK_PIXELS, width - col)
                x = x0 + col
                if x > 127:
                    break
                run = min(run, 128 - x)
                flat = rgb[row, col:col + run, :].reshape(-1).tolist()
                self.send_api([CMD_DRAW_IMAGE, zone, x, y, run, 1] + bitize7(flat))
                col += run

    def execute(self):
        if self.active_input is self.pixel_in:
            self.draw_pixel(any_to_numerical_list(self.pixel_in()))
        elif self.active_input is self.rect_in:
            self.draw_rect(any_to_numerical_list(self.rect_in()))
        elif self.active_input is self.image_in:
            self.draw_image(self.image_in())
        elif self.active_input is self.backdrop_in:
            self.set_background(self.backdrop_in())

    def set_background(self, image):
        """A picture (top row first) to draw the halos over; anything that is not an image clears it."""
        img = any_to_array(image)
        if img is None or img.ndim < 2:
            back = None
        else:
            if img.ndim == 2:
                img = np.repeat(img[:, :, None], 3, axis=2)
            elif img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            elif img.shape[2] > 3:
                img = img[:, :, :3]
            back = img.astype(np.float32)
            if not np.issubdtype(img.dtype, np.floating) or (back.size and back.max() > 1.0):
                back = back / 255.0
            back = np.clip(back[::-1, :, :], 0.0, 1.0)      # row 0 = bottom, like the halo canvas
        with self.fingers_lock:
            self.background = back
            self.display_dirty = True

    # -------------------------------------------------------------- receive
    def receive_midi_bytes(self, midi_bytes):
        if len(midi_bytes) == 0:
            return
        if midi_bytes[0] != 0xF0:
            # The pad streams MIDI clock (0xF8) all the time it is on - roughly
            # 200 a second. Real-time bytes carry nothing a patch wants here.
            if midi_bytes[0] < 0xF8:
                self.midi_out_raw.send(midi_bytes)
            return
        try:
            self.parse_sysex(list(midi_bytes[1:-1]) if midi_bytes[-1] == 0xF7 else list(midi_bytes[1:]))
        except Exception as e:
            print('erae: bad sysex', ' '.join('%02X' % b for b in midi_bytes), e)

    def parse_sysex(self, data):
        n = len(self.receiver_prefix)
        if len(data) <= n or data[:n] != self.receiver_prefix:
            return                      # some other device's, or not for this receiver
        if self.debug():
            print('erae <-', ' '.join('%02X' % b for b in data))
        off = n
        if data[off] == REPLY_NON_FINGER:
            kind = data[off + 1]
            if kind == REPLY_ZONE_BOUNDARY and len(data) >= off + 5:
                zone, width, height = data[off + 2], data[off + 3], data[off + 4]
                if width == 0x7F and height == 0x7F:
                    self.zone_sizes.pop(zone, None)
                    self.zone_size_out.send([zone, 0, 0])      # not in the current layout
                else:
                    self.zone_sizes[zone] = (width, height)
                    self.zone_size_out.send([zone, width, height])
            elif kind == REPLY_VERSION and len(data) >= off + 3:
                self.version_out.send(int(data[off + 2]))
            return

        # finger stream: ACTION ZONE FIN[10] XYZ[14] CHK
        if len(data) < off + 2 + FINGER_ID_LEN + XYZ_LEN + 1:
            return
        action_byte = data[off]
        zone = data[off + 1]
        off += 2
        finger_raw = data[off:off + FINGER_ID_LEN]
        off += FINGER_ID_LEN
        xyz_raw = data[off:off + XYZ_LEN]
        chk = data[off + XYZ_LEN]
        if xor_checksum(xyz_raw) != chk:
            print('erae: finger xyz checksum mismatch')
            return
        finger_id = struct.unpack('<Q', bytes(unbitize7(finger_raw)[:8]))[0]
        x, y, z = struct.unpack('<fff', bytes(unbitize7(xyz_raw)[:12]))

        # Embodme's V2 reference code reads the action from the low bits; the
        # older V1 sheet put it in bits 4..6 above a finger index. Take whichever
        # is populated.
        action = (action_byte >> 4) & 0x07 if (action_byte & 0x70) else (action_byte & 0x07)
        slot = self.slot_for(finger_id, action)
        if action == ACTION_UP:
            z = 0.0                     # the device repeats the last pressure on release; a lift is zero pressure
        with self.fingers_lock:
            if action == ACTION_UP:
                self.fingers.pop(finger_id, None)
            else:
                self.fingers[finger_id] = (zone, float(x), float(y), float(z))
            self.display_dirty = True
        if self.named_actions():
            act = ACTION_NAMES[action] if action < 3 else action
        else:
            act = action
        self.touch_out.send([slot, zone, act, float(x), float(y), float(z)])

    def slot_for(self, finger_id, action):
        slot = self.finger_slots.get(finger_id)
        if slot is None:
            if self.slot_free:
                slot = min(self.slot_free)
                self.slot_free.remove(slot)
            else:
                slot = self.next_slot
                self.next_slot += 1
            self.finger_slots[finger_id] = slot
        if action == ACTION_UP:
            self.finger_slots.pop(finger_id, None)
            self.slot_free.append(slot)
            if not self.finger_slots:
                self.slot_free.clear()
                self.next_slot = 0
        return slot

    # -------------------------------------------------------------- halo display
    def show_touches_changed(self):
        if self.show_touches():
            self.last_canvas = None
            with self.fingers_lock:
                self.display_dirty = True
            self.add_frame_task()
        else:
            self.remove_frame_tasks()
            if self.last_canvas is not None:
                self.clear_zone()
                self.last_canvas = None

    def frame_task(self):
        """Main thread, once per tick: redraw the zone if a finger moved, at most 'display fps' times a second."""
        if not self.show_touches() or not self.api_enabled:
            return
        zone = _byte(self.zone())
        size = self.zone_sizes.get(zone)
        now = time.time()
        if size is None:
            if now - self.last_boundary_request > 1.0:
                self.last_boundary_request = now
                self.send_api([CMD_ZONE_BOUNDARY, zone])
            return
        if now - self.last_frame_time < 1.0 / max(1.0, float(self.display_fps())):
            return
        with self.fingers_lock:
            if not self.display_dirty and self.last_canvas is not None:
                return
            self.display_dirty = False
            fingers = [f for f in self.fingers.values() if f[0] == zone]
        self.last_frame_time = now
        canvas = self.render_halos(size, fingers)
        self.send_canvas(zone, canvas)

    def render_halos(self, size, fingers):
        width, height = size
        back = np.array(self.back_color()[:3], dtype=np.float32)
        halo = np.array(self.halo_color()[:3], dtype=np.float32)
        canvas = np.empty((height, width, 3), dtype=np.float32)
        canvas[:] = back
        with self.fingers_lock:
            background = self.background
        if background is not None:
            h = min(height, background.shape[0])
            w = min(width, background.shape[1])
            canvas[:h, :w] = np.maximum(canvas[:h, :w], background[:h, :w])
        if fingers:
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
            offset = float(self.halo_offset())      # larger moves the halos left and down
            yy += offset
            xx += offset
            r_min, r_max = float(self.halo_min()), float(self.halo_max())
            ring_w = max(0.3, float(self.halo_width()))
            p_range = max(1e-6, float(self.pressure_range()))
            for _zone, x, y, z in fingers:
                pressure = min(1.0, max(0.0, z / p_range))
                radius = r_min + pressure * (r_max - r_min)
                d = np.hypot(xx - x, yy - y)
                ring = np.clip(1.0 - np.abs(d - radius) / ring_w, 0.0, 1.0)
                dot = np.clip(1.2 - d, 0.0, 1.0)
                k = np.maximum(ring, dot)
                canvas = np.maximum(canvas, k[:, :, None] * halo)
        return np.clip(np.rint(canvas * 255.0), 0, 255).astype(np.uint8)

    def send_canvas(self, zone, canvas):
        """Send the rows that changed since the last frame, as runs of <= 32 pixels, in zone coordinates (row 0 = bottom)."""
        height, width = canvas.shape[:2]
        if self.last_canvas is None or self.last_canvas.shape != canvas.shape:
            changed_rows = range(height)
            spans = {row: (0, width) for row in changed_rows}
        else:
            diff = np.any(canvas != self.last_canvas, axis=2)
            spans = {}
            for row in np.nonzero(diff.any(axis=1))[0]:
                cols = np.nonzero(diff[row])[0]
                spans[int(row)] = (int(cols[0]), int(cols[-1]) + 1)
        for row, (c0, c1) in spans.items():
            col = c0
            while col < c1:
                run = min(IMAGE_CHUNK_PIXELS, c1 - col)
                flat = canvas[row, col:col + run, :].reshape(-1).tolist()
                self.send_api([CMD_DRAW_IMAGE, zone, col, row, run, 1] + bitize7(flat))
                col += run
        self.last_canvas = canvas


# ================================================================ finger_zones
# Finger positions of a hand at rest on the pad, palm down, in units of 'spread'
# about the hand's centre: x across, y up the pad (away from the player).
# This is the right hand; the left is its mirror.
HAND_TEMPLATE = [(-1.0, -0.5), (-0.45, 0.55), (0.0, 0.75), (0.45, 0.6), (0.9, 0.15)]
FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']


class FingerZonesNode(Node):
    """finger_zones - which finger is this touch?

    The erae node numbers fingers in the order they land. This node names them
    instead: it lays out five (or ten) home circles in the natural resting
    positions of a hand, and a touch takes the finger of the nearest free circle
    at the moment it lands. It keeps that finger for as long as it is held, no
    matter where it wanders. A finger that lifts and lands again close to where
    it left, soon enough, gets its old name back (a trill does not turn the
    index finger into the middle one). The circles go out as an image, to draw
    on the pad under the halos.
    """

    @staticmethod
    def factory(name, data, args=None):
        node = FingerZonesNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)
        self.lock = threading.Lock()
        self.centres = np.zeros((0, 2), dtype=np.float32)      # finger -> home position
        self.names = []
        self.assigned = {}          # touch id -> finger index (or -1 while unassigned)
        self.positions = {}         # touch id -> (x, y, z)
        self.lifted = {}            # finger index -> (x, y, time) of its last lift
        self.held_drawn = None      # the set of held fingers the last image showed

        self.touch_in = self.add_input('touch', triggers_execution=True)
        self.size_in = self.add_input('zone size', triggers_execution=True)
        self.draw_in = self.add_input('draw', widget_type='button', callback=self.send_image)
        self.reset_in = self.add_input('reset', widget_type='button', callback=self.reset)

        self.touch_out = self.add_output('touch')
        self.voices_out = self.add_output('voices')
        self.count_out = self.add_output('count')
        self.image_out = self.add_output('zones image')

        self.hands = self.add_option('hands', widget_type='combo', default_value='right', callback=self.layout_changed)
        self.hands.widget.combo_items = ['right', 'left', 'both']
        self.source = self.add_option('source', widget_type='combo', default_value='erae')
        self.source.widget.combo_items = ['erae', 'mpe_in']
        self.width = self.add_option('width', widget_type='input_int', default_value=43, min=1, max=128, callback=self.layout_changed)
        self.height = self.add_option('height', widget_type='input_int', default_value=25, min=1, max=128, callback=self.layout_changed)
        self.centre_x = self.add_option('centre x', widget_type='drag_float', default_value=21.0, min=-128.0, max=256.0, callback=self.layout_changed)
        self.centre_y = self.add_option('centre y', widget_type='drag_float', default_value=12.0, min=-128.0, max=256.0, callback=self.layout_changed)
        self.spread = self.add_option('spread', widget_type='drag_float', default_value=11.0, min=0.1, max=100.0, callback=self.layout_changed)
        self.separation = self.add_option('separation', widget_type='drag_float', default_value=24.0, min=0.0, max=200.0, callback=self.layout_changed)
        self.rotation = self.add_option('rotation', widget_type='drag_float', default_value=0.0, min=-180.0, max=180.0, callback=self.layout_changed)
        self.radius = self.add_option('radius', widget_type='drag_float', default_value=3.0, min=0.5, max=30.0, callback=self.layout_changed)
        self.catch_radius = self.add_option('catch radius', widget_type='drag_float', default_value=0.0, min=0.0, max=200.0)
        self.reacquire_radius = self.add_option('reacquire radius', widget_type='drag_float', default_value=3.0, min=0.0, max=100.0)
        self.reacquire_time = self.add_option('reacquire time', widget_type='drag_float', default_value=0.3, min=0.0, max=10.0)
        self.named = self.add_option('finger as name', widget_type='checkbox', default_value=False)
        self.show_held = self.add_option('show held', widget_type='checkbox', default_value=True)
        self.zone_color = self.add_option('zone color', widget_type='color_picker', default_value=[0.1, 0.3, 0.6, 1.0])
        self.held_color = self.add_option('held color', widget_type='color_picker', default_value=[0.2, 0.7, 1.0, 1.0])
        self.back_color = self.add_option('background', widget_type='color_picker', default_value=[0.0, 0.0, 0.0, 1.0])

    def custom_create(self, from_file):
        self.compute_layout()
        self.send_image()

    # ------------------------------------------------------------- layout
    def compute_layout(self):
        mode = self.hands()
        if mode is None or self.spread() is None or self.centre_x() is None:
            return                      # options not made yet
        spread = float(self.spread())
        angle = np.radians(float(self.rotation()))
        cx, cy = float(self.centre_x()), float(self.centre_y())
        sep = float(self.separation())
        hands = []
        if mode == 'both':
            hands = [('left', cx - sep / 2.0), ('right', cx + sep / 2.0)]
        else:
            hands = [(mode, cx)]
        centres = []
        names = []
        for hand, hx in hands:
            mirror = -1.0 if hand == 'left' else 1.0
            a = angle * mirror                # both hands splay outward by the same angle
            ca, sa = np.cos(a), np.sin(a)
            for (tx, ty), finger in zip(HAND_TEMPLATE, FINGER_NAMES):
                x = tx * mirror * spread
                y = ty * spread
                centres.append((hx + x * ca - y * sa, cy + x * sa + y * ca))
                names.append(finger if mode != 'both' else hand + ' ' + finger)
        with self.lock:
            self.centres = np.array(centres, dtype=np.float32)
            self.names = names
            # fingers that no longer exist lose their touches
            n = len(names)
            for touch, finger in list(self.assigned.items()):
                if finger >= n:
                    self.assigned[touch] = -1
            self.lifted = {f: v for f, v in self.lifted.items() if f < n}
        self.held_drawn = None

    def layout_changed(self):
        self.compute_layout()
        if not getattr(self, 'in_loading_process', False):
            self.send_image()

    def reset(self):
        with self.lock:
            self.assigned.clear()
            self.positions.clear()
            self.lifted.clear()
        self.send_voices()
        self.send_image()

    # ------------------------------------------------------------- touches
    def execute(self):
        if self.active_input is self.size_in:
            size = any_to_numerical_list(self.size_in())
            if size is None or len(size) < 2:
                return
            w, h = (size[1], size[2]) if len(size) >= 3 else (size[0], size[1])
            if w > 0 and h > 0:
                self.width.set(int(w))
                self.height.set(int(h))
                self.layout_changed()
            return
        if self.active_input is not self.touch_in:
            return
        values = self.touch_values(self.touch_in())
        if values is None or len(values) < 6:
            return
        if self.source() == 'mpe_in':
            # voice, state, note, pitch, pressure, slide, velocity
            touch, action = int(values[0]), int(values[1])
            x, y, z = float(values[3]), float(values[5]), float(values[4])
        else:
            # finger, zone, action, x, y, z
            touch, action = int(values[0]), int(values[2])
            x, y, z = float(values[3]), float(values[4]), float(values[5])
        self.handle_touch(touch, action, x, y, z)

    @staticmethod
    def touch_values(data):
        """A touch list as numbers; the action may arrive as the word 'down' 'move' 'up'."""
        if isinstance(data, np.ndarray):
            data = data.tolist()
        if isinstance(data, str):
            data = data.split()
        if not isinstance(data, (list, tuple)):
            return None
        out = []
        for v in data:
            if isinstance(v, str):
                if v in ACTION_NAMES:
                    out.append(ACTION_NAMES.index(v))
                    continue
                try:
                    out.append(float(v))
                except ValueError:
                    return None
            else:
                out.append(v)
        return out

    def handle_touch(self, touch, action, x, y, z):
        held_changed = False
        with self.lock:
            if action == ACTION_DOWN or touch not in self.assigned:
                finger = self.choose_finger(x, y)
                self.assigned[touch] = finger
                held_changed = True
            finger = self.assigned[touch]
            if action == ACTION_UP:
                self.assigned.pop(touch, None)
                self.positions.pop(touch, None)
                if finger >= 0:
                    self.lifted[finger] = (x, y, time.time())
                held_changed = True
            else:
                self.positions[touch] = (x, y, z)
            label = self.names[finger] if (self.named() and 0 <= finger < len(self.names)) else finger
        self.touch_out.send([label, action, x, y, z])
        self.send_voices()
        if held_changed and self.show_held():
            self.send_image()

    def choose_finger(self, x, y):
        """Called with the lock held. The finger this landing belongs to, or -1."""
        n = len(self.names)
        if n == 0:
            return -1
        taken = set(f for f in self.assigned.values() if f >= 0)
        now = time.time()
        # a finger that lifted here a moment ago, and is still free, is the same finger
        r_re = float(self.reacquire_radius())
        t_re = float(self.reacquire_time())
        best, best_d = -1, None
        for finger, (lx, ly, lt) in list(self.lifted.items()):
            if now - lt > t_re:
                self.lifted.pop(finger, None)
                continue
            if finger in taken:
                continue
            d = np.hypot(x - lx, y - ly)
            if d <= r_re and (best_d is None or d < best_d):
                best, best_d = finger, d
        if best >= 0:
            self.lifted.pop(best, None)
            return best
        # otherwise the nearest free home circle, within the catch radius if one is set
        d = np.hypot(self.centres[:, 0] - x, self.centres[:, 1] - y)
        order = np.argsort(d)
        cap = float(self.catch_radius())
        for finger in order:
            finger = int(finger)
            if finger in taken:
                continue
            if cap > 0.0 and d[finger] > cap:
                break
            return finger
        return -1

    def send_voices(self):
        with self.lock:
            n = len(self.names)
            voices = np.zeros((n, 4), dtype=np.float32)
            for touch, finger in self.assigned.items():
                if 0 <= finger < n and touch in self.positions:
                    x, y, z = self.positions[touch]
                    voices[finger] = (1.0, x, y, z)
            count = sum(1 for f in self.assigned.values() if f >= 0)
        self.voices_out.send(voices)
        self.count_out.send(int(count))

    # ------------------------------------------------------------- image
    def held_fingers(self):
        with self.lock:
            return frozenset(f for f in self.assigned.values() if f >= 0)

    def render(self, held):
        width, height = int(self.width()), int(self.height())
        back = np.array(self.back_color()[:3], dtype=np.float32)
        zone = np.array(self.zone_color()[:3], dtype=np.float32)
        hot = np.array(self.held_color()[:3], dtype=np.float32)
        canvas = np.empty((height, width, 3), dtype=np.float32)
        canvas[:] = back
        with self.lock:
            centres = self.centres.copy()
        if len(centres):
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
            yy += PIXEL_CENTRE
            xx += PIXEL_CENTRE
            radius = float(self.radius())
            for finger, (cx, cy) in enumerate(centres):
                d = np.hypot(xx - cx, yy - cy)
                ring = np.clip(1.0 - np.abs(d - radius), 0.0, 1.0)
                colour = hot if finger in held else zone
                k = ring
                if finger in held:
                    k = np.maximum(ring, 0.35 * np.clip(radius - d, 0.0, 1.0))
                canvas = np.maximum(canvas, k[:, :, None] * colour)
        rgb = np.clip(np.rint(canvas * 255.0), 0, 255).astype(np.uint8)
        return rgb[::-1, :, :]          # top row first, like any picture

    def send_image(self):
        held = self.held_fingers() if self.show_held() else frozenset()
        self.held_drawn = held
        self.image_out.send(self.render(held))
