import subprocess
import platform

from dpg_system.node import Node
from dpg_system.conversion_utils import *


def register_monitor_nodes():
    Node.app.register_node('display_info', DisplayInfoNode.factory)


class DisplayData:
    def __init__(self):
        self.id = -1
        self.connection = ''
        self.offsets = [0, 0]
        self.resolution = [-1, -1]
        self.primary = False


_PLATFORM = platform.system()
_Quartz = None
if _PLATFORM == 'Darwin':
    try:
        import Quartz.CoreGraphics as _Quartz
    except Exception as e:
        print('display_info: Quartz import failed:', e)
        _Quartz = None


def _parse_xrandr_monitor_line(line):
    parts = [p for p in line.split(' ') if p != '']
    if len(parts) < 3:
        return None
    d = DisplayData()
    try:
        d.id = int(parts[0].split(':')[0])
    except ValueError:
        d.id = -1

    connection = parts[1]
    if '+' in connection:
        connection = connection.split('+')[-1]
    if '*' in connection:
        d.primary = True
        connection = connection.split('*')[-1]
    d.connection = connection

    # parts[2] looks like "1920/509x1080/286+0+0" (or without the "/N" physical
    # size sections). Splitting on '+' yields [res, x_offset, y_offset].
    geom_sections = parts[2].split('+')
    if len(geom_sections) >= 3:
        d.offsets = [any_to_int(geom_sections[1]), any_to_int(geom_sections[2])]
    res_split = geom_sections[0].split('x')
    if len(res_split) >= 2:
        d.resolution = [
            any_to_int(res_split[0].split('/')[0]),
            any_to_int(res_split[1].split('/')[0]),
        ]
    return d


def _get_displays_linux():
    try:
        result = subprocess.run(
            ['xrandr', '--listactivemonitors'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except FileNotFoundError:
        print('display_info: xrandr not found on PATH')
        return []
    except Exception as e:
        print('display_info: xrandr invocation failed:', e)
        return []

    if result.returncode != 0:
        print('display_info: xrandr exited with status', result.returncode)
        return []

    displays = []
    # First line is "Monitors: N"; subsequent lines describe each monitor.
    for line in result.stdout.split('\n')[1:]:
        if not line.strip():
            continue
        try:
            d = _parse_xrandr_monitor_line(line)
        except Exception as e:
            print('display_info: failed to parse xrandr line', repr(line), ':', e)
            continue
        if d is not None:
            displays.append(d)
    return displays


def _get_displays_darwin():
    if _Quartz is None:
        return []
    try:
        err, ids, count = _Quartz.CGGetActiveDisplayList(16, None, None)
        if err != 0:
            print('display_info: CGGetActiveDisplayList error', err)
            return []
        main_id = _Quartz.CGMainDisplayID()
        displays = []
        for display_id in list(ids)[:count]:
            d = DisplayData()
            try:
                d.id = int(display_id)
                bounds = _Quartz.CGDisplayBounds(display_id)
                d.offsets = [int(bounds.origin.x), int(bounds.origin.y)]
                d.resolution = [int(bounds.size.width), int(bounds.size.height)]
                d.primary = (display_id == main_id)
            except Exception as e:
                print('display_info: failed to read display', display_id, ':', e)
                continue
            displays.append(d)
        return displays
    except Exception as e:
        print('display_info: Quartz enumeration failed:', e)
        return []


def get_displays():
    if _PLATFORM == 'Linux':
        return _get_displays_linux()
    if _PLATFORM == 'Darwin':
        return _get_displays_darwin()
    return []


class DisplayInfoNode(Node):
    @staticmethod
    def factory(name, data, args=None):
        node = DisplayInfoNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.input = self.add_input('in', widget_type='input_int', default_value=0, min=0, triggers_execution=True)
        self.width_property = self.add_property('width', widget_type='drag_int', default_value=0)
        self.height_property = self.add_property('height', widget_type='drag_int', default_value=0)
        self.x_offset_property = self.add_property('x offset', widget_type='drag_int', default_value=0)
        self.y_offset_property = self.add_property('y offset', widget_type='drag_int', default_value=0)
        self.connection_property = self.add_property('connection', widget_type='text_input', default_value='')
        self.primary_property = self.add_property('primary display', widget_type='checkbox', default_value=False)

        self.output = self.add_output('data out')

    def _set_empty(self):
        self.primary_property.set(False)
        self.connection_property.set('')
        self.width_property.set(0)
        self.height_property.set(0)
        self.x_offset_property.set(0)
        self.y_offset_property.set(0)

    def execute(self):
        try:
            which = any_to_int(self.input())
        except Exception:
            which = 0

        try:
            displays = get_displays()
        except Exception as e:
            print('display_info: get_displays failed:', e)
            displays = []

        if 0 <= which < len(displays):
            d = displays[which]
            self.primary_property.set(d.primary)
            self.connection_property.set(d.connection)
            self.width_property.set(d.resolution[0])
            self.height_property.set(d.resolution[1])
            self.x_offset_property.set(d.offsets[0])
            self.y_offset_property.set(d.offsets[1])
            out_data = [d.resolution[0], d.resolution[1], d.offsets[0], d.offsets[1], d.connection, d.primary]
        else:
            self._set_empty()
            out_data = [0, 0, 0, 0, '', False]
        self.output.send(out_data)

    def custom_create(self, from_file):
        self.execute()
