"""Exact trackpad scroll and pinch on macOS, which dpg does not pass on.

dpg's mouse wheel handler reports int(io.MouseWheel) and only when that is
not zero, so a slow two-finger drag - a fraction of a unit a frame - never
arrives at all, and pinch is not reported by dpg in any form. Both reach the
window as Cocoa events on GLFW's content view first, so this hooks that view
through the Objective-C runtime (ctypes only, no pyobjc): scrollWheel: is
wrapped - GLFW still gets every event - and magnifyWithEvent:, which GLFW
leaves to NSResponder, is added.

The hooks run inside dpg's event polling, on the main thread, in the middle
of dpg's frame. They must not call dpg; they only hand the numbers to the
listeners, which queue whatever they do for the main loop.
"""

import ctypes
import ctypes.util
import sys

_installed = False
_scroll_listeners = []
_magnify_listeners = []
_keep = []  # the ctypes callbacks, which must outlive the hooks

# NSEvent modifierFlags
COMMAND = 1 << 20
SHIFT = 1 << 17
OPTION = 1 << 19
CONTROL = 1 << 18

_IMP = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


def add_scroll_listener(fn):
    """fn(dx, dy, precise, flags, momentum): dx, dy in points for a trackpad
    (precise True), in lines for a wheel; momentum is True for the coasting
    after the fingers lift."""
    _scroll_listeners.append(fn)


def add_magnify_listener(fn):
    """fn(magnification, flags): the change in scale since the last event,
    e.g. 0.02 for 2% bigger."""
    _magnify_listeners.append(fn)


def installed():
    return _installed


def _dpg_view_class(objc):
    """dpg's own GLFWContentView. pyGLFW loads a second GLFW (Homebrew's),
    whose class has the same name, and which of the two a lookup by name
    finds is undefined - so it is found by the image it lives in."""
    objc.objc_getClassList.restype = ctypes.c_int
    objc.objc_getClassList.argtypes = [ctypes.c_void_p, ctypes.c_int]
    objc.class_getName.restype = ctypes.c_char_p
    objc.class_getName.argtypes = [ctypes.c_void_p]
    objc.class_getImageName.restype = ctypes.c_char_p
    objc.class_getImageName.argtypes = [ctypes.c_void_p]
    count = objc.objc_getClassList(None, 0)
    classes = (ctypes.c_void_p * count)()
    count = objc.objc_getClassList(classes, count)
    for cls in classes[:count]:
        if objc.class_getName(cls) == b'GLFWContentView' and b'dearpygui' in (objc.class_getImageName(cls) or b''):
            return cls
    return None


def install():
    """Hook the view. True if the hooks are in; False (and nothing changed)
    off macOS, or if the view is not there - call after
    dpg.create_viewport()."""
    global _installed
    if _installed:
        return True
    if sys.platform != 'darwin':
        return False
    try:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))
    except OSError:
        return False
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.class_getInstanceMethod.restype = ctypes.c_void_p
    objc.class_getInstanceMethod.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    objc.method_setImplementation.restype = ctypes.c_void_p
    objc.method_setImplementation.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    objc.class_addMethod.restype = ctypes.c_bool
    objc.class_addMethod.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]

    view = _dpg_view_class(objc)
    if not view:
        return False
    send = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value

    def message(restype):
        return ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p)(send)

    send_double = message(ctypes.c_double)
    send_ulong = message(ctypes.c_ulong)
    send_bool = message(ctypes.c_bool)
    sel = {name: objc.sel_registerName(name) for name in (
        b'scrollingDeltaX', b'scrollingDeltaY', b'hasPreciseScrollingDeltas',
        b'modifierFlags', b'momentumPhase', b'magnification')}

    scroll_method = objc.class_getInstanceMethod(view, objc.sel_registerName(b'scrollWheel:'))
    if not scroll_method:
        return False

    original = [None]

    def scroll_wheel(this, cmd, event):
        try:
            if _scroll_listeners:
                dx = send_double(event, sel[b'scrollingDeltaX'])
                dy = send_double(event, sel[b'scrollingDeltaY'])
                precise = send_bool(event, sel[b'hasPreciseScrollingDeltas'])
                flags = send_ulong(event, sel[b'modifierFlags'])
                momentum = send_ulong(event, sel[b'momentumPhase']) != 0
                for fn in _scroll_listeners:
                    fn(dx, dy, precise, flags, momentum)
        except Exception as e:
            print('mac_gestures scroll:', e)
        original[0](this, cmd, event)

    def magnify(this, cmd, event):
        try:
            amount = send_double(event, sel[b'magnification'])
            flags = send_ulong(event, sel[b'modifierFlags'])
            for fn in _magnify_listeners:
                fn(amount, flags)
        except Exception as e:
            print('mac_gestures magnify:', e)

    scroll_imp = _IMP(scroll_wheel)
    magnify_imp = _IMP(magnify)
    _keep.extend([scroll_imp, magnify_imp])

    # The view does not implement magnifyWithEvent: itself (NSResponder
    # does), so adding it overrides without replacing anyone's.
    objc.class_addMethod(view, objc.sel_registerName(b'magnifyWithEvent:'),
                         ctypes.cast(magnify_imp, ctypes.c_void_p), b'v@:@')
    previous = objc.method_setImplementation(scroll_method, ctypes.cast(scroll_imp, ctypes.c_void_p))
    original[0] = _IMP(previous)
    _installed = True
    return True


class _Rect(ctypes.Structure):
    _fields_ = [('x', ctypes.c_double), ('y', ctypes.c_double),
                ('width', ctypes.c_double), ('height', ctypes.c_double)]


def window_screen_area():
    """The usable area of the screen dpg's window is on - less the menu bar
    and Dock - as (left, top, width, height) in the top-left-origin
    coordinates dpg's viewport position uses, and the height of the window's
    title bar. None off macOS or if the window cannot be found. Main thread."""
    if sys.platform != 'darwin':
        return None
    try:
        import platform
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.class_getImageName.restype = ctypes.c_char_p
        objc.class_getImageName.argtypes = [ctypes.c_void_p]
        send = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        V = ctypes.c_void_p
        send0 = ctypes.CFUNCTYPE(V, V, V)(send)
        send_index = ctypes.CFUNCTYPE(V, V, V, ctypes.c_ulong)(send)
        send_count = ctypes.CFUNCTYPE(ctypes.c_ulong, V, V)(send)
        # A 32-byte struct comes back through objc_msgSend_stret on Intel.
        if platform.machine() == 'x86_64':
            stret = ctypes.cast(objc.objc_msgSend_stret, ctypes.c_void_p).value
            send_rect = ctypes.CFUNCTYPE(_Rect, V, V)(stret)
        else:
            send_rect = ctypes.CFUNCTYPE(_Rect, V, V)(send)
        sel = objc.sel_registerName

        app = send0(objc.objc_getClass(b'NSApplication'), sel(b'sharedApplication'))
        windows = send0(app, sel(b'windows'))
        window = None
        for i in range(send_count(windows, sel(b'count'))):
            candidate = send_index(windows, sel(b'objectAtIndex:'), i)
            view = send0(candidate, sel(b'contentView'))
            if view and b'dearpygui' in (objc.class_getImageName(send0(view, sel(b'class'))) or b''):
                window = candidate
                break
        if window is None:
            return None
        screen = send0(window, sel(b'screen'))
        if not screen:
            return None
        visible = send_rect(screen, sel(b'visibleFrame'))
        # Cocoa measures up from the bottom of the primary screen.
        primary = send_index(send0(objc.objc_getClass(b'NSScreen'), sel(b'screens')), sel(b'objectAtIndex:'), 0)
        primary_height = send_rect(primary, sel(b'frame')).height
        frame = send_rect(window, sel(b'frame'))
        content = send_rect(send0(window, sel(b'contentView')), sel(b'frame'))
        title = max(0.0, frame.height - content.height)
        top = primary_height - (visible.y + visible.height)
        return (visible.x, top, visible.width, visible.height), title
    except Exception as e:
        print('window_screen_area:', e)
        return None
