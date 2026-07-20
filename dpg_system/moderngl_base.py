
import sys
import time
import ctypes
import ctypes.util
import threading
import moderngl
import numpy as np
from dpg_system.matrix_utils import *

# ---------------------------------------------------------------------------
# Pyglet — used on macOS / Windows for context + direct blit.
# On Linux we skip it to avoid X11 RANDR BadRRCrtc crashes:  Pyglet
# calls XRRGetCrtcInfo on CRTC id 0x0 during window creation on systems
# with stale/disconnected outputs, which X11 turns into a fatal error that
# ctypes error-handler tricks cannot intercept reliably (GLFW installs its
# own handler first).  On Linux we use a standalone EGL context instead.
# ---------------------------------------------------------------------------
try:
    import pyglet as _pyglet_mod
    _HAS_PYGLET = not sys.platform.startswith('linux')
except ImportError:
    _pyglet_mod = None
    _HAS_PYGLET = False

if _HAS_PYGLET:
    import pyglet  # noqa: F811 — re-import for the rest of the module


# ---------------------------------------------------------------------------
# MGLNativeWindow — Linux direct-to-screen rendering via a second GLFW window
#
# Strategy:
#   1. Load GLFW functions from DPG's bundled _dearpygui.so (same lib DPG
#      uses, same version, no extra package needed).
#   2. Create a second GLFW window sharing DPG's GL context so both windows
#      see the same GPU objects (textures, FBOs).
#   3. Each frame: make the display window current, blit our FBO texture via
#      a fullscreen quad, swap buffers, restore DPG's context.
#   All GLFW calls happen on the main thread (inside DPG's render loop) so
#   there are no thread-safety issues.
# ---------------------------------------------------------------------------

def _load_glfw_lib():
    """Find and load the GLFW shared library.
    Prefer DPG's bundled copy (same version it was compiled against) so
    the glfwCreateWindow 'share' parameter is ABI-compatible.
    Falls back to any system libglfw.
    Returns a ctypes CDLL or None.
    """
    import os, glob as _glob
    candidates = []
    try:
        import dearpygui as _dpg_mod
        dpg_dir = os.path.dirname(_dpg_mod.__file__)
        # DPG's extension module that embeds GLFW symbols
        candidates += _glob.glob(os.path.join(dpg_dir, '_dearpygui*.so'))
    except Exception:
        pass
    # Also look for standalone libglfw
    for name in ('libglfw.so.3', 'libglfw.so', 'libglfw3.so'):
        try:
            path = ctypes.util.find_library(name) or name
            candidates.append(path)
        except Exception:
            pass
    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
            # Verify glfwCreateWindow is present
            _ = lib.glfwCreateWindow
            return lib
        except Exception:
            continue
    return None


_GLFW_LIB = None   # cached handle, set on first use


def _glfw():
    """Return the cached GLFW ctypes lib, loading it once."""
    global _GLFW_LIB
    if _GLFW_LIB is None and sys.platform.startswith('linux'):
        _GLFW_LIB = _load_glfw_lib()
        if _GLFW_LIB is None:
            print('[MGLNativeWindow] WARNING: could not load GLFW lib. '
                  'Direct-to-screen mode unavailable.')
    return _GLFW_LIB


# ---------------------------------------------------------------------------
# UI event capture — shared helpers for MGLNativeWindow / MGLDisplayWindow.
#
# Both window classes queue events into self.ui_events; MGLContextNode drains
# the queue once per execute() and forwards each event out its 'ui' output.
# Key codes are normalized to match gl_context's ui output (GLFW key codes,
# with the same shift/case mapping), so patches behave identically whether
# they listen to a gl_context or an mgl_context.
# ---------------------------------------------------------------------------

_UI_EVENT_QUEUE_MAX = 256

_GLFW_KEY_ESCAPE = 256

# same shifted-punctuation map gl_context uses
_SHIFTED_KEYS = {
    '1': '!', '2': '@', '3': '#', '4': '$', '5': '%', '6': '^', '7': '&',
    '8': '*', '9': '(', '0': ')', '`': '~', '-': '_', '=': '+', '[': '{',
    ']': '}', '\\': '|', ';': ':', "'": '"', ',': '<', '.': '>', '/': '?',
}


def _char_code_with_shift(char, shift):
    """gl_context's shift/case mapping: return the code of the character
    actually typed, given the unshifted key character (either case)."""
    if shift:
        if 'a' <= char <= 'z':
            return ord(char.upper())
        return ord(_SHIFTED_KEYS.get(char, char))
    if 'A' <= char <= 'Z':
        return ord(char.lower())
    return ord(char)


if _HAS_PYGLET:
    from pyglet.window import key as _pkey, mouse as _pmouse

    # pyglet key symbols → GLFW key codes for the common non-printing keys
    _PYGLET_TO_GLFW_KEY = {
        _pkey.ESCAPE: 256, _pkey.ENTER: 257, _pkey.TAB: 258,
        _pkey.BACKSPACE: 259, _pkey.INSERT: 260, _pkey.DELETE: 261,
        _pkey.RIGHT: 262, _pkey.LEFT: 263, _pkey.DOWN: 264, _pkey.UP: 265,
        _pkey.PAGEUP: 266, _pkey.PAGEDOWN: 267, _pkey.HOME: 268,
        _pkey.END: 269,
    }
    for _i in range(12):
        _PYGLET_TO_GLFW_KEY[getattr(_pkey, 'F%d' % (_i + 1))] = 290 + _i

    # pyglet button constants are a bitmask; GLFW numbers 0=left 1=right 2=middle
    _PYGLET_TO_GLFW_BUTTON = {_pmouse.LEFT: 0, _pmouse.RIGHT: 1, _pmouse.MIDDLE: 2}


class MGLNativeWindow:
    """A second GLFW window that shares DPG's GL context.

    Creates a real OS-level window outside DearPyGui's UI space and blits
    the moderngl FBO texture to it each frame using a fullscreen-quad shader.

    All methods must be called from the main thread (DPG's render loop).
    """

    # GLFW constants
    _GLFW_VISIBLE          = 0x00020004
    _GLFW_CONTEXT_VERSION_MAJOR = 0x00022002
    _GLFW_CONTEXT_VERSION_MINOR = 0x00022003
    _GLFW_OPENGL_PROFILE   = 0x00022008
    _GLFW_OPENGL_CORE_PROFILE = 0x00032001
    _GLFW_DOUBLEBUFFER     = 0x00021010
    _GLFW_RESIZABLE        = 0x00020003

    # callback signatures for UI event capture
    _KEY_CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int)
    _MOUSE_BTN_CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_int)
    _CURSOR_CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p,
                                       ctypes.c_double, ctypes.c_double)
    _SCROLL_CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p,
                                       ctypes.c_double, ctypes.c_double)

    def __init__(self, dpg_glfw_win):
        """
        dpg_glfw_win: raw void* handle of DPG's GLFW window
                      (obtained via glfwGetCurrentContext while DPG is current).
        """
        self._win = None                # our GLFW window handle (void*)
        self._dpg_win = dpg_glfw_win    # DPG's GLFW window handle
        self._display_ctx = None        # moderngl Context wrapping our window
        self._blit_prog = None
        self._blit_vao  = None
        self._visible = False
        self.ui_events = []             # drained by the owning MGLContextNode
        self._buttons_down = set()
        self._last_cursor = (0, 0)

        gfw = _glfw()
        if gfw is None:
            print('[MGLNativeWindow] GLFW not available — direct mode disabled')
            return

        gfw.glfwCreateWindow.restype  = ctypes.c_void_p
        gfw.glfwCreateWindow.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_void_p, ctypes.c_void_p,
        ]

        # Helper: reset GLFW hints so DPG's state isn't polluted
        _GLFW_DEFAULT_HINTS = 0x00065002  # GLFW_ANY_POSITION or ignore
        def _reset_hints():
            try:
                gfw.glfwDefaultWindowHints()
            except Exception:
                pass

        # --- Attempt 1: GL 3.3 Core Profile (ideal) ---
        gfw.glfwWindowHint(self._GLFW_VISIBLE, 0)
        gfw.glfwWindowHint(self._GLFW_RESIZABLE, 1)
        gfw.glfwWindowHint(self._GLFW_DOUBLEBUFFER, 1)
        gfw.glfwWindowHint(self._GLFW_CONTEXT_VERSION_MAJOR, 3)
        gfw.glfwWindowHint(self._GLFW_CONTEXT_VERSION_MINOR, 3)
        gfw.glfwWindowHint(self._GLFW_OPENGL_PROFILE, self._GLFW_OPENGL_CORE_PROFILE)

        self._win = gfw.glfwCreateWindow(
            800, 600, b'MGL Output', None, self._dpg_win)
        _reset_hints()

        # --- Attempt 2: fallback without Core Profile constraint ---
        if not self._win:
            print('[MGLNativeWindow] GL 3.3 Core failed, trying fallback...')
            gfw.glfwWindowHint(self._GLFW_VISIBLE, 0)
            gfw.glfwWindowHint(self._GLFW_RESIZABLE, 1)
            gfw.glfwWindowHint(self._GLFW_DOUBLEBUFFER, 1)
            self._win = gfw.glfwCreateWindow(
                800, 600, b'MGL Output', None, self._dpg_win)
            _reset_hints()

        if not self._win:
            print('[MGLNativeWindow] glfwCreateWindow failed (all attempts)')
            return

        self._install_event_callbacks(gfw)

        # Make our window current to compile blit shader
        gfw.glfwMakeContextCurrent.argtypes = [ctypes.c_void_p]
        gfw.glfwMakeContextCurrent(self._win)

        try:
            self._display_ctx = moderngl.create_context()
            self._blit_prog = self._display_ctx.program(
                vertex_shader='''
                    #version 330
                    in vec2 in_pos;
                    out vec2 v_uv;
                    void main() {
                        gl_Position = vec4(in_pos, 0.0, 1.0);
                        v_uv = in_pos * 0.5 + 0.5;
                    }
                ''',
                fragment_shader='''
                    #version 330
                    uniform sampler2D tex;
                    in vec2 v_uv;
                    out vec4 f_color;
                    void main() {
                        f_color = texture(tex, v_uv);
                    }
                '''
            )
            verts = np.array([-1, -1,  1, -1,  -1, 1,  1, 1], dtype='f4')
            vbo = self._display_ctx.buffer(verts.tobytes())
            self._blit_vao = self._display_ctx.simple_vertex_array(
                self._blit_prog, vbo, 'in_pos')
            print('[MGLNativeWindow] display context and blit shader ready')
        except Exception as e:
            print(f'[MGLNativeWindow] shader init failed: {e}')
            import traceback; traceback.print_exc()

        # Return context to DPG
        gfw.glfwMakeContextCurrent(self._dpg_win)

    def _install_event_callbacks(self, gfw):
        # The CFUNCTYPE wrappers must outlive the window: GLFW keeps only the
        # raw pointer, and a GC'd wrapper means a segfault on the next event.
        self._key_cb = self._KEY_CB_TYPE(self._on_glfw_key)
        self._mouse_btn_cb = self._MOUSE_BTN_CB_TYPE(self._on_glfw_mouse_button)
        self._cursor_cb = self._CURSOR_CB_TYPE(self._on_glfw_cursor)
        self._scroll_cb = self._SCROLL_CB_TYPE(self._on_glfw_scroll)
        try:
            for name, cb_type, cb in (
                    ('glfwSetKeyCallback', self._KEY_CB_TYPE, self._key_cb),
                    ('glfwSetMouseButtonCallback', self._MOUSE_BTN_CB_TYPE, self._mouse_btn_cb),
                    ('glfwSetCursorPosCallback', self._CURSOR_CB_TYPE, self._cursor_cb),
                    ('glfwSetScrollCallback', self._SCROLL_CB_TYPE, self._scroll_cb)):
                fn = getattr(gfw, name)
                fn.argtypes = [ctypes.c_void_p, cb_type]
                fn(self._win, cb)
        except Exception as e:
            print(f'[MGLNativeWindow] event callback setup failed: {e}')

    def _push_ui_event(self, event):
        if len(self.ui_events) < _UI_EVENT_QUEUE_MAX:
            self.ui_events.append(event)

    def _on_glfw_key(self, win, key, scancode, action, mods):
        if action not in (1, 2):  # GLFW_PRESS or GLFW_REPEAT
            return
        if 32 <= key < 127:
            code = _char_code_with_shift(chr(key), bool(mods & 1))
        else:
            code = key
        self._push_ui_event(['key', code])

    def _on_glfw_mouse_button(self, win, button, action, mods):
        x, y = self._last_cursor
        if action == 1:
            self._buttons_down.add(button)
            self._push_ui_event(['mouse_down', x, y, button])
        else:
            self._buttons_down.discard(button)
            self._push_ui_event(['mouse_up', x, y, button])

    def _on_glfw_cursor(self, win, x, y):
        x, y = int(x), int(y)
        self._last_cursor = (x, y)
        if self._buttons_down:
            self._push_ui_event(['mouse_drag', x, y, min(self._buttons_down)])
        else:
            self._push_ui_event(['mouse_move', x, y])

    def _on_glfw_scroll(self, win, dx, dy):
        self._push_ui_event(['scroll', float(dx), float(dy)])

    @property
    def available(self):
        return self._win is not None and self._blit_vao is not None

    def show(self, width, height, title='MGL Output', fullscreen=False):
        if not self._win:
            return
        gfw = _glfw()
        if gfw is None:
            return
        gfw.glfwSetWindowSize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        gfw.glfwSetWindowTitle.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        gfw.glfwShowWindow.argtypes = [ctypes.c_void_p]
        gfw.glfwSetWindowSize(self._win, int(width), int(height))
        gfw.glfwSetWindowTitle(self._win, title.encode() if isinstance(title, str) else title)
        if not self._visible:
            gfw.glfwShowWindow(self._win)
            self._visible = True

    def hide(self):
        if not self._win or not self._visible:
            return
        gfw = _glfw()
        if gfw is None:
            return
        gfw.glfwHideWindow.argtypes = [ctypes.c_void_p]
        gfw.glfwHideWindow(self._win)
        self._visible = False
        self._buttons_down.clear()
        self.ui_events = []

    def set_pos(self, x, y):
        if not self._win:
            return
        gfw = _glfw()
        if gfw is None:
            return
        gfw.glfwSetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        gfw.glfwSetWindowPos(self._win, int(x), int(y))

    def get_pos(self):
        if not self._win or not self._visible:
            return None
        gfw = _glfw()
        if gfw is None:
            return None
        x, y = ctypes.c_int(0), ctypes.c_int(0)
        gfw.glfwGetWindowPos.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        gfw.glfwGetWindowPos(self._win, ctypes.byref(x), ctypes.byref(y))
        return (x.value, y.value)

    def get_size(self):
        if not self._win or not self._visible:
            return None
        gfw = _glfw()
        if gfw is None:
            return None
        w, h = ctypes.c_int(0), ctypes.c_int(0)
        gfw.glfwGetWindowSize.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        gfw.glfwGetWindowSize(self._win, ctypes.byref(w), ctypes.byref(h))
        return (w.value, h.value)

    def blit(self, render_target):
        """Blit render_target.texture to this window.
        Must be called from the main thread while DPG's context is current.
        Temporarily switches to our context, renders, then restores DPG's.
        """
        if not self.available:
            return False

        gfw = _glfw()
        if gfw is None:
            return False

        gfw.glfwMakeContextCurrent.argtypes = [ctypes.c_void_p]
        gfw.glfwGetFramebufferSize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        gfw.glfwSwapBuffers.argtypes = [ctypes.c_void_p]
        gfw.glfwPollEvents.argtypes  = []

        # Switch to display window's context
        gfw.glfwMakeContextCurrent(self._win)

        try:
            w, h = ctypes.c_int(0), ctypes.c_int(0)
            gfw.glfwGetFramebufferSize(self._win, ctypes.byref(w), ctypes.byref(h))

            self._display_ctx.screen.viewport = (0, 0, w.value, h.value)
            self._display_ctx.screen.use()
            self._display_ctx.clear(0.0, 0.0, 0.0, 1.0)
            self._display_ctx.disable(moderngl.DEPTH_TEST)

            # Bind the shared texture via raw GL (the texture was created in
            # DPG's context but is visible here via context sharing).
            _gl = ctypes.CDLL(None)
            _gl.glActiveTexture(0x84C0)                    # GL_TEXTURE0
            _gl.glBindTexture(0x0DE1, render_target.texture.glo)  # GL_TEXTURE_2D
            self._blit_prog['tex'].value = 0

            self._blit_vao.render(moderngl.TRIANGLE_STRIP)
            gfw.glfwSwapBuffers(self._win)
            # NOTE: Do NOT call glfwPollEvents() here. DPG already pumps
            # events for all GLFW windows. Calling it again from inside
            # DPG's render loop causes reentrant dispatch → freeze.
        except Exception as e:
            print(f'[MGLNativeWindow] blit error: {e}')
            import traceback; traceback.print_exc()
        finally:
            # Always restore DPG's context
            gfw.glfwMakeContextCurrent(self._dpg_win)

        return True

    def destroy(self):
        if not self._win:
            return
        gfw = _glfw()
        if gfw:
            try:
                gfw.glfwDestroyWindow.argtypes = [ctypes.c_void_p]
                gfw.glfwDestroyWindow(self._win)
            except Exception as e:
                print(f'[MGLNativeWindow] destroy error: {e}')
        self._win = None
        self._visible = False


class MGLDisplayWindow:
    """Per-node pyglet display window that shares GL resources with the master.

    Counterpart to MGLNativeWindow on Linux, but built on pyglet so it works
    on macOS/Windows where DPG doesn't expose GLFW. Each instance creates its
    own pyglet.window.Window with a GL context that shares textures, buffers,
    and shader programs with the master MGLContext's pyglet window. That way
    multiple mgl_context nodes can each own a separate OS window without
    fighting over the shared master window.
    """

    BLIT_VERTEX_SHADER = '''
        #version 330
        in vec2 in_pos;
        out vec2 v_uv;
        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
            v_uv = in_pos * 0.5 + 0.5;
        }
    '''
    BLIT_FRAGMENT_SHADER = '''
        #version 330
        uniform sampler2D tex;
        in vec2 v_uv;
        out vec4 f_color;
        void main() {
            f_color = texture(tex, v_uv);
        }
    '''

    def __init__(self, master):
        """master: a fully-initialized MGLContext singleton (must have a
        valid pyglet window already)."""
        self._master = master
        self._win = None
        self._display_ctx = None
        self._blit_prog = None
        self._blit_vao = None
        self._visible = False
        self._fullscreen = False
        self.ui_events = []   # drained by the owning MGLContextNode
        self._focused = False

        # Synthetic key repeat: pyglet's Cocoa view drops repeat keyDown
        # events (isARepeat) and nothing else surfaces them, so we re-emit
        # the held key ourselves from poll_repeats(), called by the node
        # each frame. Use the OS repeat timing when we can get it.
        self._held_key_symbol = None
        self._held_key_code = None
        self._repeat_next_time = 0.0
        self._repeat_delay, self._repeat_interval = 0.5, 0.085
        if sys.platform == 'darwin':
            try:
                from pyglet.libs.darwin.cocoapy import ObjCClass as _ObjCClass
                _NSEvent = _ObjCClass('NSEvent')
                delay = float(_NSEvent.keyRepeatDelay())
                interval = float(_NSEvent.keyRepeatInterval())
                if 0.05 <= delay <= 2.0 and 0.01 <= interval <= 1.0:
                    self._repeat_delay, self._repeat_interval = delay, interval
            except Exception:
                pass

        if not _HAS_PYGLET or master._pyglet_window is None:
            return

        try:
            # Build a NEW canvas-matched DisplayConfig each time. We can't
            # reuse the master's config: its pixel_format is released after
            # the first create_context call (see pyglet.gl.cocoa), so calling
            # create_context on it again segfaults. Calling .match() on a
            # fresh template against the master's canvas gives us a clean
            # CocoaDisplayConfig with its own pixel_format.
            master_pg = master._pyglet_window
            template = pyglet.gl.Config(
                double_buffer=True, depth_size=24,
                major_version=3, minor_version=3)
            matched = template.match(master_pg.canvas)
            if not matched:
                raise RuntimeError('no matching GL config for shared context')
            shared_pg_ctx = matched[0].create_context(master_pg.context)
            self._win = pyglet.window.Window(
                width=1, height=1, visible=False, resizable=True,
                context=shared_pg_ctx, caption='MGL Output')

            # Capture UI events from this window. Cocoa/Win32 deliver input
            # straight to the focused window's view even though DPG owns the
            # event pump — BUT with no pyglet EventLoop running, BaseWindow
            # queues every event into _event_queue instead of dispatching it,
            # and nothing ever drains that queue. Disable the queue so events
            # dispatch to our handlers at delivery time (they only append to
            # a Python list, so this is safe inside DPG's pump).
            self._win._enable_event_queue = False
            self._win.push_handlers(
                on_activate=self._on_activate,
                on_deactivate=self._on_deactivate,
                on_key_press=self._on_key_press,
                on_key_release=self._on_key_release,
                on_mouse_press=self._on_mouse_press,
                on_mouse_release=self._on_mouse_release,
                on_mouse_motion=self._on_mouse_motion,
                on_mouse_drag=self._on_mouse_drag,
                on_mouse_scroll=self._on_mouse_scroll,
                on_close=self._on_close,
            )

            # Build the blit shader + VAO on OUR context. Programs/buffers
            # built here live in the shared group; the VAO does not, so we
            # need our own per-window VAO.
            self._win.switch_to()
            self._display_ctx = moderngl.create_context()
            self._blit_prog = self._display_ctx.program(
                vertex_shader=self.BLIT_VERTEX_SHADER,
                fragment_shader=self.BLIT_FRAGMENT_SHADER)
            # Pre-set the sampler uniform once; per-frame writes can trip
            # moderngl's first-call validation against an unbound texture.
            self._blit_prog['tex'].value = 0
            verts = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
            vbo = self._display_ctx.buffer(verts.tobytes())
            self._blit_vao = self._display_ctx.simple_vertex_array(
                self._blit_prog, vbo, 'in_pos')
        except Exception as e:
            print(f'[MGLDisplayWindow] init failed: {e}')
            import traceback
            traceback.print_exc()
            if self._win is not None:
                try:
                    self._win.close()
                except Exception:
                    pass
                self._win = None
        finally:
            # Always restore master context so subsequent moderngl calls
            # against master.ctx land where they expect.
            if master._pyglet_window is not None:
                try:
                    master._pyglet_window.switch_to()
                except Exception:
                    pass

    def _push_ui_event(self, event):
        if len(self.ui_events) < _UI_EVENT_QUEUE_MAX:
            self.ui_events.append(event)

    def _mouse_xy(self, x, y):
        # pyglet's origin is bottom-left; ui output uses top-left like GLFW
        try:
            h = self._win.get_size()[1]
        except Exception:
            return int(x), int(y)
        return int(x), int(h - y)

    def _on_activate(self):
        self._focused = True

    def _on_deactivate(self):
        self._focused = False
        self._held_key_symbol = None
        self._held_key_code = None

    def _on_close(self):
        # Block pyglet's default handler, which would destroy the window out
        # from under the owning node; the node controls window lifetime via
        # its display_mode option.
        return pyglet.event.EVENT_HANDLED

    def _on_key_press(self, symbol, modifiers):
        # key events only reach the front (key) window, so no focus gate;
        # also use them to resync _focused in case on_activate was missed
        self._focused = True
        if 32 <= symbol < 127:
            code = _char_code_with_shift(chr(symbol),
                                         bool(modifiers & _pkey.MOD_SHIFT))
        else:
            code = _PYGLET_TO_GLFW_KEY.get(symbol)
        if code is not None:
            self._push_ui_event(['key', code])
            # arm synthetic repeat; like the OS, only the last pressed
            # key repeats
            self._held_key_symbol = symbol
            self._held_key_code = code
            self._repeat_next_time = time.monotonic() + self._repeat_delay
        if symbol == _pkey.ESCAPE:
            # block pyglet's default ESC-closes-window handler; the node
            # decides what ESC means (e.g. exit fullscreen)
            return pyglet.event.EVENT_HANDLED

    def _on_key_release(self, symbol, modifiers):
        if symbol == self._held_key_symbol:
            self._held_key_symbol = None
            self._held_key_code = None

    def poll_repeats(self):
        """Emit synthetic key-repeat events for the held key. Called once per
        frame by the owning MGLContextNode before it drains ui_events."""
        if self._held_key_code is None or not self._focused:
            return
        now = time.monotonic()
        if now >= self._repeat_next_time:
            self._push_ui_event(['key', self._held_key_code])
            # one repeat per frame; never accumulate a backlog after a stall
            self._repeat_next_time = max(
                self._repeat_next_time + self._repeat_interval,
                now + self._repeat_interval * 0.5)

    def _on_mouse_press(self, x, y, button, modifiers):
        self._focused = True  # clicking the window fronts it
        x, y = self._mouse_xy(x, y)
        self._push_ui_event(['mouse_down', x, y,
                             _PYGLET_TO_GLFW_BUTTON.get(button, 0)])

    def _on_mouse_release(self, x, y, button, modifiers):
        x, y = self._mouse_xy(x, y)
        self._push_ui_event(['mouse_up', x, y,
                             _PYGLET_TO_GLFW_BUTTON.get(button, 0)])

    def _on_mouse_motion(self, x, y, dx, dy):
        if not self._focused:
            return
        x, y = self._mouse_xy(x, y)
        self._push_ui_event(['mouse_move', x, y])

    def _on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if not self._focused:
            return
        x, y = self._mouse_xy(x, y)
        if buttons & _pmouse.RIGHT:
            btn = 1
        elif buttons & _pmouse.MIDDLE:
            btn = 2
        else:
            btn = 0
        self._push_ui_event(['mouse_drag', x, y, btn])

    def _on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if not self._focused:
            return
        self._push_ui_event(['scroll', float(scroll_x), float(scroll_y)])

    @property
    def available(self):
        return self._win is not None and self._blit_vao is not None

    def show(self, width, height, title='MGL Output', fullscreen=False):
        if not self._win:
            return
        try:
            if fullscreen:
                if not self._fullscreen:
                    self._win.set_fullscreen(True)
                    self._fullscreen = True
            else:
                if self._fullscreen:
                    self._win.set_fullscreen(False)
                    self._fullscreen = False
                self._win.set_size(int(width), int(height))
            self._win.set_caption(title if isinstance(title, str) else str(title))
            if not self._visible:
                self._win.set_visible(True)
                self._visible = True
                # Warmup flip: the default framebuffer isn't fully "complete"
                # until the window has presented at least once. Without this,
                # the first real blit logs a GL_INVALID_OPERATION even though
                # the second frame renders fine. Clear+flip burns that frame
                # cleanly at the right size.
                if not getattr(self, '_warmed_up', False):
                    try:
                        self._win.switch_to()
                        self._display_ctx.screen.use()
                        self._display_ctx.clear(0.0, 0.0, 0.0, 1.0)
                        self._win.flip()
                        self._warmed_up = True
                    except Exception as e:
                        # Non-fatal — the next blit will work anyway.
                        print(f'[MGLDisplayWindow] warmup blit skipped: {e}')
                    finally:
                        try:
                            self._master._pyglet_window.switch_to()
                        except Exception:
                            pass
        except Exception as e:
            print(f'[MGLDisplayWindow] show error: {e}')

    def hide(self):
        if not self._win:
            return
        if self._visible:
            try:
                self._win.set_visible(False)
            except Exception as e:
                print(f'[MGLDisplayWindow] hide error: {e}')
            self._visible = False
            self._fullscreen = False
            self._focused = False
            self._held_key_symbol = None
            self._held_key_code = None
            self.ui_events = []

    def destroy(self):
        if self._win is None:
            return
        try:
            self._win.close()
        except Exception as e:
            print(f'[MGLDisplayWindow] destroy error: {e}')
        self._win = None
        self._visible = False

    def set_pos(self, x, y):
        if self._win and self._visible:
            try:
                self._win.set_location(int(x), int(y))
            except Exception:
                pass

    def get_pos(self):
        if self._win and self._visible:
            try:
                return self._win.get_location()
            except Exception:
                return None
        return None

    def get_size(self):
        if self._win and self._visible:
            try:
                return self._win.get_size()
            except Exception:
                return None
        return None

    def blit(self, render_target):
        """Render render_target.texture to this window's backbuffer.
        MSAA resolve must already be done on the master context."""
        if not self.available or not self._visible:
            return False
        master_pg = self._master._pyglet_window
        try:
            # Flush master before switching contexts so the texture write
            # is observable from the display context. Without this, the
            # first blit can get GL_INVALID_OPERATION because the shared
            # texture handle isn't fully populated yet from the display
            # context's perspective.
            from pyglet import gl
            gl.glFlush()

            self._win.switch_to()
            try:
                fb_w, fb_h = self._win.get_framebuffer_size()
            except Exception:
                fb_w, fb_h = self._win.get_size()
            self._display_ctx.screen.viewport = (0, 0, int(fb_w), int(fb_h))
            self._display_ctx.screen.use()
            self._display_ctx.clear(0.0, 0.0, 0.0, 1.0)
            self._display_ctx.disable(moderngl.DEPTH_TEST)

            # Bind the shared scene texture by its raw GL handle — the
            # moderngl Texture object lives in master's context, but the
            # underlying GL texture name is valid in any context in the
            # share group.
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, render_target.texture.glo)

            self._blit_vao.render(moderngl.TRIANGLE_STRIP)
            self._win.flip()
        except Exception as e:
            # Rate-limit: print once per session, otherwise spam the log.
            if not getattr(self, '_blit_warned', False):
                self._blit_warned = True
                print(f'[MGLDisplayWindow] blit error (first occurrence only): {e}')
            return False
        finally:
            # Restore master context for any subsequent moderngl ops the
            # caller may perform (scene render, MSAA resolve, etc.).
            if master_pg is not None:
                try:
                    master_pg.switch_to()
                except Exception:
                    pass
        return True


class NativeGLContextManager:
    """OS-level OpenGL context switcher using raw GLX / WGL / pyglet.

    Linux: uses raw glXMakeCurrent calls with pre-captured handles.
      - On enter: saves DPG's GLX state, switches to standalone context.
      - On exit:  restores DPG's GLX state.
    macOS: delegates to pyglet's set_current().
    Windows: uses wglMakeCurrent.

    mgl_glx_* are the GLX handles captured right after
    moderngl.create_context(standalone=True) while the moderngl context
    was still current. They must be passed from MGLContext.__init__.
    """

    def __init__(self, pyglet_win,
                 mgl_glx_ctx=None, mgl_glx_dpy=None, mgl_glx_draw=None):
        self._win = pyglet_win
        self._platform = sys.platform

        # Pre-captured moderngl standalone GLX handles (Linux only)
        self._mgl_glx_ctx  = mgl_glx_ctx
        self._mgl_glx_dpy  = mgl_glx_dpy
        self._mgl_glx_draw = mgl_glx_draw

        # Saved DPG context for restore
        self._prev_ctx  = None
        self._prev_dpy  = None
        self._prev_draw = None

        # GLX function pointers
        self._glXGetCurrentContext  = None
        self._glXGetCurrentDisplay  = None
        self._glXGetCurrentDrawable = None
        self._glXMakeCurrent        = None

        # WGL function pointers
        self._wglGetCurrentContext = None
        self._wglGetCurrentDC      = None
        self._wglMakeCurrent       = None

        if self._platform.startswith('linux'):
            self._init_glx()
        elif self._platform == 'win32':
            self._init_wgl()

    def _init_glx(self):
        """Load GLX function pointers."""
        for libname in ('libGL.so.1', 'libGL.so', 'libGLX.so.0', 'libGLX.so'):
            try:
                lib = ctypes.CDLL(libname)

                lib.glXGetCurrentContext.restype  = ctypes.c_void_p
                lib.glXGetCurrentContext.argtypes = []

                lib.glXGetCurrentDisplay.restype  = ctypes.c_void_p
                lib.glXGetCurrentDisplay.argtypes = []

                lib.glXGetCurrentDrawable.restype  = ctypes.c_ulong
                lib.glXGetCurrentDrawable.argtypes = []

                lib.glXMakeCurrent.restype  = ctypes.c_bool
                lib.glXMakeCurrent.argtypes = [
                    ctypes.c_void_p,  # Display*
                    ctypes.c_ulong,   # GLXDrawable
                    ctypes.c_void_p,  # GLXContext
                ]

                self._glXGetCurrentContext  = lib.glXGetCurrentContext
                self._glXGetCurrentDisplay  = lib.glXGetCurrentDisplay
                self._glXGetCurrentDrawable = lib.glXGetCurrentDrawable
                self._glXMakeCurrent        = lib.glXMakeCurrent
                break
            except Exception:
                continue

    def _init_wgl(self):
        """Load WGL function pointers (Windows)."""
        try:
            lib = ctypes.windll.opengl32
            lib.wglGetCurrentContext.restype  = ctypes.c_void_p
            lib.wglGetCurrentContext.argtypes = []
            lib.wglGetCurrentDC.restype       = ctypes.c_void_p
            lib.wglGetCurrentDC.argtypes      = []
            lib.wglMakeCurrent.restype        = ctypes.c_bool
            lib.wglMakeCurrent.argtypes       = [ctypes.c_void_p, ctypes.c_void_p]
            self._wglGetCurrentContext = lib.wglGetCurrentContext
            self._wglGetCurrentDC      = lib.wglGetCurrentDC
            self._wglMakeCurrent       = lib.wglMakeCurrent
        except Exception:
            pass

    def __enter__(self):
        if self._platform.startswith('linux'):
            # Save DPG's current GLX handles
            if self._glXGetCurrentContext:
                self._prev_ctx  = self._glXGetCurrentContext()
                self._prev_dpy  = self._glXGetCurrentDisplay()
                self._prev_draw = self._glXGetCurrentDrawable()

            # Switch to the standalone moderngl GLX context
            if self._mgl_glx_ctx and self._mgl_glx_dpy and self._glXMakeCurrent:
                ok = self._glXMakeCurrent(
                    self._mgl_glx_dpy,
                    self._mgl_glx_draw if self._mgl_glx_draw else 0,
                    self._mgl_glx_ctx,
                )
                if not ok:
                    print('[NativeGLContextManager] glXMakeCurrent to mgl context failed')
            else:
                print('[NativeGLContextManager] WARNING: no GLX handles for mgl context; '
                      'GL calls may crash. ctx=%s dpy=%s' % (self._mgl_glx_ctx, self._mgl_glx_dpy))

        elif self._platform == 'win32':
            if self._wglGetCurrentContext:
                self._prev_ctx = self._wglGetCurrentContext()
                self._prev_dpy = self._wglGetCurrentDC()
            if self._win is not None:
                try:
                    self._win.context.set_current()
                except Exception:
                    pass

        else:
            # macOS: pyglet handles context switching
            if self._win is not None:
                try:
                    self._win.context.set_current()
                except Exception:
                    pass

        return self

    def __exit__(self, *args):
        if self._platform.startswith('linux'):
            # Restore DPG's GLX context
            if self._glXMakeCurrent and self._prev_dpy:
                ok = self._glXMakeCurrent(
                    self._prev_dpy,
                    self._prev_draw if self._prev_draw else 0,
                    self._prev_ctx,
                )
                if not ok:
                    print('[NativeGLContextManager] glXMakeCurrent restore to DPG failed')

        elif self._platform == 'win32':
            if self._wglMakeCurrent:
                self._wglMakeCurrent(self._prev_dpy, self._prev_ctx)
        # macOS: pyglet handles restore







class MGLRenderTarget:
    def __init__(self, ctx, width, height, samples=4):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.samples = samples
        self.fbo = None
        self.texture = None
        self.depth_texture = None
        self.msaa_fbo = None
        self.msaa_texture = None
        self.msaa_depth_texture = None
        self.create()

    def release(self):
        # During context loss or process shutdown, individual release() calls
        # can raise. Don't let one failure abort the rest and leak the others.
        for attr in ('texture', 'depth_texture', 'fbo',
                     'msaa_texture', 'msaa_depth_texture', 'msaa_fbo'):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
            setattr(self, attr, None)

    def create(self):
        # 1. Standard Texture & FBO (Resolve Target)
        self.texture = self.ctx.texture((self.width, self.height), 4)
        self.depth_texture = self.ctx.depth_texture((self.width, self.height))
        self.fbo = self.ctx.framebuffer(self.texture, self.depth_texture)

        # 2. MSAA Target
        if self.samples > 0:
            valid_options = [16, 8, 6, 4, 2]
            candidates = [s for s in valid_options if s <= self.samples]
            if self.samples not in valid_options:
                candidates.insert(0, self.samples)
            
            success = False
            for s in candidates:
                try:
                    self.msaa_texture = self.ctx.texture((self.width, self.height), 4, samples=s)
                    self.msaa_depth_texture = self.ctx.depth_texture((self.width, self.height), samples=s)
                    self.msaa_fbo = self.ctx.framebuffer(self.msaa_texture, self.msaa_depth_texture)
                    self.samples = s
                    success = True
                    break
                except Exception as e:
                    # Release partials — any of these can also throw if the
                    # context refused MSAA at this sample count; swallow them.
                    for attr in ('msaa_texture', 'msaa_depth_texture', 'msaa_fbo'):
                        obj = getattr(self, attr, None)
                        if obj is not None:
                            try:
                                obj.release()
                            except Exception:
                                pass
                            setattr(self, attr, None)
            
            if not success:
                print("MGLRenderTarget: MSAA failed, falling back to 0 samples")
                self.samples = 0
                self.msaa_fbo = None
        else:
            self.msaa_fbo = None

    def use(self):
        target = self.msaa_fbo if (self.samples > 0 and self.msaa_fbo) else self.fbo
        if target:
            target.use()

    def clear(self, r=0.0, g=0.0, b=0.0, a=1.0):
        target = self.msaa_fbo if (self.samples > 0 and self.msaa_fbo) else self.fbo
        if target:
            target.clear(r, g, b, a)

    def get_pixel_data(self):
        # Resolve MSAA if needed
        try:
            if self.samples > 0 and self.msaa_fbo:
                self.ctx.copy_framebuffer(self.fbo, self.msaa_fbo)
            if self.fbo:
                return self.fbo.read(components=4)
        except Exception as e:
            print(f'MGLRenderTarget.get_pixel_data error: {e}')
        return None


class MGLContext:
    _instance = None

    @staticmethod
    def get_instance():
        if MGLContext._instance is None:
            MGLContext._instance = MGLContext()
        return MGLContext._instance

    def __init__(self):
        if MGLContext._instance is not None:
            raise RuntimeError("MGLContext is a singleton; use MGLContext.get_instance()")
        
        # On Linux we defer ALL GL context creation to _ensure_initialized()
        # which is called from __enter__ when DPG's GLFW context is guaranteed
        # to be current.  On macOS/Windows we create a Pyglet window now.
        self.ctx = None
        self._gl_initialized = False
        self._pyglet_window = None   # set properly on macOS/Windows by _HAS_PYGLET path

        if _HAS_PYGLET:
            self._try_create_pyglet_context()
        else:
            # Linux: ctx will be created lazily in _ensure_initialized().
            # All GL-requiring init is deferred.
            pass

        # Default Render State (non-GL, always safe)
        init_width = 1280
        init_height = 720
        init_samples = 4

        self.current_color = (1.0, 1.0, 1.0, 1.0)
        self.lights = []
        self.current_material = {
            'ambient': [0.1, 0.1, 0.1, 1.0],
            'diffuse': [1.0, 1.0, 1.0, 1.0],
            'specular': [0.5, 0.5, 0.5, 1.0],
            'shininess': 32.0
        }
        self.model_matrix_stack = [np.identity(4, dtype=np.float32)]
        self.view_matrix = np.identity(4, dtype=np.float32)
        self.projection_matrix = np.identity(4, dtype=np.float32)

        # GL-requiring state — set to None/default so Linux deferred path is safe
        self.default_target = None
        self.active_target   = None
        self.default_shader  = None
        self.point_shader    = None
        self.current_shader  = None

        # GL objects queued for release at the next render block. Node deletion
        # runs from DPG handler callbacks (worker thread, no GL context current),
        # so releasing there segfaults — nodes must defer_release() instead.
        self._pending_releases = []
        self._pending_releases_lock = threading.Lock()

        if self._gl_initialized:
            # macOS/Windows: GL context exists, do full init now.
            self._gl_init(init_width, init_height, init_samples)

    def _gl_init(self, width=1280, height=720, samples=4):
        """GL setup that requires self.ctx to be current. Safe to call lazily."""
        # Render Target Management
        self.default_target = MGLRenderTarget(self.ctx, width, height, samples)
        self.active_target = self.default_target

        # Default smooth shader
        self.default_shader = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 M;
                uniform mat4 V;
                uniform mat4 P;
                uniform float point_size;
                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord;
                out vec3 v_normal;
                out vec3 v_pos;
                out vec2 v_texcoord;
                void main() {
                    vec4 world_pos = M * vec4(in_position, 1.0);
                    gl_Position = P * V * world_pos;
                    float perspective_size = point_size / gl_Position.w;
                    if (point_size > 1.5) {
                        gl_PointSize = perspective_size + 2.0;
                    } else {
                        gl_PointSize = max(1.0, perspective_size);
                    }
                    mat3 normal_matrix = transpose(inverse(mat3(M)));
                    v_normal = normal_matrix * in_normal;
                    v_pos = world_pos.xyz;
                    v_texcoord = in_texcoord;
                }
            ''',
            fragment_shader='''
                #version 330
                #define MAX_LIGHTS 8
                
                uniform vec4 color;
                uniform vec3 view_pos;
                uniform float point_size;
                uniform bool point_culling;
                uniform bool round_points;
                
                // Texturing
                uniform sampler2D diffuse_map;
                uniform bool has_texture;
                in vec2 v_texcoord;

                // Lighting Uniforms
                struct Light {
                    vec3 pos;
                    vec3 ambient;
                    vec3 diffuse;
                    vec3 specular;
                    float intensity;
                };
                
                uniform int num_lights;
                uniform Light lights[MAX_LIGHTS];
                
                // Material Uniforms
                uniform vec4 material_ambient;
                uniform vec4 material_diffuse;
                uniform vec4 material_specular;
                uniform float material_shininess;
                
                in vec3 v_normal;
                in vec3 v_pos;
                out vec4 f_color;
                
                void main() {
                    // Manual Culling for Points
                    if (point_culling) {
                        vec3 n = normalize(v_normal);
                        vec3 v = normalize(view_pos - v_pos);
                        if (dot(n, v) < 0.0) discard;
                    }

                    vec4 base_color = color;
                    float alpha = base_color.a * material_diffuse.a * material_ambient.a;

                    // Apply Texture
                    if (has_texture) {
                        vec4 texColor = texture(diffuse_map, v_texcoord);
                        base_color *= texColor;
                        alpha *= texColor.a;
                    }
                    
                    // Round Points
                    if (round_points && point_size > 1.5) {
                        vec2 cxy = 2.0 * gl_PointCoord - 1.0;
                        float dist = length(cxy);
                        float limit = point_size / (point_size + 2.0);
                        if (dist > limit) discard;
                    }
                    
                    vec3 norm = normalize(v_normal);
                    vec3 viewDir = normalize(view_pos - v_pos);
                    vec3 final_rgb = vec3(0.0);
                    
                    // Default Ambient if no lights
                    if (num_lights == 0) {
                        final_rgb = vec3(0.2) * base_color.rgb; // Simple fallback
                    } else {
                        vec3 total_emission = vec3(0.0); // If we had emission map
                        vec3 total_ambient = vec3(0.0);
                        vec3 total_diffuse = vec3(0.0);
                        vec3 total_specular = vec3(0.0);
                        
                        for (int i = 0; i < MAX_LIGHTS; i++) {
                            if (i >= num_lights) break;
                            
                            // Ambient
                            total_ambient += lights[i].ambient * material_ambient.rgb;
                            
                            // Diffuse
                            vec3 lightDir = normalize(lights[i].pos - v_pos);
                            float diff = max(dot(norm, lightDir), 0.0); // One-sided
                            total_diffuse += diff * lights[i].diffuse * material_diffuse.rgb * lights[i].intensity;
                            
                            // Specular (Blinn-Phong)
                            vec3 halfwayDir = normalize(lightDir + viewDir);
                            float spec = pow(max(dot(norm, halfwayDir), 0.0), material_shininess);
                            total_specular += spec * lights[i].specular * material_specular.rgb * lights[i].intensity; 
                        }
                        
                        // Premultiplied Alpha Output
                        // Body Color = (Ambient + Diffuse) * Alpha
                        // Specular = Specular (Additive, not multiplied by alpha)
                        final_rgb = (total_ambient + total_diffuse) * base_color.rgb * alpha + total_specular;
                    }
                    
                    f_color = vec4(final_rgb, alpha);
                }
            '''
        )


        # Instanced Sphere Shader (Simple)
        self.mgl_instanced_joint_shader = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 VP;
                #define MAX_BONES 50
                uniform mat4 bones[MAX_BONES];
                
                in vec3 in_pos;
                in vec3 in_norm;
                in float in_radius;     // Per-instance radius
                in int in_bone_idx;     // Per-instance bone index
                
                out vec3 v_pos;
                out vec3 v_norm;
                
                void main() {
                    mat4 bone_mat = bones[in_bone_idx];
                    
                    // Scale local sphere by radius
                    vec3 local_pos = in_pos * in_radius;
                    
                    // Transform to world space via bone matrix
                    // Note: bone_mat includes rotation and translation
                    vec4 world_pos = bone_mat * vec4(local_pos, 1.0);
                    
                    // Rotate normal
                    mat3 normal_matrix = mat3(bone_mat);
                    v_norm = normal_matrix * in_norm; 
                    v_pos = world_pos.xyz;
                    
                    gl_Position = VP * world_pos;
                }
            ''',
            fragment_shader='''
                #version 330
                
                uniform vec3 view_pos;
                uniform vec4 color;
                
                in vec3 v_pos;
                in vec3 v_norm;
                
                out vec4 f_color;
                
                void main() {
                    // Unlit / Flat Color
                    f_color = color;
                }
            '''
        )

    def _try_create_pyglet_context(self):
        """Attempt to create the pyglet window + moderngl context (macOS/Windows).
        Safe to call multiple times; does nothing once successful."""
        if self._gl_initialized:
            return
        # Cocoa (and pyglet's GIL handling) require window creation on the main
        # thread. DPG item/global handler callbacks run on its internal worker
        # thread even with manual_callback_management, and creating the window
        # there corrupts that thread's Python thread-state and aborts the
        # process. Defer: __enter__/ensure_ready retry from the main loop.
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            config = pyglet.gl.Config(
                double_buffer=True, depth_size=24,
                major_version=3, minor_version=3)
            self._pyglet_window = pyglet.window.Window(
                width=1, height=1, visible=False, resizable=True, config=config,
                caption='MGL Output')
            self.ctx = moderngl.create_context()  # wraps pyglet's active context
            self.ctx.enable(moderngl.BLEND)
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            self._gl_initialized = True
        except Exception as e:
            # Retry next frame — pyglet's signal-handler setup can fail when first
            # touched from a non-main thread, then succeed on a subsequent call.
            # Print only the first error per session to avoid spamming the log.
            if not getattr(self, '_pyglet_init_warned', False):
                self._pyglet_init_warned = True
                print(f"Failed to create Pyglet/moderngl context: {e} (will keep retrying silently)")

    def _ensure_initialized(self):
        """On Linux: lazily wrap DPG's GLFW context with moderngl.
        Called from __enter__ when DPG's context is guaranteed current.
        On macOS/Windows this is a no-op (context was created in __init__).
        """
        if self._gl_initialized:
            return
        try:
            # DPG's GLFW context is current right now — wrap it.
            # No separate context, no context switching, no X11 windows.
            self.ctx = moderngl.create_context()
            self.ctx.enable(moderngl.BLEND)
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            self._gl_init()
            self._gl_initialized = True
            print('[MGLContext] GL context initialized (wrapping DPG GLFW context)')

            # Capture DPG's GLFW window handle for per-node native windows.
            self._dpg_glfw_win = None
            gfw = _glfw()
            if gfw is not None:
                gfw.glfwGetCurrentContext.restype = ctypes.c_void_p
                self._dpg_glfw_win = gfw.glfwGetCurrentContext()
                if self._dpg_glfw_win:
                    print(f'[MGLContext] DPG GLFW handle captured: {self._dpg_glfw_win}')
                else:
                    print('[MGLContext] WARNING: glfwGetCurrentContext returned NULL')
        except Exception as e:
            print(f'[MGLContext] _ensure_initialized failed: {e}')
            import traceback; traceback.print_exc()

    def ensure_ready(self):
        """Return True once the GL context exists, creating it if possible.
        On macOS/Windows creation is deferred when first requested off the
        main thread (Cocoa constraint); callers should skip GL work until
        this returns True — the main loop's next frame task completes it."""
        if sys.platform.startswith('linux'):
            # Lazy init happens inside __enter__ while DPG's context is current.
            return True
        if not self._gl_initialized:
            self._try_create_pyglet_context()
            if self._gl_initialized:
                self._gl_init()
        return self._gl_initialized

    # ------------------------------------------------------------------ #
    #  Context manager protocol                                           #
    #  Usage: with self.context:                                         #
    #  On Linux: no context switching — we share DPG's GLFW context.    #
    #  On macOS/Windows: uses NativeGLContextManager + pyglet.          #
    # ------------------------------------------------------------------ #
    def defer_release(self, *objs):
        """Queue moderngl objects for release at the start of the next render
        block, when the GL context is guaranteed current on the render thread."""
        with self._pending_releases_lock:
            for obj in objs:
                if obj is not None:
                    self._pending_releases.append(obj)

    def _drain_pending_releases(self):
        if not self._pending_releases:
            return
        with self._pending_releases_lock:
            pending, self._pending_releases = self._pending_releases, []
        for obj in pending:
            try:
                obj.release()
            except Exception:
                pass

    def __enter__(self):
        if sys.platform.startswith('linux'):
            # Lazy-init: wrap DPG's current context if we haven't yet.
            self._ensure_initialized()
            # Save the GL state that moderngl may disturb, so we can
            # restore it for DPG after our render block.
            self._save_gl_state()
        else:
            # Retry pyglet/moderngl context creation if it failed during __init__
            if not self._gl_initialized:
                self._try_create_pyglet_context()
                if self._gl_initialized:
                    self._gl_init()
            self._native_cm = NativeGLContextManager(self._pyglet_window)
            self._native_cm.__enter__()
        if self.ctx is not None:
            self._drain_pending_releases()
        return self

    def __exit__(self, *args):
        if sys.platform.startswith('linux'):
            # Restore the GL state DPG expects.
            self._restore_gl_state()
        else:
            self._native_cm.__exit__(*args)
            self._native_cm = None

    def _save_gl_state(self):
        """Save GL state before our moderngl block (Linux shared-context path)."""
        import ctypes as _ct
        _int = _ct.c_int
        # Current draw framebuffer
        fbo = _int(0)
        _ct.CDLL(None).glGetIntegerv(0x8CA6, _ct.byref(fbo))   # GL_DRAW_FRAMEBUFFER_BINDING
        self._saved_fbo = fbo.value
        # Viewport
        vp = (_int * 4)(0, 0, 0, 0)
        _ct.CDLL(None).glGetIntegerv(0x0BA2, vp)               # GL_VIEWPORT
        self._saved_viewport = (vp[0], vp[1], vp[2], vp[3])
        # Blend enabled
        blend = _int(0)
        _ct.CDLL(None).glGetIntegerv(0x0BE2, _ct.byref(blend)) # GL_BLEND
        self._saved_blend = bool(blend.value)
        # Depth test enabled
        depth = _int(0)
        _ct.CDLL(None).glGetIntegerv(0x0B71, _ct.byref(depth)) # GL_DEPTH_TEST
        self._saved_depth = bool(depth.value)
        # Blend src/dst (RGB only, good enough)
        bsrc = _int(0); bdst = _int(0)
        _ct.CDLL(None).glGetIntegerv(0x0BE1, _ct.byref(bsrc))  # GL_BLEND_SRC
        _ct.CDLL(None).glGetIntegerv(0x0BE0, _ct.byref(bdst))  # GL_BLEND_DST
        self._saved_blend_src = bsrc.value
        self._saved_blend_dst = bdst.value

    def _restore_gl_state(self):
        """Restore GL state after our moderngl block (Linux shared-context path)."""
        if self.ctx is None:
            return
        import ctypes as _ct
        _lib = _ct.CDLL(None)
        # Restore framebuffer
        _lib.glBindFramebuffer(0x8D40, self._saved_fbo)         # GL_FRAMEBUFFER
        # Restore viewport
        vp = self._saved_viewport
        _lib.glViewport(vp[0], vp[1], vp[2], vp[3])
        # Restore depth test
        if self._saved_depth:
            _lib.glEnable(0x0B71)                               # GL_DEPTH_TEST
        else:
            _lib.glDisable(0x0B71)
        # Restore blend
        if self._saved_blend:
            _lib.glEnable(0x0BE2)                               # GL_BLEND
        else:
            _lib.glDisable(0x0BE2)
        # Restore blend func
        _lib.glBlendFunc(self._saved_blend_src, self._saved_blend_dst)


    # Properties for backward compatibility (if needed) and convenience
    @property
    def fbo(self): return self.active_target.fbo
    @property
    def texture(self): return self.active_target.texture
    @property
    def width(self): return self.active_target.width
    @width.setter
    def width(self, value): pass # Read-only strictly speaking from external view
    @property
    def height(self): return self.active_target.height
    @height.setter
    def height(self, value): pass
    @property
    def samples(self): return self.active_target.samples
    @samples.setter
    def samples(self, value): pass

    def create_render_target(self, width, height, samples=4):
        if self.ctx is None:
            raise RuntimeError(
                "MGLContext.ctx is None — GL context was never created. "
                "Ensure rendering happens inside 'with self.context:' block.")
        return MGLRenderTarget(self.ctx, width, height, samples)

    def use_render_target(self, target):
        self.active_target = target
        target.use()

    # Legacy update_framebuffer for simple singleton usage
    def update_framebuffer(self, width, height, samples=None):
        if self.ctx is None or self.active_target is None:
            return
        if samples is None: samples = self.active_target.samples
        # Check if active target matches
        t = self.active_target
        if t.width != width or t.height != height or t.samples != samples:
            # Recreate active target (if it's the default one)
            if t == self.default_target:
                t.release()
                self.default_target = MGLRenderTarget(self.ctx, width, height, samples)
                self.active_target = self.default_target
            else:
                # If using a custom target, we don't automatically update it here usually?
                # Or we assume single-context mode.
                pass

    def clear(self, r=0.0, g=0.0, b=0.0, a=1.0):
        if self.active_target is None or self.ctx is None:
            return
        self.active_target.clear(r, g, b, a)
        # Reset stacks
        self.model_matrix_stack = [np.identity(4, dtype=np.float32)]
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)

    def get_pixel_data(self):
        if self.active_target is None:
            return None
        return self.active_target.get_pixel_data()

    def update_lights(self, prog):
        # Set num_lights
        if 'num_lights' in prog:
            prog['num_lights'].value = len(self.lights)
        
        for i, light in enumerate(self.lights):
            if i >= 8: break # Shield against overflow
            
            # Position
            name = f'lights[{i}].pos'
            if name in prog:
                prog[name].value = tuple(light['pos'])
            
            # Ambient
            name = f'lights[{i}].ambient'
            if name in prog:
                prog[name].value = tuple(light['ambient'])
                
            # Diffuse
            name = f'lights[{i}].diffuse'
            if name in prog:
                prog[name].value = tuple(light['diffuse'])
                
            # Specular
            name = f'lights[{i}].specular'
            if name in prog:
                prog[name].value = tuple(light['specular'])
                
            # Intensity
            name = f'lights[{i}].intensity'
            if name in prog:
                prog[name].value = light['intensity']

    def update_material(self, prog):
        def _pad4(c):
            c = tuple(c)
            if len(c) == 3:
                return c + (1.0,)
            return c[:4]

        if 'material_ambient' in prog:
            prog['material_ambient'].value = _pad4(self.current_material['ambient'])
        if 'material_diffuse' in prog:
            prog['material_diffuse'].value = _pad4(self.current_material['diffuse'])
        if 'material_specular' in prog:
            prog['material_specular'].value = _pad4(self.current_material['specular'])
        if 'material_shininess' in prog:
            prog['material_shininess'].value = self.current_material['shininess']

    # Matrix Stack Operations
    def push_matrix(self):
        if self.model_matrix_stack:
            self.model_matrix_stack.append(self.model_matrix_stack[-1].copy())
        else:
            self.model_matrix_stack.append(np.identity(4, dtype=np.float32))

    def pop_matrix(self):
        if len(self.model_matrix_stack) > 1:
            self.model_matrix_stack.pop()
    
    def multiply_matrix(self, matrix):
        if self.model_matrix_stack:
            # Post-multiplication to match legacy OpenGL: M_new = M_old * T_new
            # Last transform in the chain is applied first to vertices, so
            # translate->rotate means object rotates locally at translated position.
            # M_gl_new = M_gl_old @ T_gl  => M_numpy_new = T_numpy @ M_numpy_old
            self.model_matrix_stack[-1] = np.dot(self.model_matrix_stack[-1], matrix)
            
    def get_model_matrix(self):
        if self.model_matrix_stack:
            return self.model_matrix_stack[-1]
        return np.identity(4, dtype=np.float32)

    def set_view_matrix(self, matrix):
        self.view_matrix = matrix

    def set_projection_matrix(self, matrix):
        self.projection_matrix = matrix

    # --- Window Management (Pyglet on macOS/Windows, MGLNativeWindow on Linux) ---
    @property
    def has_direct_window(self):
        if sys.platform.startswith('linux'):
            # Native windows are per-node; we just check if the DPG handle
            # is available (meaning a native window CAN be created).
            return getattr(self, '_dpg_glfw_win', None) is not None
        return self._pyglet_window is not None

    def show_window(self, width, height, title='MGL Output', fullscreen=False):
        """Show the display window. On Linux uses MGLNativeWindow; on macOS/Windows uses pyglet."""
        if sys.platform.startswith('linux'):
            nw = getattr(self, '_native_win', None)
            if nw:
                nw.show(width, height, title=title, fullscreen=fullscreen)
            return

        # macOS / Windows: pyglet path
        if not self._pyglet_window:
            return
        win = self._pyglet_window

        if not hasattr(self, '_win_visible'):
            self._win_visible = False
            self._win_fullscreen = False
            self._win_width = 0
            self._win_height = 0

        needs_show = not self._win_visible

        if fullscreen:
            if not self._win_fullscreen:
                try:
                    screen = win.display.get_default_screen()
                    scr_w, scr_h = screen.width, screen.height
                except Exception:
                    scr_w, scr_h = width, height
                win.set_size(scr_w, scr_h)
                win.set_location(0, 0)
                self._win_fullscreen = True
                self._win_width = scr_w
                self._win_height = scr_h
                needs_show = True
        else:
            if self._win_fullscreen:
                self._win_fullscreen = False
                needs_show = True
            if self._win_width != width or self._win_height != height:
                win.set_size(width, height)
                self._win_width = width
                self._win_height = height

        if needs_show:
            win.set_visible(True)
            self._win_visible = True

    def hide_window(self):
        """Hide the display window."""
        if sys.platform.startswith('linux'):
            nw = getattr(self, '_native_win', None)
            if nw:
                nw.hide()
            return
        if self._pyglet_window and getattr(self, '_win_visible', False):
            self._pyglet_window.set_visible(False)
            self._win_visible = False
            self._win_fullscreen = False

    def set_window_pos(self, x, y):
        """Move the display window."""
        if sys.platform.startswith('linux'):
            nw = getattr(self, '_native_win', None)
            if nw:
                nw.set_pos(x, y)
            return
        if self._pyglet_window and getattr(self, '_win_visible', False):
            self._pyglet_window.set_location(int(x), int(y))

    def get_window_pos(self):
        """Get display window position from the OS."""
        if sys.platform.startswith('linux'):
            nw = getattr(self, '_native_win', None)
            return nw.get_pos() if nw else None
        if self._pyglet_window and getattr(self, '_win_visible', False):
            return self._pyglet_window.get_location()
        return None

    def get_window_size(self):
        """Get the actual display window size from the OS, or None if not shown."""
        if sys.platform.startswith('linux'):
            nw = getattr(self, '_native_win', None)
            if nw and getattr(nw, '_visible', False):
                try:
                    return nw.get_size()
                except AttributeError:
                    return None
            return None
        if self._pyglet_window and getattr(self, '_win_visible', False):
            try:
                return self._pyglet_window.get_size()
            except Exception:
                return (self._pyglet_window.width, self._pyglet_window.height)
        return None

    @property
    def screen_size(self):
        """Get the screen dimensions."""
        if sys.platform.startswith('linux'):
            # Use a safe default for Linux (avoid RANDR calls)
            return 1920, 1080
        if self._pyglet_window:
            try:
                screen = self._pyglet_window.display.get_default_screen()
                return screen.width, screen.height
            except Exception:
                return 1920, 1080
        return 1920, 1080

    def resize_window(self, width, height):
        """Resize the display window."""
        if sys.platform.startswith('linux'):
            nw = getattr(self, '_native_win', None)
            if nw and nw._visible:
                nw.show(width, height)
            return
        if self._pyglet_window and self._pyglet_window.visible:
            self._pyglet_window.set_size(width, height)

    def blit_to_window(self, render_target):
        """Blit resolved FBO to the display window — zero CPU readback.
        On Linux: delegates to MGLNativeWindow.blit() which handles context switching.
        On macOS/Windows: renders via pyglet fullscreen quad.
        """
        if sys.platform.startswith('linux'):
            nw = getattr(self, '_native_win', None)
            if nw and nw.available:
                # MSAA resolve before blit
                if render_target.samples > 0 and render_target.msaa_fbo:
                    self.ctx.copy_framebuffer(render_target.fbo, render_target.msaa_fbo)
                return nw.blit(render_target)
            return False

        # macOS / Windows: pyglet fullscreen-quad blit
        if not self._pyglet_window:
            return False

        try:
            # Resolve MSAA if needed
            if render_target.samples > 0 and render_target.msaa_fbo:
                self.ctx.copy_framebuffer(render_target.fbo, render_target.msaa_fbo)

            # Lazily create blit shader + fullscreen quad VAO
            if not hasattr(self, '_blit_prog') or self._blit_prog is None:
                self._blit_prog = self.ctx.program(
                    vertex_shader='''
                        #version 330
                        in vec2 in_pos;
                        out vec2 v_uv;
                        void main() {
                            gl_Position = vec4(in_pos, 0.0, 1.0);
                            v_uv = in_pos * 0.5 + 0.5;
                        }
                    ''',
                    fragment_shader='''
                        #version 330
                        uniform sampler2D tex;
                        in vec2 v_uv;
                        out vec4 f_color;
                        void main() {
                            f_color = texture(tex, v_uv);
                        }
                    '''
                )
                verts = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
                vbo = self.ctx.buffer(verts.tobytes())
                self._blit_vao = self.ctx.simple_vertex_array(
                    self._blit_prog, vbo, 'in_pos')

            render_target.texture.use(location=0)
            self._blit_prog['tex'].value = 0

            fb_w, fb_h = self._pyglet_window.get_framebuffer_size()
            self.ctx.screen.viewport = (0, 0, fb_w, fb_h)
            self.ctx.screen.use()

            self.ctx.disable(moderngl.DEPTH_TEST)
            self._blit_vao.render(moderngl.TRIANGLE_STRIP)
            self.ctx.enable(moderngl.DEPTH_TEST)

            self._pyglet_window.flip()
        except Exception as e:
            # Pyglet flip / resize / fullscreen-toggle races can throw here.
            # Skip the frame; the next one usually succeeds.
            print(f'MGLContext.blit_to_window error: {e}')
            return False
        return True


