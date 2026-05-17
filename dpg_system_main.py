import os
import threading


def _filter_objc_warnings():
    # DearPyGui statically links GLFW while PyGLFW loads Homebrew's libglfw,
    # producing duplicate Obj-C class warnings on every launch. Benign here, so
    # drop them. The warnings go to fd 2 directly, not via sys.stderr, so we
    # redirect fd 2 through a pipe and filter at the byte level.
    saved_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)
    saved = os.fdopen(saved_fd, 'wb', buffering=0)
    reader = os.fdopen(read_fd, 'rb', buffering=0)

    def pump():
        buf = b''
        while True:
            chunk = reader.read(4096)
            if not chunk:
                break
            buf += chunk
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                if not line.startswith(b'objc['):
                    saved.write(line + b'\n')

    threading.Thread(target=pump, daemon=True).start()


_filter_objc_warnings()

import dpg_system.dpg_app
from dpg_system.dpg_app import App

dpg_app = None


def run_dpg():
    global dpg_app
    dpg_app = App()
    dpg_app.start()
    dpg_app.run_loop()


dpg_thread = threading.Thread(target=run_dpg)

try:
    dpg_thread.run()
except KeyboardInterrupt:
    print('exiting')
except Exception as e:
    import traceback
    print('dpg_system_main unhandled exception:')
    traceback.print_exception(e)

