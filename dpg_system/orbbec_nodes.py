import threading
import time
import ctypes
from collections import deque
from dpg_system.node import Node
from dpg_system.conversion_utils import *  # provides np

# The pyorbbecsdk native module is only present on machines that have a Femto
# attached and the OrbbecSDK built/installed. Import defensively so this module
# still loads (and the node still registers) on machines without it — the node
# then reports a clear error when someone tries to enable capture, rather than
# taking the whole node library down at import time.
try:
    import pyorbbecsdk as ob
    ORBBEC_AVAILABLE = True
    _ORBBEC_IMPORT_ERROR = None
except Exception as e:  # ImportError, or a native-load failure
    ob = None
    ORBBEC_AVAILABLE = False
    _ORBBEC_IMPORT_ERROR = e

MIN_DEPTH_MM = 20      # clamp reprojected points nearer than this (sensor noise)
MAX_DEPTH_MM = 10000   # ... and farther than this

# Femto Bolt depth work-modes. The Bolt uses the same Microsoft ToF module as
# the Azure Kinect, so the geometry is the K4A NFOV/WFOV × binned/unbinned set.
# name -> (width, height, max_fps). WFOV-unbinned tops out at 15 fps.
FEMTO_DEPTH_MODES = {
    'NFOV unbinned (640x576)':    (640, 576, 30),
    'NFOV binned (320x288)':      (320, 288, 30),
    'WFOV unbinned (1024x1024)':  (1024, 1024, 15),
    'WFOV binned (512x512)':      (512, 512, 30),
}

UNIT_SCALE = {'meters': 0.001, 'millimeters': 1.0}  # device native is millimetres

# Max seconds to wait for the main thread to release/hold the GL context around
# the depth-engine init. The actual release window is ~0.15s (just pipeline.start);
# this is only a safety net for a paused/stalled render loop.
GL_RELEASE_TIMEOUT = 5.0


class _DriverGLContext:
    """Save / clear / restore the process's *current* GL context at the driver
    level (GLX on X11, EGL otherwise).

    The Femto Bolt depth engine creates its own GL context inside
    pipeline.start(); if any GL context is current in the process at that moment
    it aborts the whole app (SIGABRT). We must therefore clear the current
    context for the duration of start(). pyglfw's make_context_current(None)
    can't do this because DearPyGui statically links its *own* GLFW instance —
    releasing via a different GLFW leaves the driver's per-thread current
    context bound. Clearing it directly through libGL/libEGL works because the
    current context is shared per-thread driver state. Must be called on the
    thread that holds the context (dpg's main render thread)."""

    _EGL_DRAW, _EGL_READ = 0x3059, 0x305A

    def __init__(self):
        self.saved = None  # (kind, dpy, draw, read, ctx)
        self.glx = None
        self.egl = None
        try:
            glx = ctypes.CDLL('libGL.so.1')
            glx.glXGetCurrentContext.restype = ctypes.c_void_p
            glx.glXGetCurrentDisplay.restype = ctypes.c_void_p
            glx.glXGetCurrentDrawable.restype = ctypes.c_ulong
            glx.glXGetCurrentReadDrawable.restype = ctypes.c_ulong
            glx.glXMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p]
            glx.glXMakeContextCurrent.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                                  ctypes.c_ulong, ctypes.c_void_p]
            self.glx = glx
        except Exception:
            self.glx = None
        try:
            egl = ctypes.CDLL('libEGL.so.1')
            egl.eglGetCurrentContext.restype = ctypes.c_void_p
            egl.eglGetCurrentDisplay.restype = ctypes.c_void_p
            egl.eglGetCurrentSurface.restype = ctypes.c_void_p
            egl.eglGetCurrentSurface.argtypes = [ctypes.c_int]
            egl.eglMakeCurrent.argtypes = [ctypes.c_void_p] * 4
            self.egl = egl
        except Exception:
            self.egl = None

    def release(self):
        """Save and clear the current GL context. Returns True if one was
        cleared (and restore() must later be called)."""
        if self.glx is not None:
            ctx = self.glx.glXGetCurrentContext()
            if ctx:
                dpy = self.glx.glXGetCurrentDisplay()
                draw = self.glx.glXGetCurrentDrawable()
                read = self.glx.glXGetCurrentReadDrawable()
                self.saved = ('glx', dpy, draw, read, ctx)
                self.glx.glXMakeCurrent(dpy, 0, None)
                return True
        if self.egl is not None:
            ctx = self.egl.eglGetCurrentContext()
            if ctx:
                dpy = self.egl.eglGetCurrentDisplay()
                draw = self.egl.eglGetCurrentSurface(self._EGL_DRAW)
                read = self.egl.eglGetCurrentSurface(self._EGL_READ)
                self.saved = ('egl', dpy, draw, read, ctx)
                self.egl.eglMakeCurrent(dpy, None, None, None)
                return True
        return False

    def restore(self):
        if self.saved is None:
            return
        kind, dpy, draw, read, ctx = self.saved
        try:
            if kind == 'glx':
                self.glx.glXMakeContextCurrent(dpy, draw, read, ctx)
            else:
                self.egl.eglMakeCurrent(dpy, draw, read, ctx)
        except Exception as e:
            print(f'_DriverGLContext.restore failed: {e}')
        self.saved = None


def register_orbbec_nodes():
    Node.app.register_node('femto', OrbbecFemtoNode.factory)
    Node.app.register_node('femto_bolt', OrbbecFemtoNode.factory)


class OrbbecFemtoNode(Node):
    """In-process Orbbec Femto Bolt depth source.

    Runs pyorbbecsdk on a background thread and emits, per frame:
      - depth       : (H, W) float32, in the selected units
      - point_cloud : (N, 3) float32, from the source chosen in the
                      'point cloud' option — 'sdk' (PointCloudFilter, includes
                      lens-distortion correction) or 'from depth' (reprojected
                      from raw depth via the intrinsics xy-table). Only the
                      chosen path is computed.

    Capture happens on the worker thread; only frame_task() (main thread)
    touches the output pins, keeping all DPG access single-threaded.
    """

    @staticmethod
    def factory(name, data, args=None):
        return OrbbecFemtoNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        # --- configuration state ---
        self.resolution_name = 'NFOV unbinned (640x576)'
        self.fps = 30
        self.units = 'meters'         # 'meters' | 'millimeters'
        self.pc_mode = 'from depth'   # 'none' | 'sdk' | 'from depth'

        # --- inputs / widgets ---
        self.enable_input = self.add_input('enable', widget_type='checkbox',
                                           default_value=False, callback=self.enable_changed)

        self.resolution_input = self.add_input('resolution', widget_type='combo',
                                               default_value=self.resolution_name,
                                               callback=self.stream_config_changed)
        self.resolution_input.widget.combo_items = list(FEMTO_DEPTH_MODES.keys())

        self.fps_input = self.add_input('fps', widget_type='combo', default_value='30',
                                        callback=self.stream_config_changed)
        self.fps_input.widget.combo_items = ['5', '15', '25', '30']

        # --- orientation correction ---
        # The native point cloud follows the optical convention (+X right, +Y
        # down, +Z forward), so a +Y-up viewer shows it inverted, and a tilted
        # mount adds pitch/roll. One checkbox drives the whole leveling flow:
        # on = enable the accel stream, gather fresh gravity samples, compute a
        # rotation mapping measured up -> +Y, apply it (recomputed on every
        # toggle-on, so re-aiming the camera just needs an off/on). Gravity
        # gives pitch/roll only, so 'yaw' sets heading about the up axis
        # manually — composed into the same 3x3. Per-axis flip options are the
        # deterministic fallback / residual fix, and work without the IMU.
        self.level_input = self.add_input('level to gravity', widget_type='checkbox',
                                          default_value=False, callback=self.level_changed)
        self.yaw_input = self.add_input('yaw (deg)', widget_type='drag_float',
                                        default_value=0.0, min=-180.0, max=180.0)

        # --- outputs ---
        self.depth_out = self.add_output('depth')
        self.point_cloud_out = self.add_output('point_cloud')

        # --- options ---
        self.units_option = self.add_option('units', widget_type='combo',
                                            default_value=self.units, callback=self.units_changed)
        self.units_option.widget.combo_items = ['meters', 'millimeters']
        self.pc_mode_option = self.add_option('point cloud', widget_type='combo',
                                              default_value=self.pc_mode, callback=self.pc_mode_changed)
        self.pc_mode_option.widget.combo_items = ['none', 'sdk', 'from depth']
        self.flip_x_option = self.add_option('flip x', widget_type='checkbox', default_value=False)
        self.flip_y_option = self.add_option('flip y', widget_type='checkbox', default_value=False)
        self.flip_z_option = self.add_option('flip z', widget_type='checkbox', default_value=False)

        # --- runtime / threading ---
        self.pipeline = None
        self.pc_filter = None
        self.capture_thread = None
        self.keep_running = False
        self.lock = threading.Lock()

        self.new_data = False
        self.latest_depth = None
        self.latest_pc = None

        # IMU / auto-level state.
        self.imu_enabled = False        # mirror of the level checkbox, read on capture thread
        self.accel_streaming = False    # accel stream up in the current pipeline
        self.accel_ext_rot = None       # (3,3) accel-frame -> depth-frame rotation
        self.gravity_samples = deque(maxlen=120)  # recent accel readings (guarded by lock)
        self.level_rot = None           # (3,3) computed leveling rotation, persisted
        self.level_pending = False      # waiting for fresh samples to compute level_rot

        # cached xy-table for the reprojection path, rebuilt whenever the depth
        # intrinsics/resolution change. xt/yt are (H, W) float32 of the
        # normalised ray directions (u-cx)/fx and (v-cy)/fy.
        self.xt = None
        self.yt = None
        self.xy_table_shape = None

        self.add_frame_task()

    # ------------------------------------------------------------------ config

    def enable_changed(self):
        if self.enable_input():
            self.start_capture()
        else:
            self.stop_capture()

    def stream_config_changed(self):
        # Resolution / fps require a pipeline restart to take effect.
        self.resolution_name = self.resolution_input()
        raw_fps = str(self.fps_input()).split()
        if raw_fps:
            try:
                self.fps = int(raw_fps[0])
            except (ValueError, TypeError):
                pass
        if self.keep_running:
            self.stop_capture()
            self.start_capture()

    def units_changed(self):
        self.units = self.units_option()
        # invalidate the cached xy-table dependence on units? no — the table is
        # unitless; unit scaling is applied at output. Nothing to rebuild.

    def pc_mode_changed(self):
        self.pc_mode = self.pc_mode_option()

    def level_changed(self):
        """The single leveling checkbox. On: enable the accel stream, gather
        fresh gravity samples, then compute and apply the leveling rotation
        (finished in frame_task once enough samples are in). Off: just stop
        applying — the accel stream is left running until the next natural
        pipeline restart, because a restart costs the depth engine's multi-
        second warm-up, and toggling back on can then recompute instantly."""
        if self.level_input():
            self.imu_enabled = True
            with self.lock:
                self.gravity_samples.clear()
            self.level_pending = True
            if self.keep_running and not self.accel_streaming:
                self.stop_capture()
                self.start_capture()
            elif not self.keep_running:
                print(f'{self.label}: will level to gravity when capture is enabled')
        else:
            self.imu_enabled = False
            self.level_pending = False

    # ------------------------------------------------------------- capture life

    def start_capture(self):
        if self.keep_running:
            return
        # Capture the current leveling widget state for the worker thread
        # (covers a start driven by loading a saved patch, where level_changed
        # never fired). A checked box means: gather samples and recompute.
        self.imu_enabled = bool(self.level_input())
        if self.imu_enabled:
            with self.lock:
                self.gravity_samples.clear()
            self.level_pending = True
        if not ORBBEC_AVAILABLE:
            print(f'{self.label}: pyorbbecsdk not available ({_ORBBEC_IMPORT_ERROR}); '
                  f'cannot start Femto capture')
            # reflect the failure back in the checkbox
            self.enable_input.set(False)
            return
        self.keep_running = True
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()

    def stop_capture(self):
        self.keep_running = False
        thread = self.capture_thread
        self.capture_thread = None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
            if thread.is_alive():
                print(f'{self.label}: capture thread did not exit within 2s')

    def _select_depth_profile(self, profile_list):
        """Pick the VideoStreamProfile matching the chosen resolution/fps,
        falling back to the device default if that exact mode is unavailable."""
        width, height, max_fps = FEMTO_DEPTH_MODES.get(
            self.resolution_name, FEMTO_DEPTH_MODES['NFOV unbinned (640x576)'])
        fps = min(self.fps, max_fps)
        try:
            return profile_list.get_video_stream_profile(width, height, ob.OBFormat.Y16, fps)
        except Exception as e:
            print(f'{self.label}: depth mode {width}x{height}@{fps} unavailable ({e}); '
                  f'using device default')
            return profile_list.get_default_video_stream_profile()

    def _gl_safe_pipeline_start(self, start_fn):
        """Run start_fn() (the pipeline.start that inits the depth engine) with
        dpg's OpenGL context briefly cleared on the main render thread.

        The Femto Bolt's depth engine creates its own GL context inside
        pipeline.start(); if any GL context is current in the process at that
        moment it aborts the whole app (SIGABRT). Only the ~0.15s start() window
        is sensitive — steady-state capture coexists with dpg's GL fine. So we
        ask the main thread to clear its current context (at the GLX/EGL driver
        level — see _DriverGLContext), run start() here on the capture thread,
        then let the main thread restore it.

        Falls back to a plain start() when GL isn't active (headless / no
        gl_nodes), where there is no context to collide with."""
        app = Node.app
        window_context = getattr(app, 'window_context', None)
        gl_separate = getattr(app, 'gl_on_separate_thread', False)
        gl_active = (window_context is not None and not gl_separate
                     and hasattr(app, 'queue_main_thread_call'))
        if not gl_active:
            start_fn()
            return

        released = threading.Event()
        start_done = threading.Event()

        def release_on_main():
            # Runs on the main render thread. Clear the current GL context, let
            # the depth engine init on the capture thread, then restore. Blocks
            # the render loop for the duration of start() (~0.15s).
            driver_ctx = _DriverGLContext()
            try:
                driver_ctx.release()
                released.set()
                start_done.wait(GL_RELEASE_TIMEOUT)
            finally:
                driver_ctx.restore()

        app.queue_main_thread_call(release_on_main)
        if not released.wait(GL_RELEASE_TIMEOUT):
            # Main loop never released the context (paused / not yet running).
            # Starting now would init the depth engine with GL bound -> SIGABRT,
            # so refuse rather than crash the whole app.
            start_done.set()
            raise RuntimeError('main thread did not release GL context; '
                               'is the patch paused? aborting Femto start')
        try:
            start_fn()
        finally:
            start_done.set()

    def capture_loop(self):
        try:
            self.pipeline = ob.Pipeline()
            config = ob.Config()
            profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            depth_profile = self._select_depth_profile(profile_list)
            config.enable_stream(depth_profile)
            self.accel_streaming = False
            if self.imu_enabled:
                try:
                    config.enable_accel_stream()
                    self.accel_streaming = True
                except Exception as e:
                    print(f'{self.label}: could not enable IMU accel stream: {e}')
            self._gl_safe_pipeline_start(lambda: self.pipeline.start(config))
            self.pc_filter = ob.PointCloudFilter()
            self.pc_filter.set_create_point_format(ob.OBFormat.POINT)  # XYZ only
            self._read_accel_extrinsic(depth_profile)
        except Exception as e:
            print(f'{self.label}: failed to start Femto pipeline: {e}')
            self.keep_running = False
            self.pipeline = None
            # bounce the enable checkbox off on the main thread's next tick
            self.new_data = False
            return

        while self.keep_running:
            try:
                frames = self.pipeline.wait_for_frames(200)
                if frames is None:
                    continue
                if self.accel_streaming:
                    self._read_accel_frame(frames)
                depth_frame = frames.get_depth_frame()
                if depth_frame is None:
                    continue
                self._process_frames(frames, depth_frame)
            except Exception as e:
                if self.keep_running:
                    print(f'{self.label}: capture error: {e}')
                    time.sleep(0.01)

        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception as e:
            print(f'{self.label}: pipeline stop error: {e}')
        self.pipeline = None
        self.pc_filter = None
        self.accel_streaming = False

    def _process_frames(self, frames, depth_frame):
        unit_scale = UNIT_SCALE.get(self.units, 0.001)

        width = depth_frame.get_width()
        height = depth_frame.get_height()
        depth_scale = depth_frame.get_depth_scale()  # -> millimetres per raw unit

        raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(height, width)
        depth_mm = raw.astype(np.float32) * depth_scale

        depth_out = depth_mm * unit_scale

        pc = None
        if self.pc_mode == 'sdk':
            pc = self._sdk_point_cloud(frames, unit_scale)
        elif self.pc_mode == 'from depth':
            pc = self._reproject_point_cloud(depth_mm, width, height, unit_scale)

        with self.lock:
            self.latest_depth = depth_out
            self.latest_pc = pc
            self.new_data = True

    def _sdk_point_cloud(self, frames, unit_scale):
        try:
            pc_frame = self.pc_filter.process(frames)
            if pc_frame is None:
                return None
            data = np.frombuffer(pc_frame.get_data(), dtype=np.float32)
            n = data.size // 3
            if n == 0:
                return None
            pts = data[:n * 3].reshape(n, 3)
            # SDK POINT output is in millimetres; scale to selected units.
            return (pts * unit_scale).astype(np.float32)
        except Exception as e:
            print(f'{self.label}: SDK point cloud error: {e}')
            return None

    def _ensure_xy_table(self, width, height):
        """Build/cache the normalised ray directions from depth intrinsics.
        get_camera_param() is only valid after the pipeline has delivered a
        frame, which is guaranteed by the time we reproject."""
        if self.xy_table_shape == (height, width) and self.xt is not None:
            return True
        try:
            cam_param = self.pipeline.get_camera_param()
            d = cam_param.depth_intrinsic
            fx, fy, cx, cy = d.fx, d.fy, d.cx, d.cy
        except Exception as e:
            print(f'{self.label}: could not read depth intrinsics: {e}')
            return False
        us = np.arange(width, dtype=np.float32)
        vs = np.arange(height, dtype=np.float32)
        self.xt = ((us - cx) / fx)[None, :].repeat(height, axis=0)   # (H, W)
        self.yt = ((vs - cy) / fy)[:, None].repeat(width, axis=1)    # (H, W)
        self.xy_table_shape = (height, width)
        return True

    def _reproject_point_cloud(self, depth_mm, width, height, unit_scale):
        if not self._ensure_xy_table(width, height):
            return None
        z = depth_mm  # (H, W) millimetres; Orbbec convention: +Z forward
        valid = (z >= MIN_DEPTH_MM) & (z <= MAX_DEPTH_MM)
        if not valid.any():
            return np.empty((0, 3), dtype=np.float32)
        zf = z[valid]
        xf = self.xt[valid] * zf
        yf = self.yt[valid] * zf
        pts = np.stack((xf, yf, zf), axis=1)  # (N, 3) millimetres
        return (pts * unit_scale).astype(np.float32)

    # --------------------------------------------------------------------- IMU

    def _read_accel_extrinsic(self, depth_profile):
        """Cache the accel-frame -> depth-frame rotation so gravity samples can
        be expressed in the point-cloud frame. Best-effort: leaves the extrinsic
        as identity-less None (treated as identity at calibrate time) if the
        device/SDK doesn't provide it."""
        self.accel_ext_rot = None
        if not self.accel_streaming:
            return
        try:
            accel_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.ACCEL_SENSOR)
            accel_profile = accel_list.get_accel_stream_profile(
                ob.OBAccelFullScaleRange.ACCEL_FS_UNKNOWN,
                ob.OBGyroSampleRate.SAMPLE_RATE_UNKNOWN)
            ext = accel_profile.get_extrinsic_to(depth_profile)
            self.accel_ext_rot = np.array(ext.rot, dtype=np.float64).reshape(3, 3)
        except Exception as e:
            print(f'{self.label}: could not read IMU->depth extrinsic '
                  f'(leveling will assume aligned axes): {e}')

    def _read_accel_frame(self, frames):
        """Stash the latest accelerometer reading (device units, ~m/s^2). At
        rest this vector points *up* (specific force opposing gravity)."""
        try:
            accel = frames.get_accel_frame()
        except Exception:
            return
        if accel is None:
            return
        sample = (accel.get_x(), accel.get_y(), accel.get_z())
        with self.lock:
            self.gravity_samples.append(sample)

    @staticmethod
    def _level_rotation(up):
        """Rotation R (3x3) mapping the measured up vector to world +Y while
        keeping the camera's forward axis (+Z, projected perpendicular to up)
        as world +Z. A minimal align-two-vectors rotation is NOT usable here:
        in the optical frame up is ~(0,-1,0), nearly antiparallel to +Y, so the
        minimal-rotation axis is set by the tiny mount tilt — pitch picks ~X and
        flips z negative, roll picks ~Z and doesn't. Fixing forward removes that
        degeneracy."""
        up = np.asarray(up, dtype=np.float64)
        up = up / (np.linalg.norm(up) + 1e-12)
        fw = np.array([0.0, 0.0, 1.0]) - np.dot(np.array([0.0, 0.0, 1.0]), up) * up
        if np.linalg.norm(fw) < 1e-6:
            # Camera looking straight up/down: use image-up (-Y) as forward.
            fw = np.array([0.0, -1.0, 0.0]) - np.dot(np.array([0.0, -1.0, 0.0]), up) * up
        fw = fw / np.linalg.norm(fw)
        right = np.cross(up, fw)
        return np.stack((right, up, fw))   # rows: R @ up = +Y, R @ fw = +Z

    LEVEL_SAMPLES_NEEDED = 20

    def _try_compute_level(self):
        """Finish a pending leveling: once enough fresh gravity samples have
        accumulated after the checkbox toggled on, build the rotation mapping
        the measured 'up' vector (in the point-cloud frame) onto world +Y.
        Fixes the optical-convention Y inversion and any mount pitch/roll in
        one go. Called from frame_task until it succeeds."""
        if self.keep_running and not self.accel_streaming:
            # Accel stream failed to come up; leveling can never complete.
            print(f'{self.label}: no accel stream; cannot level to gravity')
            self.level_pending = False
            return
        with self.lock:
            samples = list(self.gravity_samples)
        if len(samples) < self.LEVEL_SAMPLES_NEEDED:
            return   # keep waiting; capture may still be warming up
        g = np.array(samples, dtype=np.float64).mean(axis=0)  # accel frame, ~up
        if self.accel_ext_rot is not None:
            g = self.accel_ext_rot @ g                        # -> point-cloud frame
        if np.linalg.norm(g) < 1e-6:
            print(f'{self.label}: degenerate gravity vector; not leveling')
            self.level_pending = False
            return
        up = g / np.linalg.norm(g)
        self.level_rot = self._level_rotation(up).astype(np.float32)
        self.level_pending = False
        print(f'{self.label}: leveled (measured up in cloud frame = {up.round(3)})')

    def _orient(self, pts):
        """Apply the calibrated leveling rotation, manual yaw (about the up
        axis), then manual axis flips, all folded into one 3x3 applied
        column-wise. Deliberately NOT ``pts @ M.T``: a per-frame (N,3) BLAS gemm
        wakes the whole OpenBLAS thread pool, which then spin-waits on every
        core between frames, starving the render and capture threads. Runs on
        the main thread (frame_task), so reading the widgets here is safe."""
        if pts is None or pts.shape[0] == 0:
            return pts
        sx = -1.0 if self.flip_x_option() else 1.0
        sy = -1.0 if self.flip_y_option() else 1.0
        sz = -1.0 if self.flip_z_option() else 1.0
        yaw = float(self.yaw_input())
        level = self.level_rot if (self.level_input() and self.level_rot is not None) else None

        if level is None and yaw == 0.0:
            if sx != 1.0 or sy != 1.0 or sz != 1.0:
                pts = pts * np.array([sx, sy, sz], dtype=np.float32)
            return np.ascontiguousarray(pts, dtype=np.float32)

        m = level if level is not None else np.eye(3, dtype=np.float32)
        if yaw != 0.0:
            # After leveling +Y is up, so yaw about +Y (positive = camera
            # forward swings toward +X, right-hand rule).
            a = np.radians(yaw)
            c, s = np.float32(np.cos(a)), np.float32(np.sin(a))
            ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
            m = ry @ m
        if sx != 1.0 or sy != 1.0 or sz != 1.0:   # flip = scale the rows
            m = m * np.array([sx, sy, sz], dtype=np.float32)[:, None]
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        out = np.empty_like(pts)
        out[:, 0] = m[0, 0] * x + m[0, 1] * y + m[0, 2] * z
        out[:, 1] = m[1, 0] * x + m[1, 1] * y + m[1, 2] * z
        out[:, 2] = m[2, 0] * x + m[2, 1] * y + m[2, 2] * z
        return out

    # ------------------------------------------------------------- main thread

    def frame_task(self):
        if self.level_pending:
            self._try_compute_level()
        if not self.new_data:
            return
        with self.lock:
            depth = self.latest_depth
            pc = self.latest_pc
            self.new_data = False
        # Send in reverse pin order so downstream fan-out sees the point cloud
        # settle before the depth trigger, matching dpg's right-to-left output
        # convention used elsewhere.
        if pc is not None:
            self.point_cloud_out.send(self._orient(pc))
        if depth is not None:
            self.depth_out.send(depth)

    def save_custom(self, container):
        # Persist the computed leveling rotation so a saved patch renders
        # leveled immediately on load, before the toggle-on recalculation
        # (which needs capture + fresh IMU samples) completes and replaces it.
        # Widget values (level / yaw / flips / options) are saved by the framework.
        if self.level_rot is not None:
            container['level_rot'] = self.level_rot.reshape(-1).tolist()

    def load_custom(self, container):
        rot = container.get('level_rot', None)
        if rot is not None and len(rot) == 9:
            self.level_rot = np.array(rot, dtype=np.float32).reshape(3, 3)

    def custom_cleanup(self):
        self.stop_capture()
