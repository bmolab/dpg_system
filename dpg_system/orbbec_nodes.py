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
                      'point cloud' option — 'sdk' (PointCloudFilter) or
                      'from depth' (reprojected from depth via the intrinsics
                      xy-table; with 'undistort' on, the table bakes in the
                      inverse Brown-Conrady depth distortion, matching the SDK
                      cloud). Only the chosen path is computed.

    Depth-image-domain cleanup (same order as the C++ AzureKinectVoxelsApp):
    temporal hole fill -> 3x3 median -> guard-space background removal, all on
    the (H, W) frame before any cloud is built. 'remove background' snapshots
    the scene on toggle-on (min depth over N frames, minus a guard) and then
    removes any pixel at or beyond its per-pixel threshold.

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

        # Toggle-on snapshots the background (keep the volume empty for the
        # first ~N frames), then removes it; toggle off/on to recapture.
        self.remove_bg_input = self.add_input('remove background', widget_type='checkbox',
                                              default_value=False, callback=self.remove_bg_changed)

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
        self.undistort_option = self.add_option('undistort', widget_type='checkbox',
                                                default_value=True, callback=self.filters_changed)
        self.median_option = self.add_option('median filter', widget_type='checkbox',
                                             default_value=False, callback=self.filters_changed)
        self.fill_holes_option = self.add_option('fill holes', widget_type='checkbox',
                                                 default_value=False, callback=self.filters_changed)
        self.bg_frames_option = self.add_option('background frames', widget_type='drag_int',
                                                default_value=30, min=1, callback=self.filters_changed)
        self.bg_guard_option = self.add_option('background guard (mm)', widget_type='drag_float',
                                               default_value=50.0, min=0.0, callback=self.filters_changed)
        # Serve the most recent frame (k4a-style latency minimization): after
        # receiving a frameset, drain anything the SDK queued behind it and
        # process only the newest. Off = process every frameset in order, at
        # the cost of up to a full queue of latency when the host falls behind.
        self.min_latency_option = self.add_option('minimize latency', widget_type='checkbox',
                                                  default_value=True, callback=self.filters_changed)
        # Prints capture-thread and output gaps > 80ms with a thread tag, to
        # tell SDK-side stalls from main-thread (render loop) stalls.
        self.report_gaps_option = self.add_option('report frame gaps', widget_type='checkbox',
                                                  default_value=False, callback=self.filters_changed)
        # The Bolt's USB session can degrade into a bistable 'bursty' delivery
        # state (frames arrive in clumps of 3-4 every ~115ms; the camera itself
        # stays on time). It survives pipeline restarts; only a USB device
        # reset clears it. Detection watches the inter-frame cadence; the reset
        # costs ~8s of frames (re-enumeration + depth engine warm-up).
        self.auto_reset_option = self.add_option('auto usb reset on stutter', widget_type='checkbox',
                                                 default_value=True, callback=self.filters_changed)
        self.reset_usb_option = self.add_option('reset usb device', widget_type='button',
                                                callback=self.manual_usb_reset)

        # --- runtime / threading ---
        self.pipeline = None
        self.pc_filter = None
        self.capture_thread = None
        self.keep_running = False
        self.lock = threading.Lock()

        self.new_data = False
        self.latest_depth = None
        self.latest_pc = None

        # IMU / auto-level state. The accel sensor runs OUTSIDE the pipeline
        # (sensor API) and only while a leveling calibration is pending — an
        # active IMU stream degrades the Bolt's depth delivery into clumps.
        self.imu_enabled = False        # mirror of the level checkbox, read on capture thread
        self.accel_sensor = None        # ob Sensor while briefly streaming (capture thread)
        self.accel_streaming = False    # accel sensor currently delivering
        self._accel_stop_requested = False
        self.depth_profile = None       # current depth profile (for the IMU extrinsic)
        self.accel_ext_rot = None       # (3,3) accel-frame -> depth-frame rotation
        self.gravity_samples = deque(maxlen=120)  # recent accel readings (guarded by lock)
        self.level_rot = None           # (3,3) computed leveling rotation, persisted
        self.level_pending = False      # waiting for fresh samples to compute level_rot

        # Depth-image filter state. The *_enabled/bg_* scalars are mirrors of
        # the widgets, read on the capture thread (same pattern as imu_enabled).
        self.median_enabled = False
        self.fill_holes_enabled = False
        self.undistort_enabled = True
        self.low_latency = True
        self.report_gaps = False
        self._last_capture_t = None     # capture-thread frame arrival times
        self._last_output_t = None      # main-thread send times

        # Bursty-USB detection / recovery state.
        self.auto_reset_enabled = True
        self._gap_window = deque(maxlen=150)   # recent capture gaps (capture thread only)
        self._gap_frames = 0
        self._last_burst_warn = 0.0
        self._last_auto_reset = 0.0
        self._reset_requested = False
        self._resetting = False
        self.bg_guard_mm = 50.0
        self.bg_capture_frames = 30
        self.bg_remove_active = False
        self.bg_capture_remaining = 0   # >0 while snapshotting the background
        self.bg_min = None              # (H, W) min-depth accumulator during snapshot
        self.bg_threshold = None        # (H, W) float32 mm; >= here means background
        self.held_depth = None          # (H, W) last known depth per pixel (hole fill)
        self._bg_removed = None         # (H, W) bool, this frame's removals (capture thread)

        # cached xy-table for the reprojection path, rebuilt whenever the depth
        # intrinsics/resolution (or the undistort option) change. xt/yt are
        # (H, W) float32 ray directions; with undistort on, the inverse
        # Brown-Conrady model is baked in so reprojection matches the SDK cloud.
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

    def filters_changed(self):
        self.median_enabled = bool(self.median_option())
        self.fill_holes_enabled = bool(self.fill_holes_option())
        self.bg_guard_mm = float(self.bg_guard_option())
        self.bg_capture_frames = max(1, int(self.bg_frames_option()))
        self.low_latency = bool(self.min_latency_option())
        self.report_gaps = bool(self.report_gaps_option())
        self.auto_reset_enabled = bool(self.auto_reset_option())
        undistort = bool(self.undistort_option())
        if undistort != self.undistort_enabled:
            self.undistort_enabled = undistort
            self.xy_table_shape = None   # force a ray-table rebuild

    def remove_bg_changed(self):
        """Toggle-on snapshots the background from the next N frames (min depth
        per pixel, minus the guard) and then removes it; toggle off stops
        removing. Re-toggle to recapture — keep the volume empty while the
        snapshot runs."""
        if self.remove_bg_input():
            self.bg_min = None
            self.bg_threshold = None
            self.bg_capture_remaining = self.bg_capture_frames
            self.bg_remove_active = True
            if self.keep_running:
                print(f'{self.label}: capturing background over '
                      f'{self.bg_capture_frames} frames (keep the volume empty)')
            else:
                print(f'{self.label}: will capture background when capture is enabled')
        else:
            self.bg_remove_active = False
            self.bg_capture_remaining = 0

    def level_changed(self):
        """The single leveling checkbox. On: briefly stream the accelerometer
        (sensor API, no pipeline restart), gather fresh gravity samples,
        compute and apply the leveling rotation, then stop the accel again —
        an active IMU stream disrupts the Bolt's depth delivery into ~115ms
        clumps, so it must only run for the ~1s the calibration needs. Off:
        just stop applying (recomputed on every toggle-on)."""
        if self.level_input():
            self.imu_enabled = True
            with self.lock:
                self.gravity_samples.clear()
            self.level_pending = True   # capture thread starts the accel sensor
            if not self.keep_running:
                print(f'{self.label}: will level to gravity when capture is enabled')
        else:
            self.imu_enabled = False
            self.level_pending = False
            self._accel_stop_requested = True

    # ------------------------------------------------------------- capture life

    def start_capture(self):
        if self.keep_running:
            return
        # Capture the current widget state for the worker thread (covers a
        # start driven by loading a saved patch, where the callbacks never
        # fired). A checked level box means: gather samples and recompute; a
        # checked background box with no snapshot yet means: capture one.
        self.imu_enabled = bool(self.level_input())
        if self.imu_enabled:
            with self.lock:
                self.gravity_samples.clear()
            self.level_pending = True
        self.filters_changed()
        if self.remove_bg_input() and self.bg_threshold is None:
            self.remove_bg_changed()
        if not ORBBEC_AVAILABLE:
            print(f'{self.label}: pyorbbecsdk not available ({_ORBBEC_IMPORT_ERROR}); '
                  f'cannot start Femto capture')
            # reflect the failure back in the checkbox
            self.enable_input.set(False)
            return
        # fresh cadence stats for the new session (the restart itself would
        # otherwise register as a giant gap)
        self._gap_window.clear()
        self._gap_frames = 0
        self._last_capture_t = None
        self._last_output_t = None
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
            # NOTE: the accel stream is deliberately NOT part of the pipeline —
            # an active IMU stream (pipeline OR sensor API) degrades the Bolt's
            # depth delivery into ~115ms clumps, so the accel sensor is only
            # started briefly while a leveling calibration is pending.
            self.depth_profile = depth_profile
            self._gl_safe_pipeline_start(lambda: self.pipeline.start(config))
            self.pc_filter = ob.PointCloudFilter()
            self.pc_filter.set_create_point_format(ob.OBFormat.POINT)  # XYZ only
        except Exception as e:
            print(f'{self.label}: failed to start Femto pipeline: {e}')
            self.keep_running = False
            self.pipeline = None
            # bounce the enable checkbox off on the main thread's next tick
            self.new_data = False
            return

        while self.keep_running:
            try:
                # accel sensor lifecycle (kept on this thread): started only
                # while a leveling calibration wants samples, stopped as soon
                # as it completes — see level_changed / _try_compute_level.
                if self.level_pending and self.accel_sensor is None and self.imu_enabled:
                    self._start_accel_sensor()
                if self._accel_stop_requested:
                    self._accel_stop_requested = False
                    self._stop_accel_sensor()

                frames = self.pipeline.wait_for_frames(200)
                if frames is None:
                    continue
                self._note_frame_arrival()
                if self.low_latency:
                    # Drain to the newest queued frameset so a host that fell
                    # behind serves the live frame, not the back of a queue.
                    # 1ms (not 0) in case the binding treats 0 as 'wait forever'.
                    # Drained framesets still feed the gap stats: the burst
                    # detector needs the delivery cadence, not what we process.
                    while True:
                        newer = self.pipeline.wait_for_frames(1)
                        if newer is None:
                            break
                        frames = newer
                        self._note_frame_arrival()
                depth_frame = frames.get_depth_frame()
                if depth_frame is None:
                    continue
                self._process_frames(frames, depth_frame)
            except Exception as e:
                if self.keep_running:
                    print(f'{self.label}: capture error: {e}')
                    time.sleep(0.01)

        self._stop_accel_sensor()
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception as e:
            print(f'{self.label}: pipeline stop error: {e}')
        self.pipeline = None
        self.pc_filter = None

    def _process_frames(self, frames, depth_frame):
        unit_scale = UNIT_SCALE.get(self.units, 0.001)

        width = depth_frame.get_width()
        height = depth_frame.get_height()
        depth_scale = depth_frame.get_depth_scale()  # -> millimetres per raw unit

        raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(height, width)
        depth_mm = raw.astype(np.float32) * depth_scale

        depth_mm = self._filter_depth(depth_mm)
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

    def _note_frame_arrival(self):
        """Record the inter-frameset gap for the burst detector (capture
        thread). Called once per frameset pulled from the SDK, including
        stale ones consumed by the minimize-latency drain."""
        now = time.perf_counter()
        if self._last_capture_t is not None:
            gap = now - self._last_capture_t
            if self.report_gaps and gap > 0.08:
                print(f'{self.label}: capture gap {gap * 1000:.0f}ms (SDK side)')
            self._gap_window.append(gap)
            self._gap_frames += 1
            if self._gap_frames >= 120:
                self._gap_frames = 0
                self._check_burst_state()
        self._last_capture_t = now

    # ------------------------------------------------------ bursty-USB recovery

    def _check_burst_state(self):
        """Detect the Bolt's degraded USB delivery state (runs on the capture
        thread). Signature: frames arriving in clumps — median inter-frame gap
        far below the 33ms frame period, with frequent >60ms stalls. The camera
        clock stays steady, only host delivery batches, and the state survives
        pipeline restarts; a USB device reset clears it."""
        if self.accel_streaming:
            return   # clumps are expected while the leveling accel runs
        g = np.fromiter(self._gap_window, dtype=np.float64)
        if g.size < 100:
            return
        if np.median(g) < 0.020 and np.sum(g > 0.06) >= 8:
            now = time.monotonic()
            if now - self._last_burst_warn > 30:
                self._last_burst_warn = now
                print(f'{self.label}: bursty USB frame delivery detected '
                      f'(clumps + ~100ms stalls); a USB device reset clears this')
            if (self.auto_reset_enabled and not self._resetting
                    and now - self._last_auto_reset > 120):
                self._last_auto_reset = now
                self._reset_requested = True   # frame_task picks this up

    def _usb_reset_device(self):
        """Issue USBDEVFS_RESET to the first Orbbec (vid 2bc5) USB device.
        ENODEV on the ioctl is expected — the device drops off the bus and
        re-enumerates, which is exactly the point."""
        import fcntl, glob
        USBDEVFS_RESET = 21780
        for vend_path in glob.glob('/sys/bus/usb/devices/*/idVendor'):
            try:
                with open(vend_path) as f:
                    if f.read().strip() != '2bc5':
                        continue
                base = vend_path[:-len('idVendor')]
                with open(base + 'busnum') as f:
                    bus = int(f.read().strip())
                with open(base + 'devnum') as f:
                    dev = int(f.read().strip())
                node = f'/dev/bus/usb/{bus:03d}/{dev:03d}'
                try:
                    with open(node, 'wb') as f:
                        fcntl.ioctl(f, USBDEVFS_RESET, 0)
                except OSError as e:
                    if e.errno != 19:   # ENODEV == successful re-enumeration
                        raise
                print(f'{self.label}: USB reset issued to {node}')
                return True
            except Exception as e:
                print(f'{self.label}: USB reset via {vend_path} failed: {e}')
        print(f'{self.label}: no Orbbec USB device found to reset')
        return False

    def _do_usb_reset(self):
        """Worker: stop capture, reset the device, wait for re-enumeration and
        restart. Costs ~8s of frames (reset + depth engine warm-up)."""
        try:
            was_running = self.keep_running
            self.stop_capture()
            self._usb_reset_device()
            time.sleep(6.0)
            if was_running:
                self.start_capture()
        finally:
            self._resetting = False

    def manual_usb_reset(self):
        if self._resetting:
            return
        self._resetting = True
        threading.Thread(target=self._do_usb_reset, daemon=True).start()

    # ------------------------------------------------------- depth-image filters

    # Median-of-9 sorting network (19 min/max exchanges). Each exchange is a
    # SIMD pass over the whole (H, W) plane, ~3ms total at 640x576 — scipy's
    # median_filter takes ~25ms for the same result.
    _MEDIAN9_EXCHANGES = [(1, 2), (4, 5), (7, 8), (0, 1), (3, 4), (6, 7),
                          (1, 2), (4, 5), (7, 8), (0, 3), (5, 8), (4, 7),
                          (3, 6), (1, 4), (2, 5), (4, 7), (4, 2), (6, 4), (4, 2)]

    @classmethod
    def _median3x3(cls, img):
        h, w = img.shape
        p = np.pad(img, 1, mode='edge')
        win = [p[dy:dy + h, dx:dx + w].copy() for dy in range(3) for dx in range(3)]
        tmp = np.empty_like(img)
        for a, b in cls._MEDIAN9_EXCHANGES:
            np.minimum(win[a], win[b], out=tmp)
            np.maximum(win[a], win[b], out=win[b])
            win[a], tmp = tmp, win[a]
        return win[4]

    def _filter_depth(self, depth_mm):
        """Depth-image-domain cleanup on the capture thread, in the same order
        as the C++ app: temporal hole fill -> 3x3 median -> guard-space
        background removal (removed pixels are zeroed, so they drop out of both
        the depth output and the reprojection's validity mask)."""
        self._bg_removed = None

        if self.fill_holes_enabled:
            if self.held_depth is None or self.held_depth.shape != depth_mm.shape:
                self.held_depth = depth_mm.copy()
            holes = depth_mm == 0.0
            depth_mm[holes] = self.held_depth[holes]
            np.copyto(self.held_depth, depth_mm)
        else:
            self.held_depth = None

        if self.median_enabled:
            depth_mm = self._median3x3(depth_mm)

        if self.bg_capture_remaining > 0:
            self._accumulate_background(depth_mm)
        elif (self.bg_remove_active and self.bg_threshold is not None
              and self.bg_threshold.shape == depth_mm.shape):
            removed = depth_mm >= self.bg_threshold
            depth_mm[removed] = 0.0
            self._bg_removed = removed
        return depth_mm

    def _accumulate_background(self, depth_mm):
        """Min-depth snapshot: over N frames keep the closest valid reading per
        pixel; on the last frame subtract the guard and bleed each threshold to
        its 8 neighbours (3x3 minimum) so background edges are covered against
        pixel jitter. Pixels that never returned depth get +inf (never remove)."""
        frame_min = np.where(depth_mm > 0.0, depth_mm, np.inf)
        if self.bg_min is None or self.bg_min.shape != depth_mm.shape:
            self.bg_min = frame_min
        else:
            np.minimum(self.bg_min, frame_min, out=self.bg_min)
        self.bg_capture_remaining -= 1
        if self.bg_capture_remaining == 0:
            thresh = self.bg_min - np.float32(self.bg_guard_mm)
            h, w = thresh.shape
            p = np.pad(thresh, 1, mode='edge')
            for dy in range(3):
                for dx in range(3):
                    np.minimum(thresh, p[dy:dy + h, dx:dx + w], out=thresh)
            self.bg_threshold = thresh.astype(np.float32)
            covered = float(np.isfinite(thresh).mean()) * 100.0
            print(f'{self.label}: background captured '
                  f'({covered:.0f}% of pixels have a background depth)')
            self.bg_min = None

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
            # The SDK cloud is organized (one point per depth pixel, row-major),
            # so the image-domain background mask applies index-for-index. The
            # SDK consumed the *unfiltered* device frame, so hole fill / median
            # cannot reach this path — only 'from depth' gets those.
            if self._bg_removed is not None and pts.shape[0] == self._bg_removed.size:
                pts = np.compress(~self._bg_removed.ravel(), pts, axis=0)
            # SDK POINT output is in millimetres; scale to selected units.
            return (pts * unit_scale).astype(np.float32)
        except Exception as e:
            print(f'{self.label}: SDK point cloud error: {e}')
            return None

    def _ensure_xy_table(self, width, height):
        """Build/cache the ray-direction table from the depth intrinsics, with
        the inverse lens distortion baked in when 'undistort' is on — a one-time
        cost that makes the per-frame reprojection (an unchanged multiply)
        produce the same distortion-corrected cloud as the SDK filter.
        get_camera_param() is only valid after the pipeline has delivered a
        frame, which is guaranteed by the time we reproject."""
        key = (height, width, self.undistort_enabled)
        if self.xy_table_shape == key and self.xt is not None:
            return True
        try:
            cam_param = self.pipeline.get_camera_param()
            d = cam_param.depth_intrinsic
            fx, fy, cx, cy = d.fx, d.fy, d.cx, d.cy
        except Exception as e:
            print(f'{self.label}: could not read depth intrinsics: {e}')
            return False
        xd = ((np.arange(width, dtype=np.float64) - cx) / fx)[None, :].repeat(height, axis=0)
        yd = ((np.arange(height, dtype=np.float64) - cy) / fy)[:, None].repeat(width, axis=1)
        if self.undistort_enabled:
            xu, yu = self._undistort_rays(xd, yd, cam_param)
        else:
            xu, yu = xd, yd
        self.xt = xu.astype(np.float32)
        self.yt = yu.astype(np.float32)
        self.xy_table_shape = key
        return True

    def _undistort_rays(self, xd, yd, cam_param):
        """Invert the Brown-Conrady depth distortion for every pixel's
        normalised coordinates via fixed-point iteration (the standard
        undistortPoints scheme: divide out the rational radial term, subtract
        the tangential term, repeat). Returns pinhole rays unchanged if the
        device reports no distortion."""
        try:
            dist = cam_param.depth_distortion
            k1, k2, k3 = dist.k1, dist.k2, dist.k3
            k4, k5, k6 = dist.k4, dist.k5, dist.k6
            p1, p2 = dist.p1, dist.p2
        except Exception as e:
            print(f'{self.label}: no depth distortion params ({e}); using pinhole rays')
            return xd, yd
        if not any((k1, k2, k3, k4, k5, k6, p1, p2)):
            return xd, yd
        x, y = xd.copy(), yd.copy()
        for _ in range(25):
            r2 = x * x + y * y
            radial = ((1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))) /
                      (1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))))
            dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            x = (xd - dx) / radial
            y = (yd - dy) / radial
        # Report the round-trip residual so a bad model is visible immediately.
        r2 = x * x + y * y
        radial = ((1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))) /
                  (1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))))
        rx = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) - xd
        ry = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y - yd
        err = float(np.nanmax(np.hypot(rx, ry)))
        print(f'{self.label}: undistorted ray table built '
              f'(k1={k1:.4g} k2={k2:.4g} k3={k3:.4g} k4={k4:.4g} k5={k5:.4g} '
              f'k6={k6:.4g} p1={p1:.4g} p2={p2:.4g}, max residual {err:.2e})')
        if err > 1e-4:
            print(f'{self.label}: WARNING: distortion inversion residual is large; '
                  f'compare against the sdk cloud before trusting edges')
        return x, y

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

    def _start_accel_sensor(self):
        """Start the accelerometer via the sensor API (capture thread). Runs
        outside the pipeline so no restart is needed, and only for the ~1s a
        leveling calibration takes — depth delivery clumps while it's active."""
        try:
            device = self.pipeline.get_device()
            sensor = device.get_sensor_list().get_sensor_by_type(ob.OBSensorType.ACCEL_SENSOR)
            profile = sensor.get_stream_profile_list().get_accel_stream_profile(
                ob.OBAccelFullScaleRange.ACCEL_FS_UNKNOWN,
                ob.OBGyroSampleRate.SAMPLE_RATE_UNKNOWN)
            self.accel_ext_rot = None
            try:
                ext = profile.get_extrinsic_to(self.depth_profile)
                self.accel_ext_rot = np.array(ext.rot, dtype=np.float64).reshape(3, 3)
            except Exception as e:
                print(f'{self.label}: could not read IMU->depth extrinsic '
                      f'(leveling will assume aligned axes): {e}')
            sensor.start(profile, self._on_accel_frame)
            self.accel_sensor = sensor
            self.accel_streaming = True
        except Exception as e:
            print(f'{self.label}: could not start accel sensor ({e}); cannot level')
            self.accel_sensor = None
            self.accel_streaming = False
            self.level_pending = False

    def _stop_accel_sensor(self):
        """Stop the accelerometer (capture thread); depth delivery returns to a
        clean 33ms cadence immediately. Clears the cadence-watch window so the
        leveling window's clumps don't trigger a spurious USB reset."""
        if self.accel_sensor is not None:
            try:
                self.accel_sensor.stop()
            except Exception as e:
                print(f'{self.label}: accel sensor stop error: {e}')
            self.accel_sensor = None
        self.accel_streaming = False
        self._gap_window.clear()
        self._gap_frames = 0

    def _on_accel_frame(self, frame):
        """SDK callback thread: stash the latest accelerometer reading (device
        units, ~m/s^2). At rest this vector points *up* (specific force
        opposing gravity)."""
        if frame is None:
            return
        try:
            accel = frame.as_accel_frame()
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
        one go. Called from frame_task until it succeeds; every completion
        path stops the briefly-running accel sensor."""
        with self.lock:
            samples = list(self.gravity_samples)
        if len(samples) < self.LEVEL_SAMPLES_NEEDED:
            return   # keep waiting; the accel sensor may still be starting up
        g = np.array(samples, dtype=np.float64).mean(axis=0)  # accel frame, ~up
        if self.accel_ext_rot is not None:
            g = self.accel_ext_rot @ g                        # -> point-cloud frame
        self.level_pending = False
        self._accel_stop_requested = True
        if np.linalg.norm(g) < 1e-6:
            print(f'{self.label}: degenerate gravity vector; not leveling')
            return
        up = g / np.linalg.norm(g)
        self.level_rot = self._level_rotation(up).astype(np.float32)
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
        if self._reset_requested:
            self._reset_requested = False
            if not self._resetting:
                self._resetting = True
                print(f'{self.label}: auto USB reset (expect ~8s of dropped frames)')
                threading.Thread(target=self._do_usb_reset, daemon=True).start()
        if self.level_pending:
            self._try_compute_level()
        if not self.new_data:
            return
        with self.lock:
            depth = self.latest_depth
            pc = self.latest_pc
            self.new_data = False
        if self.report_gaps:
            now = time.perf_counter()
            if (self._last_output_t is not None
                    and now - self._last_output_t > 0.08):
                print(f'{self.label}: output gap '
                      f'{(now - self._last_output_t) * 1000:.0f}ms (main thread side)')
            self._last_output_t = now
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
