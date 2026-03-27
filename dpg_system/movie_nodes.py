import numpy as np
from dpg_system.node import Node
from dpg_system.conversion_utils import *
import cv2
import os
import dearpygui.dearpygui as dpg


def register_movie_nodes():
    Node.app.register_node('movie_player', MoviePlayerNode.factory)


class MoviePlayerNode(Node):
    """
    A node that loads and plays video files, outputting frames as numpy arrays.

    Supports:
    - Streaming playback via cv2.VideoCapture (open message)
    - Full movie caching in a numpy array for instant random access (import message)
    - Random frame access by sending an int
    - Play / loop / stop / pause / resume messages
    - Clip segment and speed control via messages or widgets
    - Bang output on end-of-clip or loop point
    """

    @staticmethod
    def factory(name, data, args=None):
        node = MoviePlayerNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.cap = None
        self.cached_frames = None  # numpy array (N, H, W, 3) when cached
        self.total_frames = 0
        self.fps = 30.0
        self.movie_path = ''

        # Playback state
        self.playing = False
        self.looping = False
        self.frame_accumulator = 0.0
        self.current_frame = 0

        # Parse optional path argument
        initial_path = ''
        if args is not None and len(args) > 0:
            initial_path = args[0]

        # --- Inputs ---
        self.input = self.add_input('input', triggers_execution=True)
        self.frame_input = self.add_input('frame', widget_type='drag_int', default_value=0,
                                            triggers_execution=True, callback=self.frame_widget_changed)
        self.path_input = self.add_input('path', widget_type='text_input', default_value=initial_path)

        # --- Buttons ---
        self.play_pause_button = self.add_input('play', widget_type='button', callback=self.play_button_clicked)
        self.play_pause_button.name_archive.append('pause')
        self.play_pause_button.name_archive.append('resume')
        self.stop_button = self.add_input('stop', widget_type='button', callback=self.stop_button_clicked)

        # --- Widgets ---
        self.loop_checkbox = self.add_input('loop', widget_type='checkbox', default_value=False,
                                            callback=self.loop_changed)
        self.speed_input = self.add_input('speed', widget_type='drag_float', default_value=1.0,
                                          callback=self.speed_changed)
        self.clip_start_input = self.add_input('clip_start', widget_type='drag_int', default_value=0,
                                               callback=self.clip_changed,
                                               trigger_button=True, trigger_callback=self.clip_start_set)
        self.clip_end_input = self.add_input('clip_end', widget_type='drag_int', default_value=-1,
                                              callback=self.clip_changed,
                                              trigger_button=True, trigger_callback=self.clip_end_set)

        # --- Info label ---
        self.info_label = self.add_label('no movie loaded')

        # --- Outputs ---
        self.frame_output = self.add_output('frame')
        self.frame_num_output = self.add_output('frame_num')
        self.done_output = self.add_output('done')

        # --- Message handlers ---
        self.message_handlers['open'] = self.open_message
        self.message_handlers['import'] = self.import_message
        self.message_handlers['play'] = self.play_message
        self.message_handlers['loop'] = self.loop_message
        self.message_handlers['stop'] = self.stop_message
        self.message_handlers['pause'] = self.pause_message
        self.message_handlers['resume'] = self.resume_message

    def custom_create(self, from_file):
        # Set button colors matching OpenTakeNode style
        self.play_pause_button.widget.set_active_theme(Node.active_theme_green)
        self.play_pause_button.widget.set_height(24)
        self.stop_button.widget.set_active_theme(Node.active_theme_red)
        self.stop_button.widget.set_height(24)

        path = self.path_input()
        if path and path != '':
            self.open_movie(path)

    # ------------------------------------------------------------------
    # Movie loading
    # ------------------------------------------------------------------

    def open_movie(self, path):
        """Open a movie for streaming playback (no caching)."""
        if not os.path.exists(path):
            print(f"MoviePlayerNode: file not found: {path}")
            return

        # Release previous
        self._release()

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            print(f"MoviePlayerNode: cannot open: {path}")
            self.cap = None
            return

        self.movie_path = path
        self.path_input.set(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0

        # Update clip_end widget max
        self._update_clip_limits()

        self.cached_frames = None
        self.info_label.set(f"streaming  {self.total_frames} frames  {self.fps:.1f} fps")
        print(f"MoviePlayerNode: opened '{os.path.basename(path)}' — {self.total_frames} frames @ {self.fps:.1f} fps")

    def import_movie(self, path):
        """Load entire movie into a numpy array cache."""
        if not os.path.exists(path):
            print(f"MoviePlayerNode: file not found: {path}")
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"MoviePlayerNode: cannot open: {path}")
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        print(f"MoviePlayerNode: caching '{os.path.basename(path)}' ({total} frames)...")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if len(frames) == 0:
            print("MoviePlayerNode: no frames read")
            return

        # Release previous
        self._release()

        self.cached_frames = np.array(frames)
        self.total_frames = len(frames)
        self.fps = fps
        self.movie_path = path
        self.path_input.set(path)

        self._update_clip_limits()

        mb = self.cached_frames.nbytes / (1024 * 1024)
        self.info_label.set(f"cached  {self.total_frames} frames  {mb:.0f} MB")
        print(f"MoviePlayerNode: cached {self.total_frames} frames ({mb:.0f} MB)")

    def _release(self):
        """Release capture and clear cache."""
        self.stop_playback()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cached_frames = None
        self.total_frames = 0

    def _update_clip_limits(self):
        """Update drag_int max values to match total frame count."""
        try:
            dpg.configure_item(self.clip_start_input.widget.uuid, max_value=self.total_frames)
            dpg.configure_item(self.clip_end_input.widget.uuid, max_value=self.total_frames)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame(self, frame_num):
        """Read a single frame by number. Returns RGB numpy array or None."""
        if frame_num < 0:
            frame_num = 0
        if frame_num >= self.total_frames:
            frame_num = self.total_frames - 1
        if self.total_frames == 0:
            return None

        if self.cached_frames is not None:
            return self.cached_frames[frame_num]
        elif self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def output_frame(self, frame_num):
        """Get and send a frame + frame number."""
        frame = self.get_frame(frame_num)
        if frame is not None:
            self.current_frame = frame_num
            self.frame_input.set(frame_num)
            self.frame_num_output.send(frame_num)
            self.frame_output.send(frame)

    def frame_widget_changed(self):
        """User dragged the frame widget — output that frame."""
        if not self.playing:
            data = self.frame_input()
            if data is not None:
                try:
                    frame_num = int(any_to_int(data))
                    self.output_frame(frame_num)
                except (ValueError, TypeError):
                    pass

    # ------------------------------------------------------------------
    # Playback control
    # ------------------------------------------------------------------

    def _effective_clip_start(self):
        val = int(self.clip_start_input())
        if val < 0:
            val = 0
        return val

    def _effective_clip_end(self):
        val = int(self.clip_end_input())
        if val <= 0 or val > self.total_frames:
            val = self.total_frames
        return val

    def start_playback(self, from_frame=None):
        """Start or restart playback."""
        if self.total_frames == 0:
            return

        clip_start = self._effective_clip_start()
        if from_frame is not None:
            self.current_frame = from_frame
        else:
            self.current_frame = clip_start

        self.frame_accumulator = 0.0
        self.playing = True
        self.add_frame_task()

    def stop_playback(self):
        """Stop playback completely."""
        if self.playing:
            self.playing = False
            self.remove_frame_tasks()

    def pause_playback(self):
        """Pause playback (retain position)."""
        if self.playing:
            self.playing = False
            self.remove_frame_tasks()

    def resume_playback(self):
        """Resume from current position."""
        if not self.playing and self.total_frames > 0:
            self.playing = True
            self.frame_accumulator = 0.0
            self.add_frame_task()

    # ------------------------------------------------------------------
    # Frame task (called every app frame during playback)
    # ------------------------------------------------------------------

    def frame_task(self):
        if not self.playing:
            return

        speed = float(self.speed_input())
        clip_start = self._effective_clip_start()
        clip_end = self._effective_clip_end()

        if clip_end <= clip_start:
            self.stop_playback()
            return

        # Accumulate fractional frames
        self.frame_accumulator += speed
        frames_to_advance = int(self.frame_accumulator)
        self.frame_accumulator -= frames_to_advance

        if frames_to_advance <= 0:
            # Still output current frame for sub-speed
            self.output_frame(self.current_frame)
            return

        self.current_frame += frames_to_advance

        # Check if we've reached or passed the end
        if self.current_frame >= clip_end:
            if self.looping:
                # Wrap around
                overshoot = self.current_frame - clip_end
                self.current_frame = clip_start + (overshoot % (clip_end - clip_start))
                self.output_frame(self.current_frame)
                self.done_output.send('bang')
            else:
                # Play the last frame and stop
                self.current_frame = clip_end - 1
                self.output_frame(self.current_frame)
                self.done_output.send('bang')
                self.stop_playback()
                self.current_frame = 0
                self.play_pause_button.set_label('play')
                self.play_pause_button.widget.set_active_theme(Node.active_theme_green)
        else:
            self.output_frame(self.current_frame)

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def _parse_play_args(self, message_data):
        """Parse optional args from play/loop messages: ints → start/end, float → speed."""
        ints_found = []
        float_found = None

        for arg in message_data:
            try:
                # Check if it's a float (contains '.')
                if '.' in str(arg):
                    float_found = float(arg)
                else:
                    ints_found.append(int(arg))
            except (ValueError, TypeError):
                pass

        if len(ints_found) >= 1:
            self.clip_start_input.set(ints_found[0])
        if len(ints_found) >= 2:
            self.clip_end_input.set(ints_found[1])
        if float_found is not None:
            self.speed_input.set(float_found)

    def open_message(self, message='', message_data=[]):
        if len(message_data) > 0:
            path = ' '.join([str(a) for a in message_data])
            self.open_movie(path)

    def import_message(self, message='', message_data=[]):
        if len(message_data) > 0:
            path = ' '.join([str(a) for a in message_data])
            self.import_movie(path)

    def play_message(self, message='', message_data=[]):
        if self.total_frames == 0:
            return
        self._parse_play_args(message_data)
        self.looping = False
        self.loop_checkbox.set(False)
        self.start_playback()
        self.play_pause_button.set_label('pause')
        self.play_pause_button.widget.set_active_theme(Node.active_theme_yellow)

    def loop_message(self, message='', message_data=[]):
        if self.total_frames == 0:
            return
        self._parse_play_args(message_data)
        self.looping = True
        self.loop_checkbox.set(True)
        self.start_playback()
        self.play_pause_button.set_label('pause')
        self.play_pause_button.widget.set_active_theme(Node.active_theme_yellow)

    def stop_message(self, message='', message_data=[]):
        self.stop_playback()
        self.current_frame = 0
        self.play_pause_button.set_label('play')
        self.play_pause_button.widget.set_active_theme(Node.active_theme_green)

    def pause_message(self, message='', message_data=[]):
        self.pause_playback()
        self.play_pause_button.set_label('resume')
        self.play_pause_button.widget.set_active_theme(Node.active_theme_green)

    def resume_message(self, message='', message_data=[]):
        self.resume_playback()
        self.play_pause_button.set_label('pause')
        self.play_pause_button.widget.set_active_theme(Node.active_theme_yellow)

    # ------------------------------------------------------------------
    # Widget callbacks
    # ------------------------------------------------------------------

    def loop_changed(self):
        self.looping = bool(self.loop_checkbox())

    def speed_changed(self):
        pass  # speed is read live in frame_task

    def clip_changed(self):
        pass  # clip bounds are read live in frame_task

    def clip_start_set(self):
        """Trigger button: set clip_start to current frame."""
        self.clip_start_input.set(self.current_frame)

    def clip_end_set(self):
        """Trigger button: set clip_end to current frame."""
        self.clip_end_input.set(self.current_frame)

    def play_button_clicked(self):
        """Play/pause/resume toggle following OpenTakeNode pattern."""
        if not self.playing and self.total_frames > 0:
            # Currently stopped or paused → start or resume
            label = self.play_pause_button.get_label()
            if label == 'play':
                # Fresh play from clip start
                self.looping = bool(self.loop_checkbox())
                self.start_playback()
            else:
                # Resume from paused position
                self.resume_playback()
            self.play_pause_button.set_label('pause')
            self.play_pause_button.widget.set_active_theme(Node.active_theme_yellow)
        else:
            # Currently playing → pause
            if self.playing:
                self.pause_playback()
                self.play_pause_button.set_label('resume')
                self.play_pause_button.widget.set_active_theme(Node.active_theme_green)

    def stop_button_clicked(self):
        """Stop playback, reset to frame 0, revert play button to 'play' state."""
        if self.playing or self.play_pause_button.get_label() == 'resume':
            self.stop_playback()
            self.current_frame = 0
            self.play_pause_button.set_label('play')
            self.play_pause_button.widget.set_active_theme(Node.active_theme_green)

    # ------------------------------------------------------------------
    # Execute (triggered by main input or frame input)
    # ------------------------------------------------------------------

    def execute(self):
        # Check which input triggered
        if self.active_input == self.frame_input:
            # Explicit frame input
            data = self.frame_input()
            if data is not None:
                try:
                    frame_num = int(any_to_int(data))
                    self.output_frame(frame_num)
                except (ValueError, TypeError):
                    pass
            return

        data = self.input()
        if data is None:
            return

        # Check for messages first (play, loop, stop, open, import, etc.)
        if self.check_for_messages(data):
            return

        # Int input → random frame access
        try:
            frame_num = int(any_to_int(data))
            self.output_frame(frame_num)
        except (ValueError, TypeError):
            pass

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_custom(self, container):
        container['movie_path'] = self.movie_path
        container['is_cached'] = self.cached_frames is not None

    def load_custom(self, container):
        path = container.get('movie_path', '')
        if path and os.path.exists(path):
            is_cached = container.get('is_cached', False)
            if is_cached:
                self.import_movie(path)
            else:
                self.open_movie(path)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        self.remove_frame_tasks()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cached_frames = None
