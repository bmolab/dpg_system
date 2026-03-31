import numpy as np
from dpg_system.node import Node, SaveDialog, LoadDialog
from dpg_system.conversion_utils import *
import cv2
import os
import json
import time
import threading
import dearpygui.dearpygui as dpg


def register_movie_nodes():
    Node.app.register_node('movie_player', MoviePlayerNode.factory)
    Node.app.register_node('movie_clip_dict', MovieClipDictNode.factory)


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
        self.last_time = 0.0

        # Read-ahead thread for streaming mode
        self._reader_thread = None
        self._reader_running = False
        self._reader_lock = threading.Lock()
        self._read_ahead_frame = None  # (frame_num, numpy_array)
        self._next_sequential_frame = 0  # next frame the reader will read

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
        self.clip_spec_output = self.add_output('clip_spec')

        # --- Save clip button ---
        self.save_clip_button = self.add_input('save clip', widget_type='button', callback=self.save_clip_pressed)

        # --- Message handlers ---
        self.message_handlers['open'] = self.open_message
        self.message_handlers['import'] = self.import_message
        self.message_handlers['play'] = self.play_message
        self.message_handlers['loop'] = self.loop_message
        self.message_handlers['stop'] = self.stop_message
        self.message_handlers['pause'] = self.pause_message
        self.message_handlers['resume'] = self.resume_message
        self.message_handlers['save_clip'] = self.save_clip_message

    def custom_create(self, from_file):
        # Set button colors matching OpenTakeNode style
        self.play_pause_button.widget.set_active_theme(Node.active_theme_green)
        self.play_pause_button.widget.set_height(24)
        self.stop_button.widget.set_active_theme(Node.active_theme_red)
        self.stop_button.widget.set_height(24)
        self.save_clip_button.widget.set_active_theme(Node.active_theme_blue)

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
            # Check if the read-ahead thread has this frame ready
            read_ahead = self._read_ahead_frame
            if read_ahead is not None and read_ahead[0] == frame_num:
                return read_ahead[1]
            # Fall back to direct read (random access or thread hasn't caught up)
            with self._reader_lock:
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
        self.last_time = time.perf_counter()
        self.playing = True

        # Start read-ahead thread for streaming mode
        if self.cached_frames is None and self.cap is not None:
            self._start_reader_thread(self.current_frame)

        self.add_frame_task()

    def stop_playback(self):
        """Stop playback completely."""
        if self.playing:
            self.playing = False
            self._stop_reader_thread()
            self.remove_frame_tasks()

    def pause_playback(self):
        """Pause playback (retain position)."""
        if self.playing:
            self.playing = False
            self._stop_reader_thread()
            self.remove_frame_tasks()

    def resume_playback(self):
        """Resume from current position."""
        if not self.playing and self.total_frames > 0:
            self.playing = True
            self.frame_accumulator = 0.0
            self.last_time = time.perf_counter()

            if self.cached_frames is None and self.cap is not None:
                self._start_reader_thread(self.current_frame)

            self.add_frame_task()

    # ------------------------------------------------------------------
    # Read-ahead thread (streaming mode only)
    # ------------------------------------------------------------------

    def _start_reader_thread(self, start_frame):
        """Start a background thread that reads frames sequentially."""
        self._stop_reader_thread()
        self._reader_running = True
        self._next_sequential_frame = start_frame
        self._read_ahead_frame = None
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def _stop_reader_thread(self):
        """Stop the background reader thread."""
        self._reader_running = False
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None

    def _reader_loop(self):
        """Background thread: reads frames sequentially from cv2.VideoCapture."""
        while self._reader_running:
            with self._reader_lock:
                if self.cap is None:
                    break

                target = self._next_sequential_frame
                if target >= self.total_frames:
                    # Will be handled by frame_task (loop/stop)
                    time.sleep(0.001)
                    continue

                # Seek if needed (e.g. after loop wrap)
                current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current_pos != target:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)

                ret, frame = self.cap.read()
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self._read_ahead_frame = (target, rgb_frame)
                    self._next_sequential_frame = target + 1

            # Sleep briefly to not spin-lock; the frame_task will advance _next_sequential_frame
            time.sleep(0.0005)

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

        # Time-based frame advancement
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now

        # Accumulate fractional frames based on real elapsed time and movie fps
        self.frame_accumulator += dt * self.fps * speed
        frames_to_advance = int(self.frame_accumulator)
        self.frame_accumulator -= frames_to_advance

        if frames_to_advance <= 0:
            return

        self.current_frame += frames_to_advance

        # Check if we've reached or passed the end
        if self.current_frame >= clip_end:
            if self.looping:
                overshoot = self.current_frame - clip_end
                self.current_frame = clip_start + (overshoot % (clip_end - clip_start))
                # Tell reader thread to seek to the new position
                self._next_sequential_frame = self.current_frame
                self.output_frame(self.current_frame)
                self.done_output.send('bang')
            else:
                self.current_frame = clip_end - 1
                self.output_frame(self.current_frame)
                self.done_output.send('bang')
                self.stop_playback()
                self.current_frame = 0
                self.play_pause_button.set_label('play')
                self.play_pause_button.widget.set_active_theme(Node.active_theme_green)
        else:
            # Tell reader thread where we are so it reads ahead
            self._next_sequential_frame = self.current_frame + 1
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

    def save_clip_pressed(self):
        """Save clip button: send current clip spec out the clip_spec output."""
        self._send_clip_spec()

    def save_clip_message(self, message='', message_data=[]):
        """Message handler for save_clip."""
        self._send_clip_spec()

    def _send_clip_spec(self):
        """Send [clip_start, clip_end, speed] out the clip_spec output."""
        clip_start = self._effective_clip_start()
        clip_end = self._effective_clip_end()
        speed = float(self.speed_input())
        self.clip_spec_output.send([clip_start, clip_end, speed])

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
        self._stop_reader_thread()
        self.remove_frame_tasks()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cached_frames = None


class MovieClipDictNode(Node):
    """
    Stores named movie clips as {name: [start_frame, end_frame, speed]}.

    - Connect clip_spec output of movie_player to 'clip_spec in'
    - Type a name and press 'store' to save the current clip spec
    - Select a clip from the list and press 'play' or 'loop' to send
      the command out (connect output to movie_player's input)
    - Save/load clip collections as JSON
    """

    @staticmethod
    def factory(name, data, args=None):
        node = MovieClipDictNode(name, data, args)
        return node

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.clips = {}  # {name: [start, end, speed]}
        self.current_clip_spec = None  # last received [start, end, speed]

        # --- Inputs ---
        self.input = self.add_input('input', triggers_execution=True)
        self.clip_spec_in = self.add_input('clip_spec in', callback=self.clip_spec_received)

        # --- Clip name entry ---
        self.clip_name_input = self.add_input('clip name', widget_type='text_input', default_value='')

        # --- Buttons ---
        self.store_button = self.add_input('store', widget_type='button', callback=self.store_clip)
        self.play_button = self.add_input('play', widget_type='button', callback=self.play_clip_pressed)
        self.loop_button = self.add_input('loop', widget_type='button', callback=self.loop_clip_pressed)
        self.delete_button = self.add_input('delete', widget_type='button', callback=self.delete_clip)

        # --- Clip list ---
        self.clip_listbox = self.add_property('clips', widget_type='list_box', width=200,
                                              callback=self.clip_selected)

        # --- Output ---
        self.command_output = self.add_output('command out')
        self.clip_spec_output = self.add_output('clip_spec out')

        # --- Save / Load ---
        self.save_button = self.add_input('save', widget_type='button', callback=self.save_clips)
        self.load_button_input = self.add_input('load', widget_type='button', callback=self.load_clips)
        self.file_label = self.add_label('')

        # --- Message handlers ---
        self.message_handlers['store'] = self.store_message
        self.message_handlers['play'] = self.play_clip_message
        self.message_handlers['loop'] = self.loop_clip_message
        self.message_handlers['delete'] = self.delete_message
        self.message_handlers['list'] = self.list_message
        self.message_handlers['select'] = self.select_message
        self.message_handlers['save'] = self.save_message
        self.message_handlers['load'] = self.load_message

    def custom_create(self, from_file):
        self.store_button.widget.set_active_theme(Node.active_theme_blue)
        self.store_button.widget.set_height(24)
        self.play_button.widget.set_active_theme(Node.active_theme_green)
        self.play_button.widget.set_height(24)
        self.loop_button.widget.set_active_theme(Node.active_theme_yellow)
        self.loop_button.widget.set_height(24)
        self.delete_button.widget.set_active_theme(Node.active_theme_red)
        self.delete_button.widget.set_height(24)
        self._update_listbox()

    # ------------------------------------------------------------------
    # Clip spec input
    # ------------------------------------------------------------------

    def clip_spec_received(self):
        """Receive [start, end, speed] from movie_player's clip_spec output."""
        data = self.clip_spec_in()
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            self.current_clip_spec = list(data)

    # ------------------------------------------------------------------
    # Store / Delete
    # ------------------------------------------------------------------

    def store_clip(self):
        """Store current clip spec under the name in the text input."""
        name = self.clip_name_input()
        if not name or name.strip() == '':
            print('MovieClipDict: please enter a clip name')
            return
        if self.current_clip_spec is None:
            print('MovieClipDict: no clip spec received yet')
            return
        self.clips[name] = list(self.current_clip_spec)
        self._update_listbox()
        print(f'MovieClipDict: stored "{name}" = {self.current_clip_spec}')

    def delete_clip(self):
        """Delete the currently selected clip."""
        name = self._selected_clip_name()
        if name and name in self.clips:
            del self.clips[name]
            self._update_listbox()

    # ------------------------------------------------------------------
    # Play / Loop
    # ------------------------------------------------------------------

    def _send_clip_command(self, command, clip_name=None):
        """Send a play or loop command for the named clip."""
        if clip_name is None:
            clip_name = self._selected_clip_name()
        if clip_name and clip_name in self.clips:
            spec = self.clips[clip_name]
            start = int(spec[0])
            end = int(spec[1])
            speed = float(spec[2]) if len(spec) > 2 else 1.0
            self.command_output.send(f'{command} {start} {end} {speed}')
            self.clip_spec_output.send(spec)

    def play_clip_pressed(self):
        self._send_clip_command('play')

    def loop_clip_pressed(self):
        self._send_clip_command('loop')

    # ------------------------------------------------------------------
    # Listbox management
    # ------------------------------------------------------------------

    def _update_listbox(self):
        """Refresh the listbox with current clip names."""
        names = list(self.clips.keys())
        try:
            dpg.configure_item(self.clip_listbox.widget.uuid, items=names)
            if len(names) > 0:
                dpg.set_value(self.clip_listbox.widget.uuid, names[0])
        except Exception:
            pass

    def _selected_clip_name(self):
        """Get the currently selected clip name from the listbox."""
        try:
            return dpg.get_value(self.clip_listbox.widget.uuid)
        except Exception:
            return None

    def clip_selected(self):
        """Called when user clicks a clip in the listbox."""
        name = self._selected_clip_name()
        if name and name in self.clips:
            self.clip_name_input.set(name)
            self.clip_spec_output.send(self.clips[name])

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def store_message(self, message='', message_data=[]):
        """store <name> [start] [end] [speed] — store clip by name."""
        if len(message_data) >= 1:
            name = str(message_data[0])
            if len(message_data) >= 3:
                # Explicit spec provided
                start = int(message_data[1])
                end = int(message_data[2])
                speed = float(message_data[3]) if len(message_data) > 3 else 1.0
                self.clips[name] = [start, end, speed]
            elif self.current_clip_spec is not None:
                self.clips[name] = list(self.current_clip_spec)
            self._update_listbox()

    def play_clip_message(self, message='', message_data=[]):
        """play <name> — recall and play a clip."""
        if len(message_data) >= 1:
            name = str(message_data[0])
            self._send_clip_command('play', name)

    def select_message(self, message='', message_data=[]):
        """select <name> — select a clip and output its spec (no play/loop)."""
        if len(message_data) >= 1:
            name = str(message_data[0])
            if name in self.clips:
                self.clip_name_input.set(name)
                try:
                    dpg.set_value(self.clip_listbox.widget.uuid, name)
                except Exception:
                    pass
                self.clip_spec_output.send(self.clips[name])

    def loop_clip_message(self, message='', message_data=[]):
        """loop <name> — recall and loop a clip."""
        if len(message_data) >= 1:
            name = str(message_data[0])
            self._send_clip_command('loop', name)

    def delete_message(self, message='', message_data=[]):
        """delete <name> — remove a clip."""
        if len(message_data) >= 1:
            name = str(message_data[0])
            if name in self.clips:
                del self.clips[name]
                self._update_listbox()

    def list_message(self, message='', message_data=[]):
        """list — send all clip names out."""
        names = list(self.clips.keys())
        self.command_output.send(names)

    def save_message(self, message='', message_data=[]):
        if len(message_data) > 0:
            self._save_to_file(str(message_data[0]))
        else:
            self.save_clips()

    def load_message(self, message='', message_data=[]):
        if len(message_data) > 0:
            self._load_from_file(str(message_data[0]))
        else:
            self.load_clips()

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self):
        data = self.input()
        if data is None:
            return
        if self.check_for_messages(data):
            return
        # If a string clip name is sent, play it
        if isinstance(data, str) and data in self.clips:
            self._send_clip_command('play', data)

    # ------------------------------------------------------------------
    # Save / Load JSON
    # ------------------------------------------------------------------

    def save_clips(self):
        arg = self.save_button()
        if isinstance(arg, str) and arg != '':
            self._save_to_file(arg)
            return
        SaveDialog(self, self._save_dialog_callback, extensions=['.json'])

    def _save_dialog_callback(self, path):
        if path:
            self._save_to_file(path)

    def _save_to_file(self, path):
        with open(path, 'w') as f:
            json.dump(self.clips, f, indent=4)
        self.file_label.set(os.path.basename(path))
        print(f'MovieClipDict: saved {len(self.clips)} clips to {path}')

    def load_clips(self):
        arg = self.load_button_input()
        if isinstance(arg, str) and arg != '' and os.path.exists(arg):
            self._load_from_file(arg)
            return
        LoadDialog(self, self._load_dialog_callback, extensions=['.json'])

    def _load_dialog_callback(self, path):
        if path:
            self._load_from_file(path)

    def _load_from_file(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.clips = json.load(f)
            self._update_listbox()
            self.file_label.set(os.path.basename(path))
            print(f'MovieClipDict: loaded {len(self.clips)} clips from {path}')

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_custom(self, container):
        container['clips'] = self.clips

    def load_custom(self, container):
        if 'clips' in container:
            self.clips = container['clips']
            self._update_listbox()
