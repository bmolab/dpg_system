import numpy as np
import threading
import queue
import sounddevice as sd
import os
try:
    import torchaudio
except ImportError:
    torchaudio = None

class Sample:
    def __init__(self, data, volume=1.0, loop=False, loop_start=0, loop_end=-1, crossfade_frames=0, pitch=1.0, sample_start=0, sample_end=-1):
        # Allow passing path or array
        if isinstance(data, str):
            self.data = self._load_file(data)
        else:
            self.data = data

        self.default_volume = volume
        self.loop = loop
        self.loop_start = loop_start

        # If loop_end is -1 or greater than length, set to length
        if loop_end < 0 or loop_end > len(self.data):
            self.loop_end = len(self.data)
        else:
            self.loop_end = loop_end
        
        # Ensure loop_end is at least loop_start
        self.loop_end = max(self.loop_end, self.loop_start + 1)


        self.sample_start = sample_start
        if sample_end < 0 or sample_end > len(self.data):
             self.sample_end = len(self.data)
        else:
             self.sample_end = max(sample_end, sample_start + 1)
             
        self.crossfade_frames = min(crossfade_frames, self.loop_end - self.loop_start)
        self.default_pitch = pitch
    
    def _load_file(self, filepath):
        if not os.path.exists(filepath):
            # Return silence or throw? Let's return a small silent buffer to avoid crashes
            print(f"File not found: {filepath}")
            return np.zeros((1024, 2), dtype=np.float32)
        try:
            waveform, sample_rate = torchaudio.load(filepath)
            if not waveform.is_cpu:
                waveform = waveform.cpu()
            # Convert (C, N) to (N, C)
            arr = waveform.numpy().T
            return arr
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return np.zeros((1024, 2), dtype=np.float32)

class Grain:
    def __init__(self, start_sample_idx, duration_samples, pitch, amp, pan=0.5):
        self.start_idx = start_sample_idx # Index in source sample
        self.current_idx = float(start_sample_idx)
        self.duration = duration_samples
        self.age = 0
        self.pitch = pitch
        self.amp = amp
        self.pan = pan
        self.active = True

class Voice:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.active = False
        self.sample = None
        self.position = 0.0
        self.looping = False
        
        # Gain / Envelope
        self.current_gain = 0.0
        self.target_gain = 0.0
        self.envelope_val = 0.0
        self.target_envelope = 0.0
        
        # Pitch
        self.current_pitch = 1.0
        self.target_pitch = 1.0
        
        # Envelope Params
        self.attack_s = 0.0
        self.decay_s = 0.0
        self.decay_curve = 1.0
        
        # Loop State
        self.loop_start = 0
        self.loop_end = 0
        self.play_start = 0
        self.play_end = 0
        self.crossfade_frames = 0
            # Command Queue for Thread Safety
        self._command_queue = queue.SimpleQueue()
        
        # Granular State
        self.granular_mode = False
        self.grains = [] # List of active Grain objects
        self.time_since_last_grain = 0.0
        
        # Granular Params
        self.grain_density = 10.0 # Grains per second
        self.grain_dur = 0.1 # seconds
        self.grain_pos = 0.0 # 0..1 normalized position in sample
        self.grain_pitch = 1.0
        self.grain_jitter_pos = 0.0
        self.grain_jitter_pitch = 0.0
        self.grain_jitter_dur = 0.0
        self.grain_env_shape = "hann"
        
        # Scratch Params
        self.scratch_target = 0.0
        self.scratch_max_vel = 1.0
        self.scratch_accel = 1.0
        self.scratch_mode = False
    
    # --- Public API (Main Thread) ---
    
    def trigger(self, sample, volume=None, pitch=None, mode=None):
        """Queue a trigger command."""
        self._command_queue.put({
            "type": "trigger",
            "sample": sample,
            "volume": volume,
            "pitch": pitch,
            "mode": mode
        })

    def release(self):
        """Queue a release command."""
        self._command_queue.put({"type": "release"})

    def set_volume(self, volume):
        self._command_queue.put({"type": "set_param", "param": "volume", "value": volume})

    def set_pitch(self, pitch):
        self._command_queue.put({"type": "set_param", "param": "pitch", "value": pitch})
        
    def set_envelope(self, attack, decay, curve=1.0):
        self._command_queue.put({
            "type": "set_envelope",
            "attack": attack,
            "decay": decay,
            "curve": curve
        })

    def set_loop_window(self, loop, loop_start, loop_end, crossfade_frames):
        self._command_queue.put({
            "type": "set_loop",
            "loop": loop,
            "start": loop_start,
            "end": loop_end,
            "crossfade": crossfade_frames
        })

    def set_playback_range(self, start, end):
        self._command_queue.put({
            "type": "set_range",
            "start": start,
            "end": end
        })

    def set_granular_params(self, density, duration, jitter_pos, jitter_pitch, jitter_dur):
        self._command_queue.put({
            "type": "set_granular",
            "density": density,
            "dur": duration,
            "j_pos": jitter_pos,
            "j_pitch": jitter_pitch,
            "j_dur": jitter_dur
        })

    def set_grain_position(self, pos):
        self._command_queue.put({"type": "set_grain_pos", "pos": pos})

    def set_scratch_target(self, pos):
        self._command_queue.put({"type": "set_scratch_target", "pos": pos})

    def set_scratch_params(self, max_vel, accel):
        self._command_queue.put({"type": "set_scratch_params", "max_vel": max_vel, "accel": accel})
        
    def set_mode(self, mode):
        # mode: 'normal' or 'granular'
        self._command_queue.put({"type": "set_mode", "mode": mode})
        
    # --- Audio Thread Processing ---

    def _process_commands(self):
        # print("DEBUG: Processing commands...", flush=True)
        while not self._command_queue.empty():
            # print("DEBUG: Getting command...", flush=True)
            try:
                cmd = self._command_queue.get_nowait()
            except queue.Empty:
                break
            # print(f"DEBUG: Got command {cmd['type']}", flush=True)
            t = cmd["type"]
            
            if t == "trigger":
                if "mode" in cmd and cmd["mode"] is not None:
                     self.granular_mode = (cmd["mode"] == "granular")
                     self.scratch_mode = (cmd["mode"] == "scratch")

                sample = cmd["sample"]
                self.sample = sample
                self.looping = sample.loop
                
                target_vol = cmd["volume"] if cmd["volume"] is not None else sample.default_volume
                self.target_gain = target_vol
                
                # Reset envelope if re-triggering hard, or legacy behavior
                if not self.active:
                    self.current_gain = self.target_gain
                    self.envelope_val = 0.0 if self.attack_s > 0 else 1.0
                
                self.target_envelope = 1.0
                self.active = True
                
                p = cmd["pitch"] if cmd["pitch"] is not None else sample.default_pitch
                self.target_pitch = p
                # Snap pitch on trigger? Or glide? usually snap
                self.current_pitch = p 
                self.prev_pitch = p 
                
                self.loop_start = sample.loop_start
                self.loop_end = sample.loop_end
                self.crossfade_frames = sample.crossfade_frames
                self.loop_start = sample.loop_start
                self.loop_end = sample.loop_end
                
                # Sanity check loop
                if self.loop_end > len(self.sample.data): self.loop_end = len(self.sample.data)
                if self.loop_start < 0: self.loop_start = 0
                if self.loop_start >= self.loop_end: 
                    self.loop_start = 0
                    self.loop_end = len(self.sample.data)

                self.crossfade_frames = sample.crossfade_frames
                
                # Sanity check crossfade
                loop_len = self.loop_end - self.loop_start
                if self.crossfade_frames > loop_len: self.crossfade_frames = loop_len
                if self.crossfade_frames < 0: self.crossfade_frames = 0
                if self.loop_end - self.crossfade_frames < 0:
                     self.crossfade_frames = self.loop_end

                self.play_start = sample.sample_start
                self.play_end = sample.sample_end
                
                # Sanity check play range
                if self.play_end > len(self.sample.data): self.play_end = len(self.sample.data)
                if self.play_start < 0: self.play_start = 0
                
                # Clamp position
                self.position = float(self.play_start)
                
                # Reset Granular State
                self.grains = []
                self.time_since_last_grain = 0.0
                
            elif t == "release":
                self.target_envelope = 0.0
                
            elif t == "set_param":
                if cmd["param"] == "volume":
                    self.target_gain = cmd["value"]
                elif cmd["param"] == "pitch":
                    self.target_pitch = cmd["value"]
                    
            elif t == "set_envelope":
                self.attack_s = cmd["attack"]
                self.decay_s = cmd["decay"]
                self.decay_curve = cmd["curve"]
                
            elif t == "set_loop":
                self.looping = cmd["loop"]
                self.loop_start = cmd["start"]
                self.loop_end = cmd["end"]
                self.crossfade_frames = cmd["crossfade"]
                
                # Sanity check loop
                if self.sample:
                     if self.loop_end > len(self.sample.data): self.loop_end = len(self.sample.data)
                     if self.loop_start < 0: self.loop_start = 0
                     if self.loop_start >= self.loop_end: 
                         self.loop_start = 0
                         self.loop_end = len(self.sample.data)

                     loop_len = self.loop_end - self.loop_start
                     if self.crossfade_frames > loop_len: self.crossfade_frames = loop_len
                     if self.crossfade_frames < 0: self.crossfade_frames = 0
                     if self.loop_end - self.crossfade_frames < 0:
                         self.crossfade_frames = self.loop_end
            
            elif t == "set_range":
                self.play_start = cmd["start"]
                self.play_end = cmd["end"]
                # If current position is out of new range?
                # Usually we let it play until loop or end, but if strictly enforcing:
                # if self.position < self.play_start: self.position = float(self.play_start)
                # if self.position >= self.play_end: ... handle later
                
            elif t == "set_granular":
                self.grain_density = cmd["density"]
                self.grain_dur = cmd["dur"]
                self.grain_jitter_pos = cmd["j_pos"]
                self.grain_jitter_pitch = cmd["j_pitch"]
                self.grain_jitter_dur = cmd["j_dur"]
                
            elif t == "set_grain_pos":
                self.grain_pos = cmd["pos"]

            elif t == "set_scratch_target":
                self.scratch_target = cmd["pos"]

            elif t == "set_scratch_params":
                self.scratch_max_vel = cmd["max_vel"]
                self.scratch_accel = cmd["accel"]
                
            elif t == "set_mode":
                self.granular_mode = (cmd["mode"] == "granular")
                self.scratch_mode = (cmd["mode"] == "scratch")
    
    def _process_granular(self, frames, channels):
        if self.sample is None: 
            return np.zeros((frames, channels), dtype=np.float32)

        out = np.zeros((frames, channels), dtype=np.float32)
        
        # 1. Spawn Grains
        samples_per_second = self.sample_rate
        
        # --- Envelope Logic (Granular needs to respect release) ---
        # Duplicate/Refactor simple envelope logic here or just use current_gain/target_envelope?
        # release() sets target_envelope = 0.0.
        # trigger() set target_envelope = 1.0.
        # We need to smooth self.envelope_val towards self.target_envelope.
        
        start_env = self.envelope_val
        target = self.target_envelope
        
        env_curve = np.ones(frames, dtype=np.float32) * start_env
        
        # Simple Linear envelope moving logic per block
        # For granular active voices, usually attack is instant or ignored, but release is key.
        # Let's effectively use a "release time" if defined, or default short release.
        # self.decay_s is used for release in standard mode (simplification).
        
        release_s = max(0.05, self.decay_s) # Min release time
        release_step = 1.0 / (release_s * self.sample_rate)

        # For Attack
        # If attack_s is 0, we jump instantly? 
        # But we need smooth transition if moving from 0.0 to 1.0
        # If attack_s is 0, make it very fast (e.g. 1 sample or instant)
        if self.attack_s <= 0.001:
             attack_step = 1.0
        else:
             attack_step = 1.0 / (self.attack_s * self.sample_rate)
        
        if target < start_env: # Releasing
            # Linear step using release_step
            steps = np.arange(1, frames + 1, dtype=np.float32) * release_step
            
            # ... (Existing Decay Curve Logic using release_s/coeff) ...
            # Wait, my previous edit for Decay Curve logic used multiplicative coeff derived from release_s.
            # I must preserve that.
            
            if target == 0.0:
                 eff_release = release_s / self.decay_curve
                 if eff_release < 0.001: eff_release = 0.001
                 n_samples = eff_release * self.sample_rate
                 coeff = np.power(0.01, 1.0/n_samples)
                 decay_curve_block = np.power(coeff, np.arange(1, frames+1))
                 env_curve = np.maximum(start_env * decay_curve_block, 0.0)
                 self.envelope_val = env_curve[-1]
            else:
                 env_curve = np.maximum(start_env - steps, target)
                 self.envelope_val = env_curve[-1]

            # If we hit 0 and target is 0, de-activate
            if self.envelope_val <= 0.001 and target == 0.0:
                self.active = False
                self.envelope_val = 0.0
                
        elif target > start_env: # Attacking
             # Use attack_step
             steps = np.arange(1, frames + 1, dtype=np.float32) * attack_step
             env_curve = np.minimum(start_env + steps, target)
             self.envelope_val = env_curve[-1]
        else:
             env_curve[:] = start_env

        # --- Spawning Logic ---
        # Only spawn if envelope > 0 (or we are active)
        if self.grain_density > 0 and self.envelope_val > 0:
            samples_per_grain = samples_per_second / self.grain_density
            self.time_since_last_grain += frames
            
            # How many grains to spawn this block?
            # Basic periodic spawning with jitter could be implemented here.
            # Simplified: spawn if time > threshold
            while self.time_since_last_grain >= samples_per_grain:
                self.time_since_last_grain -= samples_per_grain
                
                # Create Grain
                # Position Jitter
                pos_jit = 0.0
                if self.grain_jitter_pos > 0.0001:
                    pos_jit = (np.random.random() - 0.5) * 2.0 * self.grain_jitter_pos
                
                # Wrap position to 0..1 to handle inputs > 1.0 or < 0.0
                raw_pos = self.grain_pos + pos_jit
                actual_pos = raw_pos - np.floor(raw_pos)
                
                # Map actual_pos (0..1) to play_start..play_end
                p_start = self.play_start
                p_end = self.play_end
                # Safety checks
                if p_end < 0 or p_end > len(self.sample.data): p_end = len(self.sample.data)
                if p_start < 0: p_start = 0
                if p_start >= p_end: 
                     p_start = 0
                     p_end = len(self.sample.data)

                play_len = p_end - p_start
                start_idx = p_start + int(actual_pos * play_len)
                
                # Pitch Jitter
                actual_pitch = self.target_pitch * self.grain_pitch
                if self.grain_jitter_pitch > 0.0001:
                    pitch_jit = (np.random.random() - 0.5) * 2.0 * self.grain_jitter_pitch
                    actual_pitch *= (1.0 + pitch_jit)
                if actual_pitch < 0.01: actual_pitch = 0.01
                
                # Duration Jitter
                actual_dur_s = self.grain_dur
                if self.grain_jitter_dur > 0.0001:
                    dur_jit = (np.random.random() - 0.5) * 2.0 * self.grain_jitter_dur
                    actual_dur_s = max(0.001, self.grain_dur + dur_jit)
                
                dur_samples = int(actual_dur_s * self.sample_rate)
                
                # Amp (inherited from Voice target gain for now)
                # We apply master gain to grains? 
                # Or do we apply envelope to the OUTPUT?
                # Applying current master envelope to new grains only means old grains keep ringing (natural release).
                # Applying master envelope to OUTPUT means hard choke.
                # Standard release: Stop Spawning, let ring OR fade out everything.
                # User expects "stop". Usually fade out.
                # Let's apply target_gain to grain amplitude (volume slider).
                # And apply the valid 'env_curve' to the mixed output?
                # or spawn amplitude?
                
                # If we apply to spawn amplitude only: existing clouds will ring out. Natural.
                # But 'active=False' might kill it prematurely in nodes?
                # If active=False, node drops voice.
                # We need to ensure we don't return 0 until grains die?
                
                # Decision: Apply volume (sliders) to Grain Amp.
                # Apply Release Envelope to Grain Amp spawning? 
                # YES -> If releasing, new grains are quieter? No that's weird.
                # Standard poly synth: Envelope applies to the whole voice sum.
                
                amp = self.target_gain 
                
                g = Grain(start_idx, dur_samples, actual_pitch, amp)
                self.grains.append(g)
                
        # 2. Process Grains
        active_grains = []
        sample_data = self.sample.data
        slen = len(sample_data)
        
        # We need to mix grains.
        # Vectorizing this is hard because each grain has different pitch/pos.
        # We will iterate grains. For high density, this might be slow in Python.
        # Optimization: use small blocks or C++ extension (not available).
        # We will try to rely on numpy slicing where possible.
        
        # Pre-calculate linear ramp for volume smoothing if needed, 
        # but grains are usually short enough that per-grain env is key.
        
        for g in self.grains:
            if not g.active: continue
            
            # Calculate how many output frames we can generate from this grain
            # This depends on grain duration remaining vs block size
            
            remaining = g.duration - g.age
            if remaining <= 0:
                continue
                
            n_frames = min(frames, remaining)
            
            # Simple Playback: linear interp lookups based on pitch
            pitch = g.pitch
            
            # Indices in grain timeline (0..n_frames)
            t_grain = np.arange(n_frames, dtype=np.float32)
            
            # Indices in sample
            # current_idx + (0..n)*pitch
            sample_indices = g.current_idx + t_grain * pitch
            
            # Wrap or clamp? Grains usually clamp or silence at end.
            # Let's wrap for loop-like texture or just clamp?
            # Standard granular often plays past loop points or silence.
            # We will use modulo if we want "loop" mode, but usually standard sample playback.
            # Safe read:
            
            # Integer parts and alpha
            idx_int = sample_indices.astype(int)
            alpha = sample_indices - idx_int
            if sample_data.ndim > 1: alpha = alpha[:, np.newaxis] # broadcast for stereo
            
            # Check bounds
            # Efficient masking is hard with wrapping.
            # Let's clamp to safe range, fade out at edges if needed?
            # Creating a mask for valid indices
            valid_mask = (idx_int >= 0) & (idx_int < slen - 1)
            
            # If pitch is high, valid_mask might be sparse.
            # If we assume sample is long enough...
            
            # Fetch samples - use clip to avoid crash, mask later
            idx_safe = np.clip(idx_int, 0, slen - 2)
            s0 = sample_data[idx_safe]
            s1 = sample_data[idx_safe + 1]
            
            chunk = s0 * (1.0 - alpha) + s1 * alpha
            
            # Apply mask (silence OOB)
            valid_mask_bc = valid_mask
            if sample_data.ndim > 1: valid_mask_bc = valid_mask[:, np.newaxis]
            chunk *= valid_mask_bc
            
            # Grain Envelope (Hann window usually)
            # Global time for this chunk: g.age .. g.age + n_frames
            # Normalized time: age / duration
            t_norm = (g.age + t_grain) / g.duration
            
            # Hann: 0.5 * (1 - cos(2*pi*t))
            # optimization: pre-calc window? or just compute
            env = 0.5 * (1.0 - np.cos(2.0 * np.pi * t_norm))
            
            if sample_data.ndim > 1: env = env[:, np.newaxis]
            
            chunk *= env
            chunk *= g.amp
            
            # Add to output
            # If grain is shorter than frame?, n_frames handles it.
            # We add to out[0:n_frames]
            out[:n_frames] += chunk
            
            # Update Grain State
            g.age += n_frames
            g.current_idx += n_frames * pitch
            
            if g.age < g.duration:
                active_grains.append(g)
        
        self.grains = active_grains
        
        # Apply Master Envelope to Output
        # Broadcast env_curve across channels
        if channels > 1:
            out *= env_curve[:, np.newaxis]
        else:
            out *= env_curve
            
        return out

    def process(self, frames, channels):
        # 1. Process Commands
        self._process_commands()
        
        if self.granular_mode:
            return self._process_granular(frames, channels)

        if self.scratch_mode and self.active:
            # Scratch Physics
            dist_s = (self.scratch_target - self.position) / self.sample_rate
            sign = 1.0 if dist_s >= 0 else -1.0
            abs_dist = abs(dist_s)
            
            v_allow = np.sqrt(2 * self.scratch_accel * abs_dist)
            target_v = min(v_allow, self.scratch_max_vel) * sign
            
            # Use current_pitch as start velocity (state from end of last block)
            current_v = self.current_pitch
            
            dt = frames / self.sample_rate
            max_change = self.scratch_accel * dt
            
            if current_v < target_v:
                new_v = min(current_v + max_change, target_v)
            else:
                new_v = max(current_v - max_change, target_v)
                
            self.target_pitch = new_v
        
        # ... existing code ...
        
        # inside loop
        # ...


        if not self.active or self.sample is None:
            return np.zeros((frames, channels), dtype=np.float32)

        # Auto-Release Check
        # If not looping and we are active (not already releasing)
        if not self.looping and self.target_envelope > 0.0:
            remaining_frames = self.play_end - self.position
            # Account for pitch?
            # If pitch > 1.0, we traverse samples faster, so we need "more" remaining frames?
            # No, if pitch=2.0, we eat 2 samples per output frame.
            # Decay time is in Seconds (output time).
            # Output frames needed for decay: decay_s * sample_rate
            # Input samples needed for that many output frames: (decay_s * sample_rate) * pitch
            
            output_decay_frames = self.decay_s * self.sample_rate
            input_decay_samples = output_decay_frames * self.current_pitch
            
            if remaining_frames <= input_decay_samples:
                self.target_envelope = 0.0

        # 2. Update Gain (Linear Smoothing)
        if self.current_gain != self.target_gain:
            gain_curve = np.linspace(self.current_gain, self.target_gain, frames, dtype=np.float32)
            self.current_gain = self.target_gain
        else:
            gain_curve = self.current_gain

        # 3. Update Pitch (Linear Smoothing)
        if abs(self.current_pitch - self.target_pitch) > 0.0001:
            pitch_curve = np.linspace(self.current_pitch, self.target_pitch, frames, dtype=np.float32)
            self.current_pitch = self.target_pitch
        else:
            pitch_curve = np.full(frames, self.current_pitch, dtype=np.float32)

        # 4. Update Envelope (Attack/Decay)
        # Using a simpler state approach for block processing
        env_curve = np.empty(frames, dtype=np.float32)
        
        start_env = self.envelope_val
        target = self.target_envelope
        
        # If we need to execute attack/decay logic per sample or linearly over block?
        # Linear over block is standard for control rate, but lets try to be accurate to physics
        
        curr = start_env
        
        # Calculate rates
        attack_step = 0.0
        if self.attack_s > 0:
            attack_step = 1.0 / (self.attack_s * self.sample_rate)
            
        decay_step = 0.0
        if self.decay_s > 0:
            decay_step = 1.0 / (self.decay_s * self.sample_rate)
            
        # Unroll a bit for numpy speed? Or just use linspace if monotonic?
        # Since target doesn't change mid-block (commands processed at start), the direction is constant
        
        if target > curr: # Attacking
            if attack_step > 0:
                steps = np.arange(1, frames + 1, dtype=np.float32) * attack_step
                env_curve = np.minimum(curr + steps, 1.0)
                self.envelope_val = env_curve[-1]
            else:
                env_curve[:] = 1.0
                self.envelope_val = 1.0
        elif target < curr: # Decaying
            if decay_step > 0:
                steps = np.arange(1, frames + 1, dtype=np.float32) * decay_step
                env_curve = np.maximum(curr - steps, 0.0)
                self.envelope_val = env_curve[-1]
            else:
                env_curve[:] = 0.0
                self.envelope_val = 0.0
        else:
            env_curve[:] = curr

        # Apply built-in curve to decay only? 
        # The user requested decay curve.
        # If active (target=1.0), we are in attack or sustain.
        # If releasing (target=0.0), we are in decay.
        if self.target_envelope == 0.0:
             # Apply power curve
             if self.decay_curve != 1.0:
                 np.power(env_curve, self.decay_curve, out=env_curve)

        # Check for silence end
        if self.target_envelope == 0.0 and self.envelope_val <= 0.0001:
            self.active = False
            # Check if whole block is practically silent
            if np.all(env_curve < 0.0001):
                return np.zeros((frames, channels), dtype=np.float32)

        # 5. Render Audio
        # We need to render with variable pitch.
        # pitch_curve tells us the step size per sample.
        
        # We can implement a simplified loop here that handles variable pitch.
        # Or just use the average pitch for chunking if variation is small?
        # "Zipper noise" comes from step changes. Linear ramp solves it.
        # But complex resampling with per-sample pitch is expensive in python.
        # We will iterate in chunks where pitch is roughly constant, 
        # OR we can just use the per-sample index additions.
        
        output = np.zeros((frames, channels), dtype=np.float32)
        sample_data = self.sample.data
        
        # We can't easily vectorise "variable stride" accumulation purely with simple numpy indexing
        # without `add.accumulate`.
        
        # Calculate sample indices
        # current_pos + cumsum(pitch_curve)
        
        pitch_cumsum = np.cumsum(pitch_curve)
        sample_indices = self.position + pitch_cumsum
        
        # Make sure we update position for next block
        self.position = sample_indices[-1]
        
        # Now we have the raw indices into the sample buffer.
        # We need to handle looping and wrapping.
        # This is the tricky part with vectorized wrapping.
        
        # Since handling wrapping for *every* sample in a vectorized way with variable pitch and arbitrary loop points is complex,
        # we might fallback to the chunked approach if pitch is constant, or accept a bit of slowness.
        # However, the user snippet used a `while` loop with chunks. This is good.
        # Let's adapt that checks logic but use `pitch_curve`.
        
        # Optimization: If pitch is constant (often), used simple chunking.
        # If pitch is changing, it's a transient frame.
        
            
        # RESET position tracking for the loop
        current_pos = self.position - pitch_cumsum[-1] # Start of block
        
        processed = 0
        out_buf = np.zeros((frames, channels), dtype=np.float32)
        
        while processed < frames:
            # Determine pitch for this segment (use average or start?)


            # Determines pitch for this segment
            # If we are ramping, pitch is `pitch_curve[processed]`.
            
            p = pitch_curve[processed]
            # Avoid division by zero
            if abs(p) < 0.0001: p = 0.0001 # Or handle 0 explicitly?
            
            # Determine effective end for this segment
            limit = self.play_end
            start_limit = self.play_start
            
            if self.looping:
                 # If looping, we wrap at loop_end / loop_start
                 limit = len(sample_data) 
            
            effective_end = self.loop_end if self.looping else limit
            effective_start = self.loop_start if self.looping else start_limit
            
            dist = 0
            
            if p > 0:
                dist = effective_end - current_pos
                if dist <= 0:
                     if self.looping:
                         # Wrap
                         current_pos = float(self.loop_start) + (current_pos - self.loop_end)
                         current_pos = max(float(self.loop_start), min(current_pos, float(self.loop_end)-0.001))
                         dist = effective_end - current_pos
                     elif self.scratch_mode:
                         # Park at end
                         current_pos = float(effective_end - 0.001)
                         dist = 0.0 # Force re-eval or wait?
                         # If p > 0 and we are at end, we can't move.
                         # We must render constant?
                         # Just set chunk len?
                         # If dist <= 0, we can't play forward.
                         # We just fill output with last sample?
                         # Or just break loop if not changing?
                         pass
                     else:
                         self.active = False
                         break
            else:
                # Playing backwards
                dist = current_pos - effective_start
                if dist <= 0:
                     if self.looping:
                         # Wrap (backwards passes start -> goes to end)
                         current_pos = float(self.loop_end) - (self.loop_start - current_pos)
                         current_pos = max(float(self.loop_start), min(current_pos, float(self.loop_end)-0.001))
                         dist = current_pos - effective_start
                     elif self.scratch_mode:
                         # Park at start
                         current_pos = float(effective_start)
                     else:
                         self.active = False
                         break
            
            # How many frames fit in this dist?
            # frames * |p| = samples
            
            if abs(p) < 0.0001:
                # p is nearly 0. We hold position.
                chunk_len = frames - processed
            else:
                # If scratch parked at bound, dist might be <= 0 still.
                if self.scratch_mode and dist <= 0.001:
                     chunk_len = frames - processed
                     # We force indices to current_pos
                else:
                    chunk_len = int(abs(dist) / abs(p))
            
            chunk_len = max(1, chunk_len)
            chunk_len = min(frames - processed, chunk_len)
            
            # Normalize p_slice logic for bidirectional
            # p_slice is signed.
            p_slice = pitch_curve[processed : processed + chunk_len]
            
            # valid_indices calculation
            # cumsum works for negative p too.
            valid_indices = np.concatenate(([0.0], np.cumsum(p_slice)[:-1])) + current_pos
            
            # Check bounds for safety (overshoot prevention)
            if p > 0:
                overshoot = valid_indices >= effective_end
                if np.any(overshoot):
                    cut = np.argmax(overshoot)
                    if cut == 0:
                        if self.scratch_mode:
                             chunk_len = frames - processed
                             valid_indices = np.full(chunk_len, effective_end - 0.001)
                             idx_int = valid_indices.astype(int)
                             alpha = valid_indices - idx_int
                             # Skip interpolation calc
                             s0 = sample_data[idx_int]
                             chunk_out = s0 # Hold value
                             # ... buffer copy ...
                             # We need to restructure render block to handle this "hold" case cleanly
                             # Or just clamp valid_indices?
                             valid_indices = np.minimum(valid_indices, effective_end - 1.0)
                        else:
                             chunk_len = 0 # Force wrap logic next time
                             valid_indices = valid_indices[:0]
                    else:
                        chunk_len = cut
                        valid_indices = valid_indices[:cut]
                        p_slice = p_slice[:cut]

            else: # p <= 0
                overshoot = valid_indices < effective_start
                if np.any(overshoot):
                    cut = np.argmax(overshoot)
                    if cut == 0:
                        if self.scratch_mode:
                             valid_indices = np.maximum(valid_indices, effective_start)
                        else:
                             chunk_len = 0
                             valid_indices = valid_indices[:0]
                    else:
                        chunk_len = cut
                        valid_indices = valid_indices[:cut]
                        p_slice = p_slice[:cut]

            idx_int = valid_indices.astype(int)
             
            # ... interpolation logic needs to handle idx+1 for bounds
            # If p<0, idx can go down. S0 is correct.
            # Interpolation: s0*(1-alpha) + s1*alpha
            # If p<0, we are moving from idx down.
            # alpha is negative? No, alpha = val - int(val). Always [0, 1).
            # e.g. val=10.5. int=10. alpha=0.5.
            # val=10.4. int=10. alpha=0.4.
            # val=9.9. int=9. alpha=0.9.
            # Interpolation uses S0 and S1 (idx+1).
            # If val=9.9, we want between S9 and S10. Correct.
            
            # Boundary handling for S1 (idx+1)
            # If idx=len-1, idx+1 is out.
            
            # Bounds check clamp
            # If scratch mode, we clamped to limits.
            
            idx_int = np.clip(idx_int, 0, len(sample_data)-1)
            
            alpha = valid_indices - idx_int
            if sample_data.ndim > 1: alpha = alpha[:, np.newaxis]

            s0 = sample_data[idx_int]
            
            idx_next = idx_int + 1
            if self.looping:
                 mask_loop = idx_next >= self.loop_end
                 idx_next[mask_loop] = self.loop_start
            else:
                 idx_next = np.minimum(idx_next, len(sample_data)-1)
            
            s1 = sample_data[idx_next]
            
            chunk_out = s0 * (1.0 - alpha) + s1 * alpha
            
            # Crossfade Logic
            if self.looping and self.crossfade_frames > 0:
                 cf_start = self.loop_end - self.crossfade_frames
                 in_fade_mask = valid_indices >= cf_start
                 if np.any(in_fade_mask):
                     loop_len = self.loop_end - self.loop_start
                     fade_indices = valid_indices[in_fade_mask]
                     
                     fade_alpha = (fade_indices - cf_start) / self.crossfade_frames
                     fade_alpha = np.clip(fade_alpha, 0.0, 1.0)
                     if sample_data.ndim > 1: fade_alpha = fade_alpha[:, np.newaxis]
                     
                     wrap_indices = fade_indices - loop_len
                     w_idx_int = wrap_indices.astype(int)
                     w_alpha = wrap_indices - w_idx_int
                     if sample_data.ndim > 1: w_alpha = w_alpha[:, np.newaxis]
                     
                     ws0 = sample_data[np.clip(w_idx_int, 0, len(sample_data)-1)]
                     ws1 = sample_data[np.clip(w_idx_int+1, 0, len(sample_data)-1)]
                     w_sample = ws0 * (1.0 - w_alpha) + ws1 * w_alpha
                     
                     c_out = chunk_out[in_fade_mask]
                     mixed = c_out * (1.0 - fade_alpha) + w_sample * fade_alpha
                     chunk_out[in_fade_mask] = mixed

            # Mix accumulation
            if channels == 1 and out_buf.shape[1] > 1:
                out_buf[processed:processed+chunk_len] += chunk_out[:, np.newaxis]
            else:
                out_buf[processed:processed+chunk_len] += chunk_out
            
            # Update state
            processed += chunk_len
            if chunk_len > 0:
                 current_pos = valid_indices[-1] + p_slice[-1]
            else:
                 # Force wrap trigger if Stuck
                 # If scratch mode, we stay stuck (parked)
                 pass

            
        self.position = current_pos
        
        # Apply Volumes
        # gain_curve * env_curve
        total_gain = gain_curve * env_curve
        if total_gain.ndim == 1 and channels > 1:
            total_gain = total_gain[:, np.newaxis]
            
        out_buf *= total_gain
        return out_buf


class SamplerEngine:
    def __init__(self, sample_rate=44100, channels=2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.voices = [Voice(sample_rate=sample_rate) for _ in range(128)]
        self.stream = None
        self.active = True
        self.master_volume = 1.0
        self.output_level = 0.0

    def start(self):
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
            blocksize=512
        )
        self.stream.start()
        self.active = True

    def stop(self):
        if self.stream:
            self.master_volume = 0.0
            sd.sleep(100)
            self.active = False
            self.stream.stop()
            self.stream.close()

    def set_master_volume(self, vol):
        self.master_volume = max(0.0, float(vol))

    def play_voice(self, voice_index, sample, volume=None, pitch=None, mode='normal'):
        if 0 <= voice_index < 128:
            # We pass mode directly to trigger to ensure atomic reset
            self.voices[voice_index].trigger(sample, volume, pitch, mode=mode)

    def set_voice_pitch(self, voice_index, pitch):
        if 0 <= voice_index < 128:
            self.voices[voice_index].set_pitch(pitch)

    def set_voice_volume(self, voice_index, volume):
        if 0 <= voice_index < 128:
            self.voices[voice_index].set_volume(volume)

    def set_voice_loop_window(self, voice_index, loop, loop_start, loop_end, crossfade_frames):
        if 0 <= voice_index < 128:
            self.voices[voice_index].set_loop_window(loop, loop_start, loop_end, crossfade_frames)

    def set_voice_playback_range(self, voice_index, start, end):
        if 0 <= voice_index < 128:
            self.voices[voice_index].set_playback_range(start, end)

    def set_voice_envelope(self, voice_index, attack, decay, curve=1.0):
        if 0 <= voice_index < 128:
            self.voices[voice_index].set_envelope(attack, decay, curve)

    def set_voice_scratch_target(self, voice_index, pos):
        if 0 <= voice_index < 128:
            self.voices[voice_index].set_scratch_target(pos)

    def set_voice_scratch_params(self, voice_index, max_vel, accel):
        if 0 <= voice_index < 128:
            self.voices[voice_index].set_scratch_params(max_vel, accel)

    def set_voice_mode(self, voice_index, mode):
        if 0 <= voice_index < 128:
            self.voices[voice_index].set_mode(mode)
            
    def set_voice_granular_params(self, voice_index, density, duration, jitter_pos, jitter_pitch, jitter_dur):
        if 0 <= voice_index < 128:
            self.voices[voice_index].set_granular_params(density, duration, jitter_pos, jitter_pitch, jitter_dur)

    def set_voice_grain_position(self, voice_index, pos):
        if 0 <= voice_index < 128:
            self.voices[voice_index].set_grain_position(pos)

    def stop_voice(self, voice_index):
        if 0 <= voice_index < 128:
            self.voices[voice_index].release()

    def stop_all(self):
        for v in self.voices:
            v.release()

    def audio_callback(self, outdata, frames, time, status):
        outdata.fill(0)
        mix = np.zeros((frames, self.channels), dtype=np.float32)

        active_voices = 0
        for v in self.voices:
            # We process even if not 'active' to catch tail end of releases? 
            # The process method checks active and returns zeros efficiently.
            # But we can optimize by checking a flag? 
            # Voice.active is set/unset inside process() sometimes (end of decay).
            # But checking it here is RACY if we don't own it.
            # Voice state is owned by audio thread (us). So it is safe to read.
            
            # Optimization: Check if active before calling process (save function overhead)
            # BUT process() handles the command queue! We must call process() if there are pending commands!
            if v.active or not v._command_queue.empty():
                voice_out = v.process(frames, self.channels)
                mix += voice_out
                active_voices += 1

        mix *= self.master_volume
        
        # Calculate Level (Peak)
        if mix.size > 0:
            self.output_level = np.max(np.abs(mix))
        else:
            self.output_level = 0.0
            
        np.clip(mix, -1.0, 1.0, out=mix)

        if not self.active: mix.fill(0)
        outdata[:] = mix
