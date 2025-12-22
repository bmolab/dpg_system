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
    def __init__(self, data, volume=1.0, loop=False, loop_start=0, loop_end=-1, crossfade_frames=0, pitch=1.0):
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
        self.crossfade_frames = 0
        
        # Command Queue for Thread Safety
        self._command_queue = queue.SimpleQueue()
    
    # --- Public API (Main Thread) ---
    
    def trigger(self, sample, volume=None, pitch=None):
        """Queue a trigger command."""
        self._command_queue.put({
            "type": "trigger",
            "sample": sample,
            "volume": volume,
            "pitch": pitch
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

    def set_loop_window(self, loop_start, loop_end, crossfade_frames):
        self._command_queue.put({
            "type": "set_loop",
            "start": loop_start,
            "end": loop_end,
            "crossfade": crossfade_frames
        })
        
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
                
                self.loop_start = sample.loop_start
                self.loop_end = sample.loop_end
                self.crossfade_frames = sample.crossfade_frames
                self.position = 0.0
                
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
                self.loop_start = cmd["start"]
                self.loop_end = cmd["end"]
                self.crossfade_frames = cmd["crossfade"]

    def process(self, frames, channels):
        # 1. Process Commands
        self._process_commands()
        
        # ... existing code ...
        
        # inside loop
        # ...


        if not self.active or self.sample is None:
            return np.zeros((frames, channels), dtype=np.float32)

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


            # Determine pitch for this segment (use average or start?)
            # If we are ramping, pitch is `pitch_curve[processed]`.
            # To avoid complexity, let's assume valid pitch is > 0
            
            p = max(0.001, pitch_curve[processed]) # Use start of chunk pitch
            
            effective_end = self.loop_end if self.looping else len(sample_data)
            
            # Dist to end
            dist = effective_end - current_pos
            
            if dist <= 0:
                if self.looping:
                    # Looping wrap
                    current_pos = float(self.loop_start) + (current_pos - self.loop_end)
                    # Safety clamp
                    if current_pos >= effective_end: current_pos = float(self.loop_start)
                    dist = effective_end - current_pos
                else:
                    self.active = False
                    break # Finish
            
            # How many frames fit in this dist?
            # frames * p = samples
            # frames = samples / p
            
            chunk_len = int(dist / p)
            chunk_len = max(1, chunk_len)
            chunk_len = min(frames - processed, chunk_len)
            
            # Now render this chunk
            idx_range = np.arange(chunk_len)
            # if pitch is varying significantly, this linear approx `current_pos + idx * p` might slightly drift from `cumsum`
            # but it is usually acceptable for standard pitch bends.
            # ideally we use the slice of `pitch_curve`
            
            p_slice = pitch_curve[processed : processed + chunk_len]
            p_cumsum = np.cumsum(p_slice)
            # Adjust so first sample is `current_pos + p[0]`? No, `current_pos` is state before first sample.
            # Sample 0 is at `current_pos + p[0]`? Or `current_pos` is Sample 0?
            # Usually: Sample[t] = Sample[t-1] + pitch.
            # So Sample[0] = current_pos (if we assume current_pos is the sampling point)
            # But usually we advance then sample? or sample then advance?
            # Let's say: current_pos is the float index we want to sample NOW.
            # Then next is current_pos + p.
            
            # If so:
            indices = current_pos + (p_cumsum - p_slice[0]) # This starts at current_pos? 
            # No. cumsum starts at p_slice[0].
            # We want [0, p0, p0+p1, ...] or [0, 1, 2] * p?
            # Let's use simple linear interpolation if random access.
            
            # Accurate:
            # indices[0] = current_pos
            # indices[1] = current_pos + p_slice[0]
            # ...
            # This means we sample at current_pos, then advance.
            
            valid_indices = np.concatenate(([0.0], np.cumsum(p_slice)[:-1])) + current_pos
            
            # Interpolation
            idx_int = valid_indices.astype(int)
            alpha = valid_indices - idx_int
            if sample_data.ndim > 1: alpha = alpha[:, np.newaxis]
            
            # Bounds check (safe due to chunk calc, but careful with +1)
            # We know valid_indices are < effective_end (mostly).
            # But s1 might be at effective_end.
            
            s0 = sample_data[idx_int] # Should be safe
            
            # Wrapper for s1
            idx_next = idx_int + 1
            
            # Handle wrapping for S1
            # If we are at the very end of loop, S1 should be loop_start
            mask_over = idx_next >= len(sample_data) # Global end
            if self.looping:
                 # If idx_next hits loop_end, it should wrap to loop_start
                 mask_loop = idx_next >= self.loop_end
                 idx_next[mask_loop] = self.loop_start
            else:
                 idx_next[mask_over] = len(sample_data) - 1
            
            s1 = sample_data[idx_next]
            
            chunk_out = s0 * (1.0 - alpha) + s1 * alpha
            
            # Crossfade Logic (Keeping it conditional for perf)
            # ... (Simplified for brevity, but should copy original logic if needed)
            if self.looping and self.crossfade_frames > 0:
                 cf_start = self.loop_end - self.crossfade_frames
                 in_fade_mask = valid_indices >= cf_start
                 if np.any(in_fade_mask):
                     # Calculate fade pos
                     fade_pos = valid_indices[in_fade_mask] - cf_start
                     cf_alpha = fade_pos / float(self.crossfade_frames)
                     if sample_data.ndim > 1: cf_alpha = cf_alpha[:, np.newaxis]
                     
                     # Sample B (Wrap)
                     pos_b = self.loop_start + fade_pos
                     idx_b = pos_b.astype(int)
                     alph_b = pos_b - idx_b
                     if sample_data.ndim > 1: alph_b = alph_b[:, np.newaxis]
                     
                     sb0 = sample_data[idx_b]
                     sb1 = sample_data[np.minimum(idx_b + 1, len(sample_data)-1)]
                     src_b = sb0 * (1.0 - alph_b) + sb1 * alph_b
                     
                     src_a = chunk_out[in_fade_mask]
                     chunk_out[in_fade_mask] = src_a * (1.0 - cf_alpha) + src_b * cf_alpha

            out_buf[processed : processed + chunk_len] = chunk_out
            
            # Update loop state
            processed += chunk_len
            # Update current_pos to be where the NEXT chunk would start
            # i.e. last sample pos + last pitch
            current_pos = valid_indices[-1] + p_slice[-1]
            
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

    def start(self):
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
            blocksize=512
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.master_volume = 0.0
            sd.sleep(100)
            self.active = False
            self.stream.stop()
            self.stream.close()

    def play_voice(self, voice_index, sample, volume=None, pitch=None):
        if 0 <= voice_index < 128:
            self.voices[voice_index].trigger(sample, volume, pitch)

    def set_voice_pitch(self, voice_index, pitch):
        if 0 <= voice_index < 128:
            self.voices[voice_index].set_pitch(pitch)

    def set_voice_volume(self, voice_index, volume):
        if 0 <= voice_index < 128:
            self.voices[voice_index].set_volume(volume)

    def set_voice_loop_window(self, voice_index, loop_start, loop_end, crossfade_frames):
        if 0 <= voice_index < 128:
            self.voices[voice_index].set_loop_window(loop_start, loop_end, crossfade_frames)

    def set_voice_envelope(self, voice_index, attack, decay, curve=1.0):
        if 0 <= voice_index < 128:
            self.voices[voice_index].set_envelope(attack, decay, curve)

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
        np.clip(mix, -1.0, 1.0, out=mix)

        if not self.active: mix.fill(0)
        outdata[:] = mix
