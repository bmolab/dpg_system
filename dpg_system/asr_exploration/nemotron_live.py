"""
nemotron_live.py — minimal live harness for cache-aware streaming ASR via mlx-audio.

Drives mlx-audio's Nemotron 3.5 streaming model one 80 ms block at a time, the way a
microphone delivers audio, and prints every newly emitted token with:
  wall  = seconds since start
  audio = seconds of audio fed so far
  lag   = wall - audio  (how far behind the live edge the text is)

Modes:
  --file X.wav          feed a 16 kHz wav at real-time pace (sleeps between blocks)
  --file X.wav --fast   feed the file as fast as possible (compute cost only)
  --mic [--device N]    live microphone
Options:
  --ctx 56,1            attention context [left, right]; right ∈ {0,1,3,6,13} → 80/160/320/560/1120 ms
  --lang en-US          language prompt key ("auto" lets the model detect)
  --seconds N           stop after N seconds (mic mode)
  --silence S           close a phrase after S seconds of audio with no new token (default 0.8; 0 = off)
  --no-punct            disable the sentence-punctuation boundary rule

Phrase rules (the "phrases" output the whisper node has; tokens are the "in_progress" view):
  silence : no token for --silence seconds of audio while a phrase is open → emit it
  punct   : a token ending in . ? ! closes the phrase (punctuation included); lang tags are dropped
  amend   : a bare punctuation token arriving right after a silence-closed phrase is appended to
            that phrase and logged as an amendment (Nemotron emits sentence-final punctuation with
            the first word of the NEXT sentence)
"""
import argparse, queue, sys, time
import numpy as np
import mlx.core as mx

from mlx_audio.stt import load
from mlx_audio.stt.models.nemotron_asr.audio import StreamingLogMelSpectrogram
from mlx_audio.stt.models.nemotron_asr.streaming import ConformerStreamingState
from mlx_audio.stt.models.nemotron_asr import tokenizer as tok

SR = 16000
BLOCK = 1280  # 80 ms at 16 kHz = one encoder frame after 8x subsampling


class StreamingTranscriber:
    """Mel → cache-aware encoder → prompt → greedy RNNT with carried decoder state."""

    def __init__(self, model, att_context_size, language):
        self.model = model
        self.language = language
        self.mel = StreamingLogMelSpectrogram(model.preprocessor_config)
        self.enc = ConformerStreamingState(model.encoder, att_context_size=att_context_size)
        self.last_token = model.blank_id
        self.decoder_hidden = None
        self.text = ""
        self.frames_decoded = 0
        self.enc_ms = []
        self.dec_ms = []

    def push(self, samples: np.ndarray, final=False):
        """Feed mono float32 samples. Returns list of new text pieces."""
        new = []
        t0 = time.perf_counter()
        mel = self.mel.push(mx.array(samples), final=final)
        chunks = self.enc.push(mel, final=final) if mel.shape[1] > 0 or final else []
        for encoded in chunks:
            self.enc.materialize(encoded)
            t1 = time.perf_counter()
            self.enc_ms.append((t1 - t0) * 1000)
            prompted = self.model.apply_prompt(encoded, self.language)
            new += self._decode(prompted)
            self.dec_ms.append((time.perf_counter() - t1) * 1000)
            t0 = time.perf_counter()
        return new

    def _decode(self, prompted):
        m = self.model
        pieces = []
        chunk_len = prompted.shape[1]
        t = 0
        new_symbols = 0
        while t < chunk_len:
            feature = prompted[:, t:t + 1]
            cur = mx.array([[self.last_token]], dtype=mx.int32) if self.last_token != m.blank_id else None
            dec_out, (h, c) = m.decoder(cur, self.decoder_hidden)
            dec_out = dec_out.astype(feature.dtype)
            joint = m.joint(feature, dec_out)
            pred = int(mx.argmax(joint))
            if pred != m.blank_id:
                self.last_token = pred
                self.decoder_hidden = (h.astype(feature.dtype), c.astype(feature.dtype))
                piece = m.vocabulary[pred]
                if not tok.is_special_token(pred, m.vocabulary):
                    text = tok.piece_to_text(piece)
                    self.text += text
                    pieces.append((text, False))
                else:
                    pieces.append((f"⟨{piece}⟩", True))  # lang tags; EOU if a model has one
                new_symbols += 1
                if m.max_symbols is not None and new_symbols >= m.max_symbols:
                    t += 1
                    new_symbols = 0
            else:
                t += 1
                new_symbols = 0
        self.frames_decoded += chunk_len
        return pieces


SENTENCE_END = (".", "?", "!")
BARE_PUNCT = {".", ",", "?", "!", ";", ":"}


class PhraseSegmenter:
    """Turns the committed token stream into phrases using silence and punctuation rules."""

    def __init__(self, silence, punct, start_wall, realtime):
        self.silence = silence
        self.punct = punct
        self.start_wall = start_wall
        self.realtime = realtime
        self.current = ""            # in_progress text (only ever grows until a boundary)
        self.last_token_audio = None
        self.phrases = []            # (text, how)
        self.last_closed_by_silence = False

    def _stamp(self, audio_t):
        wall = time.perf_counter() - self.start_wall
        lag = wall - audio_t if self.realtime else float("nan")
        return f"wall {wall:7.2f}  audio {audio_t:6.2f}  lag {lag:5.2f}"

    def _close(self, audio_t, how):
        text = self.current.strip()
        self.current = ""
        if not text:
            return
        self.phrases.append((text, how))
        self.last_closed_by_silence = how.startswith("silence")
        print(f"{self._stamp(audio_t)}  PHRASE[{len(self.phrases)}] ({how}): {text!r}", flush=True)

    def tokens(self, audio_t, pieces):
        for text, special in pieces:
            if special:
                print(f"{self._stamp(audio_t)}      {text}", flush=True)
                continue
            self.last_token_audio = audio_t
            # late punctuation belonging to the phrase silence just closed
            if self.last_closed_by_silence and not self.current and text.strip() in BARE_PUNCT:
                t, how = self.phrases[-1]
                self.phrases[-1] = (t + text.strip(), how + "+amend")
                print(f"{self._stamp(audio_t)}  PHRASE[{len(self.phrases)}] amended: {self.phrases[-1][0]!r}", flush=True)
                continue
            self.last_closed_by_silence = False
            self.current += text
            print(f"{self._stamp(audio_t)}  +{text!r}   in_progress: {self.current.strip()!r}", flush=True)
            if self.punct and text.rstrip().endswith(SENTENCE_END):
                self._close(audio_t, "punct")

    def tick(self, audio_t):
        """Called every audio block, whether or not tokens arrived."""
        if (self.silence > 0 and self.current.strip() and self.last_token_audio is not None
                and audio_t - self.last_token_audio >= self.silence):
            self._close(audio_t, f"silence {self.silence:.1f}s")

    def finish(self, audio_t):
        self._close(audio_t, "end of input")


def run(blocks, st, realtime, start_wall, seg):
    """blocks: iterable of (audio_seconds_fed_after_block, samples)."""
    first = None
    audio_t = 0.0
    for audio_t, samples in blocks:
        new = st.push(samples)
        if new:
            seg.tokens(audio_t, new)
            if first is None and any(not s for _, s in new):
                first = (time.perf_counter() - start_wall, audio_t)
        seg.tick(audio_t)
    new = st.push(np.zeros(0, np.float32), final=True)
    if new:
        seg.tokens(audio_t, new)
    seg.finish(audio_t)
    print("\nPHRASES:")
    for i, (t, how) in enumerate(seg.phrases, 1):
        print(f"  {i:2d} [{how}] {t}")
    print("\nFINAL:", st.text.strip())
    if st.enc_ms:
        e, d = np.array(st.enc_ms), np.array(st.dec_ms)
        print(f"encoder chunk ms: mean {e.mean():.1f} p95 {np.percentile(e, 95):.1f} max {e.max():.1f}   "
              f"decoder chunk ms: mean {d.mean():.1f} p95 {np.percentile(d, 95):.1f} max {d.max():.1f}   "
              f"chunks {len(e)}")
    if first:
        print(f"first text: wall {first[0]:.2f}s, audio {first[1]:.2f}s")


def file_blocks(path, realtime):
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    assert sr == SR, f"need {SR} Hz, got {sr}"
    t0 = time.perf_counter()
    for i in range(0, len(y), BLOCK):
        blk = y[i:i + BLOCK]
        audio_t = (i + len(blk)) / SR
        if realtime:
            wait = audio_t - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)
        yield audio_t, blk


def mic_blocks(device, seconds):
    import sounddevice as sd
    q = queue.Queue()

    def cb(indata, frames, t, status):
        if status:
            print("mic status:", status, file=sys.stderr)
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=BLOCK,
                        device=device, callback=cb):
        print(f"listening on device {device if device is not None else 'default'} … Ctrl-C to stop", flush=True)
        fed = 0
        t0 = time.perf_counter()
        try:
            while seconds is None or time.perf_counter() - t0 < seconds:
                try:
                    blk = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                fed += len(blk)
                yield fed / SR, blk
        except KeyboardInterrupt:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/nemotron-3.5-asr-streaming-0.6b")
    ap.add_argument("--file")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--mic", action="store_true")
    ap.add_argument("--device", type=int)
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--ctx", default="56,1")
    ap.add_argument("--lang", default="en-US")
    ap.add_argument("--silence", type=float, default=0.8)
    ap.add_argument("--no-punct", action="store_true")
    a = ap.parse_args()
    acs = [int(v) for v in a.ctx.split(",")]

    t = time.perf_counter()
    model = load(a.model)
    print(f"loaded {a.model} in {time.perf_counter() - t:.1f}s; ctx={acs} → chunk {(acs[1] + 1) * 80} ms; lang={a.lang}")
    st = StreamingTranscriber(model, acs, a.lang)
    # warm up kernels so the first real chunk isn't charged for compilation
    st.push(np.zeros(BLOCK * (acs[1] + 1) * 2, np.float32))
    st = StreamingTranscriber(model, acs, a.lang)

    start = time.perf_counter()
    realtime = a.mic or not a.fast
    seg = PhraseSegmenter(a.silence, not a.no_punct, start, realtime)
    if a.mic:
        run(mic_blocks(a.device, a.seconds), st, True, start, seg)
    elif a.file:
        run(file_blocks(a.file, not a.fast), st, not a.fast, start, seg)
    else:
        ap.error("--file or --mic")


if __name__ == "__main__":
    main()
