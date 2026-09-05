# Streaming ASR exploration — parakeet-mlx vs. cache-aware Nemotron (2026-09-04)

Test machine: MacBook Pro M1 Max, 32 GB (NOT the M2 Pro Mini). Timings are indicative only.
Env: `dpg_system_2025` now has `mlx 0.32.2`, `mlx-audio 0.5.1` (git main), `parakeet-mlx 0.5.2`,
and pulled in `transformers 5.16.1`; `sounddevice` bumped 0.5.2 → 0.5.6.

## Task 1 answer: parakeet-mlx cannot load the cache-aware streaming checkpoints

- `parakeet_mlx/conformer.py` raises unless `subsampling == "dw_striding" and causal_downsampling is False`.
  Every cache-aware checkpoint (`parakeet_realtime_eou_120m-v1`, `nemotron-3.5-asr-streaming-0.6b`,
  `nemotron-speech-streaming-en-0.6b`) has `causal_downsampling: true`, `att_context_style: chunked_limited`,
  `conv_context_size: causal`. There is no chunked-limited mask, no conv cache, no `<EOU>` handling.
- Verified empirically at config level: `from_config` fails on the animaslabs Nemotron conversion
  (multi-context `att_context_size` type error) and on the EOU checkpoint's own model_config.yaml
  (`Model is not supported yet!`). animaslabs' model card claiming parakeet-mlx support is wrong.
- Its `transcribe_stream` is windowed local attention over the batch TDT model. Measured on
  `mlx-community/parakeet-tdt-0.6b-v3` with the synthetic clip (`test_parakeet_chunks.py`):
  - 0.1 s chunks: ~310 ms compute per chunk (3× slower than real time) and garbled text.
  - 0.5–2 s chunks, ctx (256,256): correct text, ~260–280 ms per chunk regardless of chunk length.
  - Smaller contexts (128,32)/(64,16) get faster (~110/80 ms) but accuracy collapses.
  - "finalized" region lags by right_context × depth = 256 frames = 20 s, so finalized/draft is not
    usable as confirmed/in-progress at the defaults. Coarse 0.5 s partials, no real confirmation.

## What works: mlx-audio `nemotron_asr` (true cache-aware streaming)

`mlx_audio/stt/models/nemotron_asr/streaming.py`: per-layer attention cache + causal-conv cache +
incremental mel; carried RNNT decoder state. Emission is monotonic. Loaded
`mlx-community/nemotron-3.5-asr-streaming-0.6b` (bf16, 1.2 GB) and `-8bit` (720 MB).

Harness: `nemotron_live.py` (this folder). Feeds 80 ms blocks like a mic, prints each new token with
wall time, audio time fed, and lag.

    python nemotron_live.py --file speech_test.wav --ctx 56,1        # real-time paced
    python nemotron_live.py --file speech_test.wav --ctx 56,1 --fast # compute only
    python nemotron_live.py --mic --device 3 --ctx 56,1               # live mic (Ctrl-C to stop)

Results on the synthetic clip (19.5 s, `say` voice, with 1.5 s / 2 s / 1 s inserted silences):

| ctx     | chunk  | lag behind live edge | bf16 encoder ms/chunk | 8-bit ms/chunk | revisions |
|---------|--------|----------------------|-----------------------|----------------|-----------|
| [56,0]  | 80 ms  | –                    | ~63 (paced)           | –              | 0         |
| [56,1]  | 160 ms | 70–100 ms            | 36 (fast) / 61 (paced)| 21             | 0         |
| [56,6]  | 560 ms | –                    | –                     | –              | 0         |
| [56,13] | 1.12 s | –                    | 88 (paced)            | –              | 0         |

Per-chunk cost is nearly flat across chunk sizes → dominated by per-call overhead (24 layers of
kernel launches + Python greedy loop), not FLOPs. Decoder ≈ 3–6 ms/chunk. "Paced" numbers are higher
than "fast" because the GPU idles between chunks. Plenty of headroom at [56,1]; [56,0] is ~80% of
real time on this machine, so probably marginal on an M2 Pro.

Accuracy vs look-ahead (same clip): [56,13] perfect except "calls" for "pause"; [56,1] drops the
period after "Macintosh"; [56,0] loses one more comma. Synthetic voice; needs David's real speech.

Pause behaviour: text stops at the pause and resumes 0.1 s after speech resumes. Sentence-final
punctuation and the `<en-US>` tag are emitted together with the FIRST WORD OF THE NEXT SENTENCE, not
at the pause — so Nemotron 3.5 gives no "finished vs paused" signal beyond silence. The model
also emits a `<lang>` tag after every sentence end even with `--lang en-US`; the harness shows it as
⟨<en-US>⟩ — it is a free sentence-boundary event.

Mic mode ran 8 s on the built-in mic without error but nobody was speaking; not yet verified with
real speech.

## `parakeet_realtime_eou_120m-v1` (the `<EOU>` model) — not yet runnable on MLX

Config extracted from the .nemo → `eou_model_config.yaml`. Target `EncDecRNNTBPEModel` (no prompt),
17 layers d=512, att_context [70,1] (5.6 s left / 80 ms right), 1-layer LSTM prediction net,
vocab 1026 with `<EOU>` as a normal token. No MLX conversion exists on HF (only ONNX/CoreML).
mlx-audio's converter chokes on it (flat `att_context_size`, required prompt block), and its
`Model` always applies `prompt_kernel`.

Estimated port inside mlx-audio's `nemotron_asr`: make the prompt optional, accept a flat
att_context_size, keep `<EOU>` visible instead of treating it as special, run the existing
`.nemo` converter. Encoder/decoder/joint modules are the same NeMo classes, so weight names should
match. Roughly a half-day if nothing surprises. Alternative: sherpa-onnx / ONNX Runtime with the
`ysdede/parakeet-realtime-eou-120m-v1-onnx` export (CPU, no MLX).

## Not run

- Kyutai STT via moshi-mlx (semantic VAD) — separate env, not tried.
- sherpa-onnx streaming Zipformer — not tried.
- Whisper control comparison on the same clip — `whisper_nodes.py` needs the DPG app; do it live.

## Side note

The env's `ffmpeg` is broken (`liblept.5.dylib` → missing `libtiff.5.dylib`), which is why
`parakeet_mlx.transcribe(path)` fails; `/opt/homebrew/bin/ffmpeg` works. Not fixed.

## Phrase boundary rules on Nemotron (added to nemotron_live.py, 2026-09-04)

`--silence S` (default 0.8 s of audio with no new token) and the punctuation rule (token ending
. ? ! closes the phrase; lang tags dropped). A bare punctuation token arriving right after a
silence-closed phrase is appended to it and logged as an amendment.

Synthetic clip, ctx [56,1]: 4 phrases, 2 by punctuation, 2 by silence, no amendments (the model
never emitted the missing periods at this look-ahead). ctx [56,6]: 5 phrases, 1 by punctuation,
4 by silence, 3 of which were amended with late punctuation ~0.3–1.0 s later (arrives with the
next sentence's first word). So: at short look-ahead the silence rule does the work and punctuation
is mostly absent; at longer look-ahead punctuation is present but trails the silence boundary, so a
"phrase never changes" guarantee holds only if trailing punctuation is either dropped or accepted
as a cosmetic amendment.

## nemotron node (2026-09-04)

`dpg_system/nemotron_nodes.py`, registered as `nemotron` (module added to dpg_app.py). Engine and
segmenter are standalone classes (tested on speech_test.wav without the app); the node is a thin
wrapper following whisper_nodes.py conventions: on/off, audio device or audio_in, worker thread
runs MLX, frame_task hands events to outputs on the main thread.

Outputs: in_progress (current phrase, growing), phrase (closed by pause / sentence end / length),
sentence_in_progress (sentence so far, re-sent at each phrase close → context_tracker
'provisional in'), sentence (closed by . ? ! / long pause / length → context_tracker 'text in').
Options: phrase_silence 0.8 s, sentence_silence 2.0 s, max_phrase_words, max_sentence_words, gain.
Inputs: model (bf16 / 8bit), look-ahead (80…1120 ms), language (prompt key, 'auto' allowed).

context_tracker gained 'provisional in' the same day: processes text against a snapshot of the
state and restores it, 'detected' stays quiet. Not yet run inside the live app.

## Lost words after long pauses (reported live, 2026-09-04) — not reproduced

- Engine alone: utterance A, gap of 2/5/15/30/60 s (digital zeros, −60 dBFS and −40 dBFS white
  noise), utterance B → B always complete, first token 0.5–0.7 s after onset. Same in `auto` language.
- Node capture path (NemotronCapture on the MacBook mic, afplay through speakers, 2 s and 20 s gap):
  B complete, onset level normal, first token ~0.9 s after onset (includes afplay start).
- GPU wake-up after 20 s idle adds ~20 ms per push. Not a cause.
- Remaining suspects are outside what can be reproduced here: input-side processing (macOS mic
  mode / Voice Isolation, interface gate or AGC), GPU contention inside the live app, or something
  about the real room. Node now has a `record_debug` option: writes ~/nemotron_debug/<stamp>.wav
  (the 16 kHz audio the engine received) and .log (tokens/phrases/sentences with audio time).
  If the words are audible in the wav but absent from the log, the model dropped them; if faint or
  missing in the wav, the input side did.

## Continuous speech runs on (2026-09-04)

Synthetic 10-sentence "podcast" clip (say -r 190, inter-sentence pauses 0.2–0.26 s):
- Model emits sentence-final marks for only 6–8 of 10 sentence ends at EVERY look-ahead
  (160: 8, 320: 6, 560: 8, 1120: 7). Look-ahead does not buy punctuation on continuous speech;
  the run-ons are a model limit.
- FIXED segmenter defect: tokens were stamped at chunk end and silence measured against audio
  fed, so at 560/1120 ms look-ahead every chunk gap looked like a pause (60 phrases). Tokens are
  now stamped at their encoder frame and silence is measured to engine.decoded_t. After the fix:
  9/9/8 sentences and 9/9/8 phrases at phrase_silence 0.8 across look-aheads.
- spaCy en_core_web_lg parser does NOT split unpunctuated run-ons (0 of 4 test cases).
- Punctuation restoration `oliverguhr/fullstop-punctuation-multilingual-sonar-base` (xlm-r base,
  ~1.1 GB, CPU) recovers the dropped boundaries ("new | over the years", "consequences | some of
  them", "homemade | we built"), misses the unmarked question; 50–80 ms per 40-word buffer.
  `Qishuai/distilbert_punctuator_en` output was unusable as decoded here.
  Candidate design: run it on the open sentence buffer at each phrase close, split at a predicted
  sentence end ≥K words from the buffer end. Not built.

## Semantic sentence splitter (built 2026-09-04)

`semantic_split` option on the nemotron node. `PunctuationSplitter` wraps
`oliverguhr/fullstop-punctuation-multilingual-sonar-base` (transformers, CPU, 50–80 ms per
40-word check). The segmenter calls it on sentence-buffer + current-phrase text at each
pause/length phrase close and every `split_check_every` (6) new words, once ≥ `split_min_words`
(12), never cutting inside the last `split_guard_words` (3). Ends where the ASR already put , ; :
are skipped (ASR heard the prosody). A cut inside the current phrase emits phrase → sentence →
sentence_in_progress → in_progress (remainder), cutting the string at the character so a bare
space token followed by continuation pieces is not glued (that bug bit once: "questionnever").
Results: podcast clip @160 ms 9/9 correct sentences (no false comma split); @320 ms recovers the
two boundaries the model missed. Model loads on the worker thread when the option is on
(first time downloads ~1.1 GB).

## Clause-boundary breaks for run-on speech (2026-09-05)

`split_max_words` (default 30): when the open text is at least this long and the splitter finds no
sentence end, break at the most plausible clause boundary: ASR-heard , ; : count as score 1.0,
splitter-predicted , : - at their score; near-equal candidates (within 0.1) → the later wins.
Head must be ≥ max(4, split_min_words//2) words. Splitter sentence ends now also require that
minimum head (a one-word "sentence" 'in' had appeared from ends on adjacent words).
Synthetic 130-word run-on (say): 49-word and 44-word pieces became 27+22 and 30+14, cut at
"for seconds, | and I remember" and "around the room | which was". Podcast clip unchanged.
