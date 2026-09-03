# help patch tooling

Four small scripts that make writing help patches a checkable job rather than a
careful one. Nothing here runs at patch time — `dpg_system` never imports it.

## the scripts

### extract_interface.py
Reads every `*.py` in `dpg_system/` with `ast` — no import, no GUI, no torch —
and writes `iface.json`: for each registered node name, the class that builds it
and every inlet, outlet, property and option it declares, with widget types,
defaults and combo choices.

    python3 extract_interface.py iface.json

It follows base classes, resolves factories that dispatch on the node name
(`ValueNode` → `IntNode` / `StringNode` / …), prefers the module that registered
the name when a class name is used in several files, and picks up ports renamed
later with `set_label`. Ports built from a variable show up as `<dynamic>` with
the source line attached, so you can look them up by hand.

`iface.json` is generated. Regenerate it after touching a node's interface.

### validate_help.py
Checks help patches against `iface.json`:

    python3 validate_help.py ../*.json

It catches node names that are not registered, links naming an inlet or outlet
the node does not have, properties matching nothing on the node, and links
pointing at node ids absent from the file.

**Two different name rules, deliberately.** `restore_properties()` compares
labels with a leading `#` guard stripped, so a property saved as `operand`
matches a widget labelled `###operand`. Link resolution in `dpg_app` does NOT
strip — it compares exactly, then falls back to `name_archive`. The validator
mirrors both: stripped for properties, exact plus archive for links. Getting
this wrong in the validator is worse than not checking, because a cord aimed at
`###input` by the bare name `input` then validates and silently fails to
connect on load. This is what catches drift — when a
port is renamed, the help patches that still name the old one show up here.

Links do resolve at load time by name when the index is wrong, and fall back to
the single inlet or outlet when the name is wrong too, so a stale name is
usually invisible until it silently attaches to the wrong port.

### check_coverage.py
Reports which node names have a help patch and which do not, mirroring
`Node.get_help()`'s resolution order. Pass a module name for the detail:

    python3 check_coverage.py
    python3 check_coverage.py signal_nodes.py

It also flags `help_index.json` entries pointing at a file that is not there, or
naming something that is not a registered node.

**help_file_name is inherited.** It is usually set on a BASE class —
`TorchDistributionNode` carries `'t.dist_help'` for all 21 `t.dist.*` nodes — so
resolving it by exact class misses every subclass and badly understates
coverage. `extract_interface.py` records it through the class chain and
`check_coverage.py` reads that. Fixing this reclaimed 21 labels that were
already documented.

### relayout.py  (needs the conda env)
Guessing node sizes does not work — a node's width depends on the widgets inside
it, so a hand-placed comment ends up under a plot. This loads each patch for
real, headlessly, reads the ACTUAL rendered size of every node, writes those
sizes back into the file, then puts every comment in a gutter clear of the demo
column, level with the row it annotates. It also drops the close button below
the demo and keeps everything clear of the title.

    python3 dpg_system/help/tools/relayout.py dpg_system/help/foo_help.json

It reports any demo nodes still colliding — those need fixing in the generator
spec, since relayout will not move the demo itself.

**`--minimal`** is for hand-placed patches. Re-flowing those would throw away
the author's layout, so minimal mode leaves everything exactly where it is
unless it genuinely overlaps something, then moves it the shortest distance
that clears. It still writes the measured sizes back, which matters on its own:
a patch saved with guessed width/height reports overlaps that are not real, and
hides ones that are. Add **`--dry-run`** to see what it would move first.

    python3 dpg_system/help/tools/relayout.py --minimal --dry-run dpg_system/help/*.json

Use plain relayout for generated patches, `--minimal` for hand-made ones.

### two rules these tools must obey

**Close each patch after use.** Both relayout and smoke load every patch into
one app session. Without closing, a full rebuild ends with 169 open tabs and
each later load is slower than the last — the run went from over ten minutes to
205 seconds once `close_current_node_editor()` was called per patch.

**Exit the process yourself when the work is done.** Both tools call
`os._exit(0)` after `main()`. Tearing the app down headlessly — GL context,
audio, MIDI threads — does not reliably complete, and the process then hangs
forever having already written every file. That looked like a hang in the work
and was not: relayout finished all 189 patches in under four minutes and then
sat at exit. Forcing the exit took a full rebuild from "never finishes" to 30
seconds.

A related trap: the in-process watchdog cannot save you from this. A block that
holds the GIL stops the timer thread from ever running, so the watchdog is only
useful for a modal dialog that releases it.

**relayout prints progress per patch, flushed.** Without that the measure loop
is silent until every patch is done, so a patch that blocks is invisible — you
get no output at all and no way to tell which one it was. That one line is what
located this.

**Never let a node raise a modal dialog.** `gl_text` and `mgl_text` have a
`font` property whose callback opens a NATIVE file dialog. `restore_properties`
fires a property's callback when the saved value DIFFERS from the constructed
one — so a help patch that omits `font` gets the dialog on open, and a headless
run waits on it forever with no output. The fix is to save `font` at exactly its
default, which makes the restore a no-op and skips the callback. Both tools also
run a watchdog that aborts after 180s of no progress rather than hanging a build.

Anyone opening such a patch would be met by a file picker instead of
documentation, so this matters for the patches themselves and not only the
tooling.

### smoke.py  (needs the conda env)
Runs each patch for a few seconds, driving the frame tasks, and taps every
outlet. Flags any node that never sent anything or only ever sent zeros.

    python3 dpg_system/help/tools/smoke.py dpg_system/help/foo_help.json

**What it cannot check:** the audio graph. A ~ node's signal outlets carry
samples through the compiled DSP program rather than sending messages, and
headlessly there is no audio callback running at all, so nothing in the signal
path moves. The tool skips signal outlets and says how many audio nodes a patch
holds. For those patches what IS verified is that every node loads and every
link resolves by name — which is what catches authoring mistakes — but whether
they SOUND right has to be checked by opening them.

Repeated node labels are numbered in the report (`t.is_contiguous#2`). Tagging
by label alone merged instances, so one silent node out of three reported the
whole label as silent — twice that read as a bug in a demo that was correct.

Add **`--click`** to bang every button node partway through. Many demos are
event-driven on purpose — dicts, replace, repeat and tracing are not streams —
and without a click their chains never run, so plain smoke reports them as
silent and cannot tell a deliberate design from a broken one.

This is the check that catches a demo which loads, validates and looks right but
teaches nothing. `--click` caught two: a `repeat` demo feeding `accumulate` from
a button (a button sends the word `bang`, which accumulate reads as 0, so the
total never moved) and a `dict` demo that needed three clicks in the right order
to show anything. It found three real node bugs the first time it ran — see the
note at the end of this file. Outlets that only fire on a click (toggle, button)
are skipped; "never sent" on a counter's carry outlet is normal.

### test_node_fixes.py  (needs the conda env)
Regression tests for the node bugs listed at the end of this file. Each was
silent — the patch loaded, validated and looked right, and the demo sat at zero.

    python3 dpg_system/help/tools/test_node_fixes.py

Verified to fail when a fix is reverted, so it is a real guard and not decoration.

### rebuild.py  (needs the conda env)
Runs the whole pipeline in the right order:

    python3 dpg_system/help/tools/rebuild.py

generate -> relayout -> validate -> coverage. Order matters: the generators
write provisional positions, so relayout has to run after them. Use this rather
than running the generators by hand.

### build_help.py
Builds a help patch from a compact spec — title, prose, a list of demo nodes with
positions and properties, and a list of links given by port NAME rather than
index. It writes the same JSON structure the app saves, so a generated patch and
a hand-made one are interchangeable; you can open a generated one, rearrange it
by hand, and save over it.

`help_common.py` holds the shared demo pieces — the property blocks for `signal`
and `plot`, and the `load_bang` → `t 1` pair that switches a signal node on when
the patch opens.

The `make_*_help.py` scripts are the per-module generators. They are kept so the
prose stays editable in a text file rather than buried in JSON.

## how a node finds its help

`Node.get_help()` looks, in order:

1. `help_file_name` set on the class, if there is one
2. a help patch named after the node itself — `metro` → `metro_help.json`
3. `../help_index.json`, which maps a node name to the family patch that covers it

The index is the organising document: one readable file listing every family and
its members, rather than a `help_file_name` line scattered through 900 classes.
Keys are help file stems, so `"trig"` means `trig_help.json`. Keys starting with
`_` are comments and are ignored.

## adding a family

1. `python3 extract_interface.py iface.json` if any node changed
2. write a `make_<family>_help.py`, or copy an existing one
3. run it, then `python3 validate_help.py ../<family>_help.json`
4. add the node names to `../help_index.json`
5. `python3 rebuild.py` to regenerate, relayout and validate everything
6. `python3 smoke.py ../<family>_help.json` to confirm the demo actually moves
7. `python3 check_coverage.py <module>.py` to see what is left

## bugs these tools found

Writing the patches surfaced twelve real bugs, all now fixed and covered by
`test_node_fixes.py`:

- `Node.get_help()` appended `_help.json` to a `help_file_name` that already
  ended in `_help`, so every shared family patch was unreachable.
- `SignalNode.signal_value` started as int `0`. The `on` inlet triggers
  execution, so that int was the first thing the node ever emitted, and
  `ArithmeticNode` conforms its operand to the input type — turning `* 0.5`
  into `* 0` permanently. Only fractional operands were affected, which is why
  `* 30` always looked fine.
- `DifferentiateNode` sent `np.zeros_like(received)` for its first sample. On a
  Python float that is a 0-dimensional array, and anything indexing `shape[0]`
  downstream (plot, the matrix buffers) raised IndexError.
- `ArithmeticNode.execute` conformed its operand to the input type in both
  directions. Widening is right — an array or tensor input needs a matching
  operand — but NARROWING destroyed the parameter: one integer arriving at
  `in` ran a float operand through `any_to_int`, so `* 0.5` became `* 0` and
  stayed there. It is uniquely damaging here because the operand is a parameter
  the user set, never recomputed; the filter nodes coerce running state that is
  reassigned every frame, so they heal on their own.
- `ThresholdTriggerNode` (`trigger`, `hysteresis`) added TWO properties both
  labelled `threshold`. Properties restore by label and the search stops at the
  first match, so the saved release value was applied to the trigger widget and
  the release widget kept its default — a node saved with 0.8/0.2 came back as
  0.2/0.1, with both values wrong. The second is now `release threshold`.
- `MidiDeviceNode.custom_cleanup` called `remove_client()` on `self.in_port`
  without a None check, so closing a patch containing one raised and left the
  patch half-closed. The other two `custom_cleanup` methods in the file guard.
- `MidiDeviceNode.__init__` and `params_changed` called `add_client()` on
  `self.in_port` without checking it for None, which it is whenever no MIDI
  input port exists — the normal case away from the hardware. `midi_device`,
  `mpd218` and `blue_board` therefore could not be created at all. Four other
  call sites in the same file already guarded this way.
- `SMPLBodyNode.receive_betas` dereferenced `self.smpl_model` unconditionally,
  but it is None until a model file is chosen and there is no default. Wiring a
  beta editor into `smpl_body` therefore failed the whole patch load with
  AttributeError, on any machine. `load_smpl_model()` already guarded the same
  way.
- `GLAlignNode.__init__` set `self.axis` AFTER calling `super().__init__()`,
  but `initialize()` runs inside that call and reads it — so `gl_align` raised
  AttributeError and could not be created at all. `self.ready = False` was
  already set before the call, so the ordering was known and these were missed.
- `QuaternionToRotationMatrixNode` guarded with
  `if type(data) not in [torch.Tensor, np.ndarray]: convert`, then handled only
  `torch.Tensor` — so a NumPy quaternion passed the guard unconverted and fell
  out of the branch, sending nothing. `euler_to_quaternion` emits exactly that,
  so the two most natural nodes to chain did not connect, silently.
- `ValueNode.options_changed` called `self.width_option()` unconditionally, but
  knobs deliberately have none (DPG's `knob_float` is a fixed size). Restoring
  any changed option on a saved knob therefore raised during load, and the node
  was dropped from the patch with only a line on the console. The sibling method
  twenty lines above already had the `is not None` guard.
- `RandomGaussNode`, `RandomGammaNode` and `RandomTriangularNode` called
  `arg_as_number(default_value=...)` for every parameter without passing
  `index`, which defaults to 0 — so all parameters read the FIRST argument.
  `random.gauss 0 1` became `gauss(0, 0)` and emitted 0.0 forever.

The last three were invisible to the file-level checks and only showed up under
`smoke.py`.


## help_index.json is generated, not hand-edited

The index maps a help file to the node names it documents:

    {"tcp_numpy_send": ["tcp_numpy_send", "tcp_numpy_receive", ...]}

Node.load_help_index() inverts it to label -> stem. Note the direction: writing
`{"tcp_numpy_receive": "tcp_numpy_send"}` looks equivalent and is not -- the
loader then iterates the string, and every CHARACTER becomes a node name.

Keys beginning with '_' are separator comments that group the file by module.
Sorting the file flat pulls them all to the top and the grouping is lost, so
`order_index.py` regenerates the order from iface.json; rebuild.py runs it.

## socket_nodes bugs found while documenting them (2026-09-02)

- `ProcessGroup` used `os.environ` and socket_nodes never imported `os`, so
  `process_group` raised NameError and could not be created at all.
- Its rendezvous thread was non-daemon and `init_process_group` blocks until
  every participant arrives -- which may be never. Merely having the node in a
  patch could stop the process exiting. Now daemon.
- `udp_numpy_send` called `add_input(..., width=120)`. add_property/add_option
  take `width`; add_input takes `widget_width`, so `width` fell into **kwargs
  and collided -- TypeError, node uncreatable. This asymmetry is a live trap;
  a codebase scan found this was the only caller that hit it.

Two traps that are node behaviour rather than bugs, both documented on the page:

- tcp_latent_send defaults to port 4501, tcp_latent_receive to 4500. A naked
  pair never connects.
- tcp_latent_receive calls get_default_ip() in __init__, so its serving ip is
  the machine's NETWORK address, not loopback. Two patches on one machine need
  127.0.0.1 set explicitly at both ends.
- position and serial are cold inlets; the latents inlet is what triggers the
  send, so they must be set first or the send goes out with nothing in them.

- `udp_numpy_receive` had no default port: with no argument `self.port` was
  never set and `UDPReceiveSocket(port=self.port)` raised AttributeError, so a
  bare node could not be created. Defaulted to 3500 to match udp_numpy_send,
  and the port widget now shows it instead of 0.

## torch_kornia_nodes bugs found while documenting them (2026-09-02)

Both were inconsistencies with the module's own conventions, found by probing
all 7 nodes against CHW/HWC and float/uint8 inputs rather than by reading:

- `k.rgb_to_grayscale` was the only node in the file not calling `.float()`
  first, so a uint8 image came back uint8 -- the weighted RGB sum truncated to
  integers and most of the precision thrown away.
- `k.apply_colormap` was the only node not stripping the leading batch dim.
  apply_colormap adds a channel axis, so (1,H,W) came back (1,3,H,W) and it
  alone sent 4-D where its siblings all send CHW.

Not bugs, documented on the pages: the HWC/CHW guess in
`data_to_torchvision_tensor` is ambiguous for very small images; and
`k.rgb_to_grayscale`/`k.rgb_to_hls` reject an already-grayscale image rather
than passing it through.

Demo tuning worth remembering: canny THRESHOLDS, so over-blurring makes it
return an empty map while sobel still reports small non-zero values. At 32x32,
gaussian_blur sigma 3.0 gives canny exactly zero edge pixels; sigma 1.5 gives
~230. Heat maps need scaling to the filter's real range (sobel here tops out
near 0.10, DoG spans about -0.05..0.05) or they read as blank.

## torchvision_nodes (2026-09-02)

No new bugs -- the tv.adjust dispatch bug was fixed in an earlier session and now
has a regression test. Behaviour worth knowing, all measured rather than read:

- The tv nodes PRESERVE dtype: uint8 in, uint8 out, correctly scaled. The k.
  nodes convert everything to float. That is the main reason to pick one family
  over the other.
- tv.Grayscale passes an already-grayscale image through unchanged;
  k.rgb_to_grayscale raises on it (it wants 3 channels).
- The four factor adjustments are EXACTLY identity at 1.0 (measured max abs
  change 0.0), and hue is identity at 0.0 to within 1e-6.
- Factor 0 is the clearest definition of each: brightness -> pure black,
  contrast -> one flat grey at the image mean, saturation -> grey with detail
  intact, sharpness -> slightly softened.
- They CLIP float images to 0..1, and they do it even at factor 1.0 -- a no-op
  adjustment permanently flattens anything above 1.0. tv.adjust_hue is the
  exception; being a rotation it leaves the range alone.
- hue -0.5 and +0.5 produce an identical image: a full turn is 1.0.

The demo's claims are verified by the patch itself: at defaults max reads 0.9995
and min 0.0004; setting brightness to 0 drives max to 0.0, and setting contrast
to 0 makes min and max meet at 0.4972, the image's own average.

## spacy_nodes (2026-09-02)

No new bugs. The findings were behavioural, and two of them change how the nodes
should be described:

- `rephrase` does NOT simplify a sentence, despite the name. It ACCUMULATES:
  each fragment is folded into the phrase built so far using both parse trees.
  Two gates hide this from a naive test -- `conditional_parse` returns early
  while `self.doc is None` (so the first sentence is never rewritten), and a
  fragment whose complexity is ABOVE the threshold is passed through untouched
  rather than reduced. The threshold is a ceiling on what it will attempt, not
  a trigger. A high `clip score` silences the node entirely.
- spacy similarity does not use the bottom of its 0..1 range. Measured over
  deliberately unrelated material on en_core_web_lg: single words 0.01..0.61
  (mean 0.21), whole phrases 0.25..0.85 (mean 0.44). So 0.4 is the baseline,
  not zero, and scores compress as phrases lengthen -- 'apple' vs 'apple' 1.0,
  vs 'a red apple' 0.78, vs 'a red apple sitting on a wooden table in the
  morning light' 0.52, for the same subject.

Tooling note: smoke's `--click` bangs `button` nodes only, so a page driven by
clickable `message` nodes reports everything as never sent. That is a limit of
the harness, not the patch -- verify those by firing the message nodes in a
probe. Left as is deliberately: clicking every message at once would give
misleading results on order-dependent demos.

## speech_analysis_nodes (2026-09-02)

A real node bug, found by checking that a demo's counter counted the right thing:

- `trigger` / `hysteresis` (ThresholdTriggerNode) SILENTLY DROPPED BOOLEANS. The
  scalar branch tests `type(data) in [float, np.double, int, np.int64]`, and
  `type(True) is bool` -- not in that list, even though bool subclasses int. A
  boolean fell through every branch with no output and no error. That is exactly
  how speech_envelope's `onset` (which sends a bool) reached a trigger and did
  nothing. `one_euro` shared the identical list and the identical drop; both
  fixed. NoiseGateNode has a related but different list (`[float, np.double]`,
  no int at all) -- left alone, not demonstrated.

Behaviour worth knowing, measured:

- These nodes analyse on the WALL CLOCK at `analysis_fps`, not per chunk. Push
  audio through faster than real time and most of it is buffered, not analysed.
  A probe that feeds chunks in a tight loop gets nothing out of pitch, spectral
  or voice_quality -- add a sleep.
- Reference values from a synthetic harmonic stack: f0 lands exactly on the
  input frequency; crest factor 1.77 (pure sine 1.41 = sqrt 2, white noise 3.45);
  flatness 0.00 tonal but white noise reads only ~0.56, not 1.0; HNR 72 dB clean,
  13 dB with 30% noise, -5.7 dB for pure noise.
- jitter reads exactly 0.0 for white noise -- no periodicity to measure, NOT a
  perfect voice. Gate jitter/shimmer on speech_pitch's `voiced`.

Two tooling lessons:

- A `plot` does not forward what it is shown -- it only speaks when asked (see
  the plot_nodes section: send it 'dump'). Tapping a plot's outlet to check
  whether a link works therefore proves nothing; tap the INPUT. I briefly
  mis-diagnosed four working links this way.
- validate_help now flags a link whose dest port name is empty when the node has
  several inlets AND no unnamed one to fall back to. The loader resolves an
  empty name via the single-inlet fallback, or by exact match against an inlet
  literally named '' (np.rand's first inlet), so both of those stay legal. The
  rule caught one genuine pre-existing break: text_help wired a counter into
  `prepend`, which has ['in', 'prefix'] and no unnamed inlet.

## point_cloud_nodes (2026-09-02)

One usability fix, plus measurements that became the substance of both pages.

- `pc_background` cancelled a learning run in SILENCE. `_ensure_grid()` discards
  hits/bg_mask/learn_remaining whenever the voxel geometry changes, which is
  correct -- the hits array no longer maps -- but the most natural gesture
  triggers it: press 'learn' before the first cloud arrives and the grid is
  built from the fallback bounds, then the first frame's carried crop rebuilds
  it and the run is thrown away, right after announcing it had started. It now
  says so. The documented procedure is: start the sensor, settle the crop, THEN
  press learn.

Measured, and used directly in the pages:

- The cloud-frame convention works as documented: pc_crop turns a raw (N,3)
  array into {'point_cloud', 'crop'}, and downstream grid nodes use the carried
  volume. Now covered by a regression.
- pc_voxel on a 1,506-point crop: 1,328 points at 0.05 m, 746 at 0.1, 247 at 0.2.
- `dilate` on a 3,000-point wall with realistic jitter: 229 points of false
  foreground survive at dilate 0, 1 at dilate 1, 0 at dilate 2 -- and the person
  in front is untouched (1,500 in, 1,500 out) from dilate 1 up. 1 is the right
  default.
- min points and voxel size are ONE control. On a solid 1,500-point object:

                   min points 1     2       4
      voxel 0.04 m       1500     194       5
      voxel 0.08 m       1500     861     215
      voxel 0.15 m       1500    1345    1099

  At the default voxel size, min points 2 destroys seven eighths of a real
  object. This is the first thing to check when denoising eats the subject.

## matrix_nodes (2026-09-03)

Two real bugs, both hit immediately by probing rather than reading.

- `buffer` CRASHED on resize. BufferNode called `ndarray.resize()` in place, and
  numpy refuses that once another object holds a reference -- and the node sends
  the buffer itself, so any downstream connection is such a reference. Changing
  'sample count' or 'update style' on a wired buffer raised
  `ValueError: cannot resize an array that references or is referenced by
  another array`. The vertical branch had already papered over it with
  `refcheck=False`, which is worse than it looks: it silences the check by
  leaving the other holder pointing at freed memory. Both branches now allocate
  a fresh array. Both already reset write_pos, so starting from zeros is what
  was meant. Covered by a regression.

- `cwt` HUNG THE APP. Not the node -- `ssqueezepy.cwt()` never returns on its
  parallel path in this environment, right after OpenMP announces itself (more
  than one OpenMP runtime is loaded: numpy/torch plus ssqueezepy's, the usual
  macOS cause). matrix_nodes now sets `SSQ_PARALLEL=0` before importing
  ssqueezepy, via setdefault so it can still be overridden. Serial is not slower
  here on frame-sized data: 512 samples took 0.8s parallel and under 0.05s
  serial.

Measured, and used in the pages:

- buffer 'update style' on [1,2,3] into a buffer of 8: 'holds one sample' gives
  shape (3,) and IGNORES the sample count; 'stream of samples' gives (8,);
  'multi-channel sample' gives (8, 3).
- `octaves` on cwt is really VOICES PER OCTAVE (it is passed as nv). On 512
  samples: 2 -> (17, 512), 8 -> (65, 512). Columns always match input length.
- `confusion` here is EXACT EQUALITY, not similarity -- unrelated to
  spacy_confusion despite the identical output shape. 'apple' vs 'apples' is 0.

## vision_describe_nodes (2026-09-03)

One real breakage, and one demo bug of my own that the page now warns about.

- `vision_describe_smol` COULD NOT RUN. It imported `AutoModelForVision2Seq`,
  which transformers 5 removed (5.5.3 here), so every inference died with
  ImportError. The Gemma node in the same file already used the replacement,
  `AutoModelForImageTextToText`. The import now tries the new name and falls
  back to the old, so it works on both. The other three were already fine:
  moondream uses AutoModelForCausalLM, Qwen uses
  Qwen2_5_VLForConditionalGeneration, Gemma the new name.

- My first demo fed the prompt from a `message` node. message SPLITS on
  whitespace and emits a list; the prompt inlet requires a str and silently
  falls back to its own default when handed anything else -- so the patch looks
  wired and the model answers a question you did not ask. Use `string` or
  `text`, which send the line intact. Now documented on the page.

Measured on SmolVLM 500M (already on disk), Apple silicon:

- first description 3.3 s, subsequent 1.9 s
- DROP-FRAME confirmed: 10 frames sent as fast as the patch could manage
  produced 2 descriptions. execute() overwrites the single pending frame rather
  than queueing, which is correct -- a queue would fall ever further behind the
  camera -- but it means you must never assume one answer per frame.
- End-to-end in the built patch: prompt arrives intact, answer reaches
  text_display in 2.8 s.

Probing note: set HF_HUB_OFFLINE=1 when testing these so a missing model fails
fast instead of silently downloading gigabytes.

## node.py — subpatchers, and multi-patch file support (2026-09-03)

No node bugs. This one needed TOOLING work instead: a p/patcher page is close to
useless without a subpatch you can actually open, and the pipeline only handled
single-patch files.

The file format for a patcher and its subpatches:

    {"name": ..., "path": ..., "patches": {"0": {...}, "1": {...}}}

Each entry is a full patch. The linkage is two-way: a subpatch's `id` equals the
parent node's `patcher id`, and the subpatch's `parent_node_uuid` equals the
parent node's `id`. The top patch is the one with no `parent_node_uuid`. Nesting
is arbitrary — the example at examples/deeper_embed_example.json is three deep.

Three tools changed:

- `build_help.build()` takes an optional `subpatch={'name', 'host', 'demo',
  'links'}` and writes the multi-patch format, wiring both sides of the linkage.
- `relayout` lays out the TOP patch and leaves subpatches as authored — they are
  small, hand-placed, and have no title or text block for the gutter logic.
- `validate_help` now checks EVERY patch in the file. It used to return early on
  anything without a top-level 'nodes', so a subpatch was skipped in silence —
  and it immediately caught two links in the new page's subpatch that would
  never have connected (`*` and `+` have inlets ['in', '###operand', 'operand']
  and no unnamed one to fall back to).

Verified by loading the built page: two editors open (`patcher_help` and
`scale_and_offset`), the p node's first outlet is named `scaled` by the `out`
node inside it, and the scaled plot runs 0.000..1.000 against the raw signal's
-1.000..1.000.

Behaviour worth knowing, from the source: a p node carries 20 hidden inlets and
20 hidden outlets and reveals only the ones an `in`/`out` node has claimed;
slots are handed out in creation order, and deleting an in/out frees its slot
for the next one made, dropping any cords into it.

## eos_nodes, torch_loss_nodes, gang_nodes, opencv_nodes (2026-09-03)

Five bugs, four of them in eos_nodes alone -- `color_source` was broken in three
separate ways at once and could not be created with an argument, so none of them
had ever been reached.

eos_nodes:
- `ColorSourceNode` did not inherit `OSCBase`, whose class attribute
  `osc_manager` OSCSender.__init__ reads when given a single argument. So
  'color_source 7' raised AttributeError and the node could not be created --
  while a bare 'color_source' happened to work, because that branch is only
  taken when there is exactly one argument.
- It wired a widget to `self.address_changed`, which exists on OSCSendNode but
  not on OSCSender, so construction raised AttributeError.
- Its dirty flags were named `<param>_changed` -- the same names as the callback
  methods. Before a slider had moved, the flag WAS the bound method, which is
  truthy, so the first change to any one parameter sent all five. On a lighting
  desk that means nudging red also sends intensity 0 and blacks out the channel.
  Renamed to `<param>_dirty` and initialised.
- A lone numeric argument means the CHANNEL, but OSCSender had already taken any
  single argument as the address, so 'color_source 7' composed
  '/7/7/param/red'. A purely numeric address is now treated as "not given",
  restoring the '/eos/user/99/chan' default.

torch_loss_nodes:
- `t.cross_entropy_loss` passed `match_tensor=input_tensor`, copying the input's
  float dtype onto the target -- but class indices must be int64, so the normal
  way to use this loss always failed with 'expected scalar type Long but found
  Float'. It now distinguishes class indices from soft targets by shape.

Measured, and used in the pages:
- one element off by 4: MSE 16.0, L1 4.0. Off by 0.4: MSE 0.16, L1 0.40 -- the
  order swaps, because squaring makes errors under 1 count for less.
- both use reduction='sum', so the number grows with tensor size.
- cross entropy on logits [2.0, 0.5, 0.1]: correct class 0 -> 0.32, class 1 ->
  1.82, class 2 -> 2.22. Confidently wrong costs far more than uncertain.
- color_source composes /eos/user/99/chan/<channel>/param/<name>.

Demo note: `target` is a cold inlet on all three loss nodes, so a demo driven by
message clicks must fire the target first. The page says so; my first version
silently produced nothing from the cross-entropy chain.

opencv_nodes: no bugs. Worth knowing that 'refresh' finds cameras by OPENING
each index up to ten in turn -- there is no polite enumeration -- so it takes a
moment and briefly lights every camera on the machine. Output is converted
BGR->RGB before sending.

## oscquery_nodes (2026-09-03)

No bugs. The page is about what OSCQuery buys over plain OSC -- a device that
announces itself by name AND describes its whole namespace with types and
ranges, so the patch can ask instead of being told.

Worth knowing, from the source:

- `oscq_browse`'s 'subset' control is `self.channel_input` internally (grepping
  for 'subset' finds only the declaration and makes it look dead -- it is read
  at two call sites). It parses '1', '1-5', '1,3,5', '1-3,7,9-11' and filters
  which numbered channels get built. Without it, 'create all' on a 64-channel
  desk builds all 64.
- 'create as' picks widget / send / receive for the same parameter -- the
  namespace description does not decide direction, you do.
- `oscq_host` takes its service name from the CONTAINING PATCHER, which is a
  real reason to use a subpatcher even when tidiness is not the point.

The page ships a working subpatch (second one in the project after
patcher_help): verified two editors open, the p node's outlet named 'level' by
the `out` node inside, and oscq_host having created its osc_device on load.

## plot_nodes (2026-09-03)

One fix, and the answer to something I got slightly wrong earlier.

- `plot` had an outlet that could NEVER fire. HeatMapNode and ProfileNode both
  answer the string 'dump' by sending their collected buffer; PlotNode did not,
  so its outlet was inert. Added the same three lines its siblings use. Now
  `plot` dumps 200 samples, `heat_scroll` 200, `heat_map` 1 (its sample count
  defaults to 1, because the incoming array IS the picture).

That also corrects the note above from the speech_analysis work: these nodes do
forward, but only on request. Nothing leaves the outlet at any other time, which
is why tapping it tells you nothing about whether a link is live.

Worth knowing:
- `heat_map` and `heat_scroll` are ONE class; the name you type sets the default
  'update_mode' and it can be switched afterwards. heat_map treats an incoming
  array as the whole picture; heat_scroll treats it as one column and keeps
  history. Switching to heat_scroll bumps sample count to 200 if it was 1.
- min y / max y is the commonest reason a display looks broken -- this project
  has hit it repeatedly, which is why several pages carry measured ranges.
- plot's 'update style' is the same three-way choice as `buffer`, with the same
  three phrases.

## prompt_nodes (2026-09-03)

No node bugs. Two behaviours found by probing that the page now leads with, and
one demo mistake of mine that is the SECOND time the same trap has bitten.

- `message` SPLITS ON WHITESPACE. My first demo fed 'a dark forest@2' from a
  message node; weighted_prompt received a list of words, joined them back, and
  never parsed the '@' -- so the entry came out as
  ['a dark forest@2', 0.0], weight zero, contributing nothing. ambient_prompt
  fared worse: its list branch expects exactly [text, number], so a four-word
  list yielded just 'harsh'. Use `string` (or `text`) for anything multi-word.
  This is the same fault as the vision_describe prompt inlet -- worth treating
  as a rule: an inlet that wants a phrase must never be fed from `message`.

- `prompt_composer` DROPS weight <= 0 as EXPIRED (`_as_weighted`, commented as
  such). That is the fade-out mechanism for live speech -- a phrase decays with
  age and vanishes at zero without anything having to remove it. The
  consequence, now documented: negative "push away" weights work in
  weighted_prompt and ambient_prompt but are discarded by the composer.

- `weighted_prompt` keeps a slot's PREVIOUS weight when a phrase arrives with no
  '@', which is zero on a fresh slot. It rewrites the box to 'phrase@0.000' so
  this is visible rather than silent -- good design, worth documenting.

Verified in the built patch, all three readouts live:
    weighted_prompt  [['a dark forest', 2.0], ['rain', 1.0],
                      ['harsh light', -2.0], ['fog', 0.5]]
    ambient_prompt   '((a dark forest)), (rain), [[harsh light]], fog, '
    prompt_composer  [['cinematic', 1.0], ['a cold room', 1.0],
                      ['a dark forest', 2.0], ['rain', 1.0], ['fog', 0.5]]
(harsh light absent from the last, which is the expiry rule doing its job.)

## vae_nodes (2026-09-03)

No bugs. The VPoser model is present at
/Users/drokeby/Dev/human_body_prior_/support_data/dowloads/V02_05, so this one
could be measured rather than described.

Measured, and the substance of the page:

- Shapes: 63 in (21 body joints x 3), 32 latent, (22, 3) out -- 22 because the
  root joint comes back with it, which is what 'pass root orientation' is about.
- The round trip is NOT lossless, and that is the point. Putting an arbitrary
  made-up pose through repeatedly and measuring how far each pass moved it:
  0.4540, 0.1008, 0.0649, 0.0372 rad. The first pass does nearly all the work,
  because one pass is enough to land it somewhere the model considers plausible.
  That measurement is what turns "it is a pose prior" from an assertion into a
  demonstration.
- 32 zeros is NOT a neutral pose. It decodes to joint values spanning about
  -0.915 to +0.969 -- the middle of the learned space is a particular pose, not
  an empty one.
- 'mean of dist' ticked gives the encoder's mean (same pose -> same latents);
  unticked samples the distribution, so the same pose wobbles.

Wiring traps the validator caught, both mine: `smpl_take` emits 'joint_data',
not 'pose'; and `smpl_body`'s only inlet is 'betas' (body SHAPE) -- a decoded
pose goes to `gl_body`'s 'pose in'. The existing patches under patches/vposer/
were the right reference and I should have read them first.

## gemma_4_node (2026-09-03)

No bugs. A big node -- 29 inlets, 9 outlets -- so the page is organised around
what is UNUSUAL about it rather than around the interface. Most of that is about
reaching into a generation while it happens.

Read from the source:

- Three distinct stops: `polite_stop` (stop_at_next -- finish the sentence),
  `interrupt` (stop now, mid-word), `hard interrupt` (abandon). Worth separating
  because only the first leaves text an audience can read.
- `target_length` is NOT a truncation. It is an EosTokenRewardLogitsProcessor
  that makes the end-of-turn token progressively more likely as the text nears
  the target, so the model finds its own ending. `max_tokens` is the hard cut.
- `step` + `choice` is the standout capability: generate one token at a time,
  then walk the alternatives the model was weighing and substitute one (it sends
  '<backspace>' then the replacement). Needs `show_probs` on. That is writing
  WITH the model rather than receiving from it.
- `score_incoming_text` runs text you send THROUGH the model and reports how
  likely each token was -- a predictability measure over existing text, with no
  generation at all. Low means unusual.
- Thinking is a separate outlet from output.
- Models are cached in a CLASS-level dict keyed by model, so several gemma_4
  nodes share one copy of the weights. Mixing gemma_4 and gemma_4_31b loads
  both, which is what will not fit.
- Contexts differ by size on purpose: 12B gets 8192, 31B only 2048, because 31B
  with more does not fit in 32GB of unified memory.

Nothing is loaded until the on/off toggle, so the help patch opens instantly and
does not trigger a multi-gigabyte download just by being read -- checked.

## movie_nodes (2026-09-03)

No bugs. Two facts I nearly got wrong by assuming, and checked instead:

- Playback is TIME-BASED. frame_task accumulates `dt * fps * speed` and advances
  only by whole frames, returning without emitting when the accumulator has not
  reached one. So a heavy patch drops video frames rather than playing in slow
  motion -- it keeps real time -- and at slow speeds it sends FEWER frames, not
  duplicated ones. I had written "repeats frames", which is wrong: nothing is
  emitted between whole frames and nothing is interpolated. Matters for anything
  counting frames downstream.
- `done` fires on every loop wrap, not only at the end, so it doubles as a lap
  marker.

The design worth documenting: movie_player and movie_clip_dict are two halves of
one workflow, and BOTH cords are needed -- clip_spec up from the player, command
back down to its 'input'. Without the second, storing names does nothing.

The commands are plain text ('play 240 512 1.0'), which the player accepts from
anywhere, so the dictionary is optional if the numbers come from elsewhere.
In the argument parser, a whole number is a frame and a number with a decimal
point is the speed -- that is the only thing distinguishing them.

A clip spec carries no reference to the movie: a saved collection belongs to the
file it was made from.

## pjlink_nodes (2026-09-03)

No bugs. One node under two names; the page is mostly operational knowledge that
is not visible from the interface.

- 'connect' is a CHECKBOX, not a button -- unticking it disconnects -- and it
  defaults off, so opening the help page sends nothing to the network. Checked,
  because the default ip is a real address (10.1.1.141).
- Authentication: the projector sends a one-time seed, the node replies with
  md5(seed + password). The password never crosses the network. On a garbage
  handshake the node deliberately refuses to mark itself connected rather than
  letting later commands fail silently -- that guard is already in the source.
- The operational point worth the most: POWER is slow and ignores commands while
  cooling; SHUTTER (AVMT) is instant, reversible and free. Black the shutter
  between cues and leave power alone. freeze holds the last frame instead, which
  lets the computer change without the audience seeing it.
- Input codes are PJLink's, not the projector's front-panel labels: RGB 1 = 11,
  Video 1 = 21, HDMI 1 = 31, Storage = 41, Network = 51. A projector will only
  have some, so an input change that appears to fail usually means that input
  does not exist.
- custom_cmd sends a raw command as typed, class prefix included ('%1POWR ?'),
  and 'response' carries the answer -- which is how to ask about lamp hours or
  error status.

## visca_nodes (2026-09-03)

No node bugs -- but a REAL mistake of mine, caught by smoke, that is worth a
standing rule.

My first demo wired a `signal` into the camera's 'pan'. visca_camera has no
connect gate: it sends the moment a value arrives. So smoke filled the console
with 'VISCA: Send Error: No route to host' -- and on a machine where 10.1.1.160
actually exists, opening the help file would have started panning a real camera.

RULE: before wiring a demo to a device node, check whether it sends on receipt
and whether it has a connect gate. pjlink_projector has one (a checkbox,
defaulting off) so its demo can be live; visca_camera has none, so its demo is
deliberately left UNCONNECTED with a comment saying why. A help file must not
move someone's equipment.

Content worth knowing, from the source:

- pan/tilt (-20..20), zoom (-7..7) and focus (-1..1) are RATES, not positions.
  Set one non-zero and the head keeps moving until it is set back to zero. A
  patch that sends a value and forgets has left the camera turning.
- Presets are the practical way to work: frame by hand, store under a number,
  and the patch only ever recalls numbers.
- `reset_sequence` is the recovery button and deserves its own paragraph. VISCA
  over IP numbers every packet; if the count desyncs (camera power-cycled,
  packets lost, another controller spoke to it) the camera IGNORES everything
  SILENTLY -- no error, node looks fine. reset_sequence restarts the count and
  sends an IF_Clear. Try that and `reconnect` before suspecting the network.

## clip_nodes (2026-09-03)

No bugs, but the probing produced a correction to an EARLIER page of mine.

- These are CLIP's TEXT ENCODER ONLY -- clip_nodes.py imports CLIPTextModel and
  nothing else. There is no image side, so they cannot score a picture against
  words. The vision_describe page said "clip nodes score an image against words
  you supply", which is wrong; that cross-reference is now corrected.

Measured:

- 512 dimensions, whatever the phrase length (77 tokens max, then truncated).
- Cosine similarity behaves like the spacy nodes: 'a dark forest' vs 'a forest
  at night' 0.831, but UNRELATED phrases sit at 0.39-0.45, not near zero. Judge
  against that baseline.
- `clip_embedding_length` is NEARLY CONSTANT and should carry a warning:
  a dark forest 24.34, a forest at night 23.54, a bright kitchen 23.41,
  quantum field theory 24.46. A one-in-twenty spread across phrases with nothing
  in common. The meaning is in the DIRECTION, not the magnitude -- which is why
  similarity is computed on normalised vectors and the length is normally
  discarded. A patch mapping that length to something audible is responding
  mostly to noise.

Verified in the built patch: both strings fire, info and plot receive (512,),
and the float reads 23.535 -- matching the measured length for that phrase.

## layout_nodes (2026-09-03)

No bugs. The page's job is to explain what these are FOR, which is not obvious
from the interface: text_display is for the person patching, these are for an
audience -- they render through Cairo to an image for a projector or a texture.

Measured: cairo_layout's 'layout' outlet is (1080, 1920, 3) float32 -- a full HD
frame, HEIGHT-first like a camera frame rather than channels-first. So it feeds
the same places a camera picture does and the k./tv. filters guess its
orientation correctly.

Read from the source:

- `llm_layout` is built to receive gemma_4's `layout_out`, which does NOT send
  text -- it sends commands: add, prompt, streaming_prompt, choose, choice_list,
  step_back, temperature, show_probs, save, scroll_up/down. That is why the pair
  can show which alternatives the model weighed, not just what it said.
- `active_line` (default 17) is why it reads well: new text always appears on
  the same line and everything above scrolls up, so the writing happens at a
  fixed height rather than creeping down the page. The node moves it to line 5
  while a streaming prompt is entered and back afterwards -- documented so the
  jump does not look like a fault.
- `colour_mode` paints each token by temperature, entropy or probability, so the
  text carries the model's certainty as it wrote.
- Font fallback is already careful in the source: requested path, then bundled
  Inconsolata-g.otf, then Cairo's default toy face -- a missing font never
  crashes construction.

## google_translate_nodes (2026-09-03)

A real fix, prompted by David's warning that these "might fail without the
correct login".

- The Google Cloud imports were at MODULE level, so a machine without the SDK
  lost the entire module -- dpg_app reported 'Skipped google_translate_nodes:
  missing dependency google.cloud' and BOTH nodes disappeared. But only
  `translate_api` needs the SDK: `translate` uses the unofficial endpoint and
  needs nothing but requests, which is installed. The imports are optional now,
  and translate_api is registered only when they are available, with a console
  line saying so. Covered by a regression.

Verified after the fix: `translate` creates and works with no credentials at
all -- 'the room was colder than it had been' -> "la pièce était plus froide
qu'avant" in about 0.7 s.

The distinction the page is built on, since it decides which belongs in a piece:

- `translate` is unofficial. No account, works immediately, and can stop working
  without warning if the page changes; rate-limited with no quota to raise;
  5000 characters at a time. Right for making work and rehearsal, wrong for an
  installation running unattended for months.
- `translate_api` is the paid Cloud service: needs a project, the SDK, and
  GOOGLE_APPLICATION_CREDENTIALS. Supported and stable. The node already
  degrades gracefully on missing credentials (translate.Client() is wrapped and
  the node reports 'disabled' rather than crashing) -- that guard was already in
  the source.

Both send the text to Google, which the page says plainly.

The help patch makes no request on open: the string has to be clicked. Checked.

## orbbec_nodes (2026-09-03)

No bugs -- this module is already careful (defensive pyorbbecsdk import, so the
node registers and reports clearly when capture is enabled rather than taking
the library down at import).

The content is operational, read from the source:

- Depth modes and their real trade, from FEMTO_DEPTH_MODES: NFOV unbinned
  640x576@30, NFOV binned 320x288@30, WFOV binned 512x512@30, WFOV unbinned
  1024x1024@15. WFOV unbinned is the ONLY mode that cannot do 30 fps -- worth
  calling out, because a patch that stops feeling responsive after a resolution
  change is the camera doing exactly what it was asked.
- 'level to gravity' deserves top billing: it uses the accelerometer to rotate
  the cloud so the floor is flat whatever angle the camera is bolted at, which
  makes every height and crop box downstream mean what it says. It calibrates
  for ~1s then STOPS the accel deliberately -- an active IMU stream makes the
  Bolt deliver depth in ~115ms clumps. That is in the source comment and is
  exactly the sort of thing a user would otherwise read as a fault.
- The bursty-USB state is real and has three controls (report frame gaps, auto
  usb reset on stutter, reset usb device). Frames arriving in clumps mid-session
  is the USB session degrading, not the patch.
- units: device native is millimetres; the pc_ nodes' crop boxes are in metres,
  so metres is the sane choice.

The page points straight at pc_crop as the next node, matching what the
point_cloud page already says about cropping first.

## ultracwt_nodes (2026-09-03)

Two real bugs, one of which had killed both nodes outright.

- `scipy.signal.morlet2` was REMOVED in SciPy 1.15 (this env has 1.15.2). Both
  nodes build their wavelet bank from it, so every build failed with
  "module 'scipy.signal' has no attribute 'morlet2'" -- and because the
  constructor is wrapped in a try/except, `self.cwt` stayed None and the nodes
  produced NOTHING while raising nothing. Silent and completely dead. Added a
  local `_morlet2` (SciPy's own formula) used only when scipy no longer supplies
  one, so older installs are unaffected. Nine call sites redirected.
- `widths_changed` passed the 'scales' value straight to `re.findall`, so
  sending a LIST of numbers -- the obvious way to drive scales from another node
  -- raised TypeError. `_coerce_widths` already understood lists; both nodes now
  use it for list input and parse text otherwise.

Both covered by regressions. After the fix t.ultracwt runs: 600 frames in,
(1, 5) out for five scales, magnitudes 0.0001..0.1038.

Documented honestly rather than changed: on t.ultracwt the outlet named
'phase out' carries the IMAGINARY COMPONENT of the convolution, not an angle --
the atan2 line is commented out in the source. It moves with the phase and is
bounded by the magnitude rather than by pi. Left as-is because changing it would
alter numbers in any existing patch; the page says what it actually is.

## relayout segfault is intermittent

A full rebuild during this module hit 'RELAYOUT PRODUCED NOTHING (exit -11) for
194 paths' -- the guard added earlier working exactly as intended: it refused to
report success, and every patch then showed overlap warnings because layout had
not been applied. The same patch laid out fine alone, and simply re-running the
rebuild succeeded with all 238 clean. So: if that message appears, re-run before
investigating.

## torch_voxel_nodes (2026-09-03)

No bugs. The page's job was working out WHY these exist alongside pc_crop and
pc_voxel, since the point_cloud module docstring argues numpy wins straight off
a camera.

Two answers, and the second is the real one:

- Use them when the cloud is ALREADY a tensor -- out of a model or on its way
  into one -- so you are not paying to convert twice.
- `voxels out` is a DENSE 3D ARRAY, which the pc_ nodes cannot produce at all.
  Not a tidier cloud: a volume, depth by height by width, each cell holding how
  many points fell in it. That is what a 3D convolution or a learned model
  wants, and what you can slice and threshold like an image with an extra axis.

Measured, 5000-point scene cropped to a 2 m box at 0.2 m voxels:
    point cloud out  (2996, 3)     the surviving points
    voxels out       (10, 10, 10)  sum 2996, 300 cells occupied
    voxels cloud out (300, 3)      those 300 cells as points
Three views of one result, and they agree. Verified again in the built patch at
1 m / 0.2 m: (5, 5, 5) summing to 3000.

Worth documenting:
- Each output has its own switch and only 'output voxels cloud' starts on --
  building the dense grid and the reduced cloud are separate work.
- Bounds are six named numbers rather than min/max triples, and 'top' is the
  SMALLER of the vertical pair (screen convention, down positive). A crop box
  that seems vertically inverted is this.
- 'front' defaults to 0.1 m, not 0, because depth sensors return noise at very
  short range.
- These send PLAIN TENSORS -- no cloud-frame dict, so unlike pc_crop they do not
  tell anything downstream what volume to use. Mixing the families means a pc_
  node has no volume to inherit and falls back to its own widgets.

## vive_tracker_nodes (2026-09-03)

LINUX ONLY -- David confirmed it. These need OpenVR plus SteamVR and the
vendored `dpg_system.triad_openvr`, which is not in the repo; on this Mac
neither openvr nor triad_openvr is installed and dpg_app reports
'Skipped vive_tracker_nodes'. So the nodes DO NOT REGISTER here and could not be
created, let alone run.

Deliberately NOT done: no defensive-import change, no vendoring of
triad_openvr, no attempt to install openvr. The module being skipped with a
clear console line is correct behaviour for an unsupported platform, and making
the nodes register non-functionally on macOS would be worse, not better.

What that means for verification, stated plainly: the page's port names are
checked (validate_help reads them from the AST-extracted interface, which does
not need the module to import) and the patch loads clean. The WIRING and
BEHAVIOUR are unverified -- there is no way to run them here.

Tested first, and worth recording: relayout copes with a node that is not
registered. It keeps the authored position and size and measures the rest
normally, so a demo containing platform-specific nodes is shippable and will be
right on the platform that has them. Leave generous space around such a node,
since its real size cannot be measured.

Content, read from the source:

- The corner capture is the part worth learning: put the tracker at each floor
  corner in order FL/FR/BR/BL, then compute_from_corners derives centre, size,
  yaw and floor height in one go and fills the fields. It averages opposite
  edges, so a slightly trapezoidal capture still works. apply_chaperone pushes
  the result into SteamVR.
- vive_base_stations is a DIAGNOSTIC, not a source, and it separates two
  problems that need different remedies: `jitter_mm` is RMS spread about the
  window mean (reflections, blocked view, sunlight -- an environment problem),
  `drift_mm` is how far that mean has moved from a captured baseline (a bumped
  stand, a flexing truss, a building warming up -- a screwdriver problem).
  `all_stable` is the single yes/no to wire to a warning light.

## wavelet_nodes (2026-09-03)

No bugs. `t.cwt` is the THIRD wavelet transform in the system, so the page's
first job is placing it:

    cwt          whole window, several wavelet families (ssqueezepy)
    t.ultracwt   streaming, one sample at a time, newest column only
    t.cwt        whole window, Morlet only, the real parameters exposed, on GPU

The documentation contribution is translating the friendly labels back to the
standard names, because anyone who knows wavelets is looking for these:

- `sample_scaling` is dt -- seconds per sample. Measured: it does NOT change the
  output shape, only what the rows mean in Hz. So a wrong value is wrong
  SILENTLY, because the picture looks identical. The setting most likely to be
  left at its default by mistake.
- `scale_distribution` is dj -- octave spacing, and what sets the row count.
  Measured on 512 samples: dj 0.25 -> 33 rows, 0.125 -> 65, 0.0625 -> 129.
- `wavelet_constant` is w0 -- oscillations in the wavelet, the time/frequency
  resolution trade. 6 is conventional. It also shifts the scale range, so the
  row count moves with it: w0 12 gave 113 rows where 6 gave 129.
- Output is (batch, scales, time) -- (1, 65, 512) for one signal, unlike the cwt
  node which has no batch dimension. Needs a t.squeeze before a heat_map.

Caught by measuring rather than assuming: the magnitudes are NOT normalised. In
the demo they run 0..27.5 with mean 2.58, and I had set the heat_map to 0..0.5 --
which would have shipped a uniformly saturated display. Corrected to 0..10 and
the page now says to read the range off the data first. That is the third time
this project has hit the min-y/max-y trap; it earns its warning on the plot page.

## digico_nodes (2026-09-03)

No bugs. One node, `digico.fader` -- a bank of console channel faders in dB
(-80..+10), bidirectional over OSC.

Verified by probe:
- moving fader 3 sends /channel/3/fader -12.5 (addresses are 1-based)
- a LIST into 'fader 1' sets and sends every fader in order: eight values gave
  eight messages and eight widgets at -6..-13
- an incoming /channel/5/fader updates the slider correctly (the inlet stores
  [-3.0] but the widget shows -3.0 -- receive_data unwraps it, so no bug)

The demo mistake worth recording, because it is the EXACT INVERSE of the
prompt_nodes one: I first fed the bulk-set inlet from a `string` node. It wants
a LIST, so the whole line arrived as one piece, became 0.0, and set a single
fader. `message` splits into separate numbers and is correct here.

So the pair of rules:
  an inlet wanting a PHRASE must not come from `message` (it splits)
  an inlet wanting a LIST must not come from `string` (it does not)

Context worth documenting, from David's memory note: DiGiCo has no query API --
the console only announces a control when it moves, or when 'Resend All' is
pressed. So the setup sequence is connect, then Resend All, and only then does
the patch know the desk's state.

## nvx_nodes (2026-09-03)

No bugs. There is already a thorough NVX_KVM_README.md, so the help page
distils it for someone USING a working rig and points at the README for setting
one up.

Confirmed here: the node reads ~/.nvx_kvm.json and built real per-target inlets
('mac studio', 'linux') plus 'select'. Opening the help patch does not switch
anything -- the button callbacks check in_loading_process, which the README says
was learned the hard way, and smoke confirmed no switch fired.

The points worth carrying into the page, all from the README's measurements:

- Video and USB route INDEPENDENTLY, so a switch is two changes. A picture that
  changed but a mouse that did not is the USB pairing, not the subscription.
- Changing the target list means RE-CREATING the node: buttons are built in
  __init__ because node shape is fixed at creation. Reopening the patch is not
  enough.
- Switch time measured 0.8, 0.9, 3.5, 7.7 s. The long tail is the receiver
  reporting itself paired, not the picture arriving.
- IGMP snooping is a hard prerequisite, not a tuning option: a 4K60 stream is
  ~700 Mbps and without snooping it flooded every port and took all four units
  off the network until a transmitter was physically unplugged.
- HTTP 200 does NOT mean success on these devices -- a refused write returns 200
  and the real outcome is a per-property StatusId in the reply. Worth saying on
  the page because it explains why the status line matters.

Note that reading state is harmless, unlike visca_camera: this node polls
reachability, which cannot disturb anything, so the demo can be live.

## Grouping the remaining single-node modules (2026-09-03)

David asked whether it made sense to do several together. It does, on both
readings -- shared pages where the nodes answer one question, and batching the
work per turn. Of the last 15, four genuinely pair:

    whisper + eleven_labs        speech in and out (one topic, two directions)
    mgl_smpl_mesh + heatmap      both draw an SMPL body into the mgl chain
    ndi_receiver + depth_anything  looser: other ways to get a picture

The rest are singletons that share no reader's question: mgl_shader,
context_tracker, pybullet_body, movesense, gemma, noise_review, display_info,
t.data_set.

## whisper_nodes + elevenlabs_nodes (2026-09-03)

No bugs. The page's organising idea is the split that decides where each belongs:
whisper runs LOCALLY (no account, works unplugged), eleven_labs SENDS YOUR TEXT
to a service and needs a key in dpg_system/elevenlabs_key.py. Anything an
audience said in confidence can go through one and not the other.

The distinction worth the most space is whisper's two text streams:
`in_progress` is the live guess and gets revised; `phrases` is settled and
emitted once. Display in_progress, ACT on phrases -- acting on in_progress means
acting on words it is about to withdraw. They are not alternatives; the usual
arrangement uses both.

How a guess becomes a phrase, from the source: segments carry an AGE, and
`confirmation_age` is how old one must be before it is trusted; a phrase is
emitted when a confirmed segment ends in . ? or !. The required age SHRINKS as
the sentence lengthens (scaled by `length_factor`) because a long utterance has
already given the model context, and waiting the full age on every segment would
leave the transcript badly behind the speaker.

`noises` is not a fault: whisper hallucinates words out of breathing and room
tone, and those segments are routed there instead of into `phrases`.

## mgl_smpl_mesh + mgl_smpl_heatmap (2026-09-03)

No bugs. They are designed to STACK -- mesh first to draw the body, heatmap
after to lay a translucent torque-coloured skin over the same pose.

Documented rather than changed: `mgl_smpl_heatmap` labels its chain ports
'gl chain in/out' where every other mgl node says 'mgl chain'. It derives from
plain Node and hand-rolls its chain participation, so the naming is a leftover,
not the older gl system. Left alone because patches use those ports; the page
says so instead. Worth a rename with name_archive if David wants consistency.

Honest caveat carried from David's own notes: the muscle weight modes are a
fixed axis per muscle times joint torque, which tracks facing well and
mis-tracks arm elevation -- a single fixed axis cannot decompose that torque. The
page calls them expressive rendering, not anatomical measurement.

Validator catch: mgl_context's outlet is 'mgl_chain', not 'mgl chain out'.

## COMPLETE — 924 of 924 (2026-09-03)

Every registered node name in dpg_system now resolves to a help page. 254
patches, all validating clean, all node-fix regressions passing.

The last nine pages, written as a batch after David asked whether grouping made
sense:

- gemma + neuronpedia_search — REVISED PAIRING. `gemma` is Gemma 2 with Gemma
  Scope SAE feature steering, and `neuronpedia_search` returns
  [index, description, votes] — the very feature indices gemma's 'interventions'
  wants. They are two halves of one workflow, which only became clear on reading
  the source; my first grouping had them apart. Note the page warns that model
  and layer must MATCH, since the same index in another layer is a different
  concept and nothing warns you.
- ndi_receiver + depth_anything — other ways to get a picture. depth_anything is
  RELATIVE depth: good for masking and depth-order, not a substitute for femto
  when the question has a unit in it, and it has no memory between frames so a
  static object shimmers slightly.
- movesense, display_info, t.data_set — three singletons.
  `display_info` needs Quartz (pyobjc) on macOS and reports nothing without it —
  verified here, where it returns []. On Linux it reads xrandr.
  `t.data_set` genuinely has NO inlets and NO outlets; the page says so plainly
  rather than implying it is usable from a patch.
- mgl_shader — ShaderToy-compatible. The point worth the most: a shader that
  fails to compile leaves the PREVIOUS one running, so a silent failure looks
  exactly like a shader that did nothing. Watch 'status'.
- pybullet_body — the pelvis is driven and the rest simulated, which is what
  lets a recorded performance drive a body that can still overbalance.
- noise_review — review ONE flag category at a time, because the judgement
  differs per category; and walk 'clean sections' deliberately, since that is
  the only way to find false negatives.
- context_tracker — place and time are LADDERS held at several scales, so naming
  a room does not discard the season; moving place ejects props but keeps
  actors; the weights change assertion strength downstream, not what is tracked.

Validator catches in this batch, both mine: mgl_context's outlet is 'mgl_chain'
(not 'mgl chain out'), and the take player is `take`, not `open_take`.
