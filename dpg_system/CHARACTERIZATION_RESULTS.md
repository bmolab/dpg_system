# Torque field characterization — results

Companion to `GANG_NOTES.md`, which carries the formalism and the design
decisions. This file carries what the data says: the ranges any mapping needs,
which presets the corpus confirms, which look wrong, and how much of movement
the vocabulary cannot express.

**Sections are chronological and later ones correct earlier ones.** §7a is
superseded by §8c, §12d by §12g, and §6 is qualified by §11. Read §12g before
trusting §12d, and §13 before building anything on the dynamic stream.

The analysis scripts are NOT in the repo — they ran from a session scratchpad.
Method is recorded here in enough detail to rewrite them; the numbers are not
reproducible without doing so.

---

Corpus: `AMASS_Dynamic`, 14,279 files (5 unreadable), **20,405,738 frames**,
21 subsets. Streams: net (`total`), `gravity`, `dynamic`. Passive excluded —
90.9% zero, only 4 of 42 gangs ever see it non-zero.

Aggregation is exact, not sampled: log-magnitude histograms and cross-product
accumulators combine across files without approximation, so these numbers are
what a single pass over all 20.4 M frames would give. First 3 frames of each
file trimmed (derivative start-up).

Scripts (scratchpad): `characterize.py` (113 s on 8 workers), `analyze.py`.
Outputs: `conditioning.json`, `correlation.npz`, `per_file.npz` — per-file
records kept separable so any subset can be re-sliced without re-running.

---

## 1. Conditioning — the headline number is crest factor

Every gang is heavily skewed. Median crest (p99/p50) by stream:

    total      11.2      p99 spans 22x across gangs  (0.042 .. 0.944)
    gravity     8.9      p99 spans 28x across gangs
    dynamic    22.4      p99 spans 35x across gangs

A loud moment is typically **11x the median**, and on the dynamic stream 22x.
A linear map from `net` to any synth parameter therefore sits near the bottom
of its range essentially always, and the same is true inverted. This is the
single most consequential result for mapping, and it is direction-neutral:
it constrains direct, inverted and tangential mappings identically.

The 22–35x spread of p99 *between* gangs matters just as much. `leg_push|left`
peaks at 0.944 while `wrist_flex|common` peaks at 0.042. Patching them into
the same parameter with the same scaling means one saturates while the other
never leaves silence. **Per-gang normalization is not optional.**

Implication: condition on a percentile basis (the p50/p90/p99 table in
`conditioning.json` is exactly the lookup needed), and prefer a soft
compressive curve — sigmoid/Hill on the normalized value — over linear.
That matches the soft-valving preference already established for detector
scoring, for the same reason: retain magnitude, don't threshold.

## 2. Coherence works — but only where it can

Corrected from an earlier single-file reading. Split by term count:

    stream     multi-term (n=34)              single-term (n=8)
    total      coh_p10=0.215  coh_p50=0.905   all 1.0 by construction
    gravity    coh_p10=0.217  coh_p50=0.907   all 1.0 by construction
    dynamic    coh_p10=0.180  coh_p50=0.893   all 1.0 by construction

Single-term gangs (`hip_flex`, `leg_abduct`, `shoulder_girdle`, `wrist_flex`)
have coherence identically 1 — nothing to cancel. They carry no coherence
information and should not be patched as if they did.

Multi-term gangs do discriminate, mostly in the low tail: median p10 of 0.18–0.22
against a median p50 of ~0.90. So the useful reading of coherence is **"how far
below 1 does it dip, and when"**, not its central value. The most discriminating:

    leg_push|differential      p10=0.040  p50=0.370
    shoulder_girdle|common     p10=0.040  p50=0.360
    arm_reach|common           p10=0.015  p50=0.105
    contralateral_swing|diff   p10=0.070  p50=0.600
    leg_push|common            p10=0.155  p50=0.805

## 3. Structure — the field is NOT low-dimensional

    stream=total     56 live channels of 66
      PC1 10.2%   PC2 8.5%   PC3 7.2%   PC1-5 38.0%   PC1-10 57.8%
      participation ratio (effective dimensionality): 30.3 of 56

    stream=dynamic   60 live channels
      PC1 11.1%   PC2 9.2%   PC3 8.5%   PC1-5 42.1%   PC1-10 61.1%
      participation ratio: 29.6 of 60

**No small fixed basis captures human movement torque.** Ten components get
under 60%; effective dimensionality is ~30. This is direct evidence for the
long-term intuition that ganging should be fluid and gesture-dependent rather
than a fixed handful of groups — a static basis of any tractable size leaves
most of the field unexplained.

### The data's own top gangs

    PC1 (total)   spine3.x +.36  spine2.x +.33  spine1.x +.29
                  right_collar.x +.26  left_collar.x +.25   ...
    PC2 (total)   spine2.z +.39  spine1.z +.37  spine3.z +.36
                  neck.z +.32  head.z +.32
    PC3 (total)   neck.x +.30  head.x +.29  spine1.x +.27
                  collar.y +/-.26  shoulder.x/y +/-.25

PC1 is `spine_flex` — the three spine joints on x, in the same order and with
roughly the weights the preset guessed (1.0/1.0/0.7 vs .29/.33/.36). PC2 is
`spine_bend` **extended into neck and head**. So the spine gangs are confirmed
by the data as the dominant modes of the whole field, which is what GANG_NOTES
predicted was the strongest case.

## 4. Presets scored against the data

Two independent scores per gang: `var_ratio` (variance captured by that
direction ÷ variance of a random direction — how *present* this mode is in
real movement) and `coh_data` (mean pairwise correlation among the gang's own
channels, signed by its weights — whether the data agrees the terms belong
together the way they are weighted).

**Confirmed, both streams:**

    spine_flex               var_ratio 2.78 / 2.77   coh_data 0.93 / 0.92
    spine_bend               var_ratio 2.67 / 2.84   coh_data 0.86 / 0.92
    arm_elevate|differential var_ratio 2.52 / 2.56
    spine_twist              var_ratio 2.34 / 2.16   coh_data 0.74 / 0.66

**Rare modes (low var_ratio).** Most `common` and `differential` variants, and
`counter_rotation` (0.35/0.33). Low variance is not a defect — a differential
deliberately projects onto a subtle mode. But it *is* a conditioning fact:
these gangs are quiet most of the time and need much more gain than the loud
ones. Note `arm_elevate|differential` (2.52) beats `arm_elevate|common` (0.74):
asymmetric arm elevation is common in this corpus, symmetric is rare.

**Sign conflicts worth investigating** — single-side, non-differential gangs
whose weights fight the data's own co-variation:

    arm_reach|left    coh_data -0.39   arm_reach|right  -0.34
    leg_twist|left    coh_data -0.13   leg_twist|right  -0.13
    leg_push|left/right  -0.15 / -0.17  (dynamic stream only; total is +0.19/+0.14)

`arm_reach` weights shoulder-flex +1 against elbow-flex −1, but the two channels
correlate **positively** (r≈+0.39). GANG_NOTES flagged `leg_push` as the most
likely to be wrong; the data says it is fine on net/gravity and contrary on
dynamic, which is subtler than a flipped sign and needs a look.

Caveat: `coh_data` and the coherence statistic in §2 disagree for `arm_reach`
(coh_p50 = 0.905, i.e. it reads as coherent). They measure different things —
correlation is scale-free, coherence is magnitude-weighted, so a large shoulder
term can dominate a small elbow term and keep coherence high while the two
channels still co-vary against the weighting. Neither is wrong; the disagreement
itself is the finding.

## 5. Couplings the presets miss

Strongest channel pairs not spanned by any preset (total / dynamic):

    right_collar.x  right_shoulder.x   r=+0.891 / +0.920
    left_collar.x   left_shoulder.x    r=+0.885 / +0.928
    neck.z          head.z             r=+0.832 / +0.911
    left_elbow.y    left_wrist.y       r=+0.709 / +0.846
    right_hip.z     right_knee.z       r=+0.757 / +0.686

Three of these are actionable:

1. **Collar↔shoulder couples on x and y at r≈0.89–0.93, but `arm_elevate`
   gangs them on z (abduct).** The ganging decision was right — GANG_NOTES said
   "gang almost unconditionally" and the data agrees emphatically — but the
   *axis* may be the wrong one. The strongest coupling is not where the preset
   is looking.
2. **neck↔head couple more strongly on z (lateral) than the x that `head_flex`
   uses.** Same shape of issue.
3. **hip↔knee couple on z (abduction) at r≈0.75**, a leg coupling no preset
   captures; `leg_push` works on x.

`elbow.y↔wrist.y` at r=0.71–0.85 is real but the wrist was excluded from the
arm gangs deliberately, because on Shadow capture it carries the forearm/hand
yaw magnetisation error. AMASS has no such error, so this coupling is genuine
here and would still be a bad idea to rely on live.

---

---

# 6. Mechanical power (τ·ω) — full corpus, `power_analysis.py`

Torque is only the *force* half of every physical model's (force, velocity)
pair. Power is the product, and needs both halves of the archive.

Gang power uses the generalized-coordinate construction: for weights defining
θ_j = w_j q, the generalized force is Q = Σ w_j τ_j and the generalized
velocity q̇ = (Σ w_j ω_j)/‖w‖². Power along that coordinate is Q·q̇. Both sums
are linear, so the existing compiled bank computes them — run it twice, once on
torque and once on ω. Torque left unnormalized so results are in watts.

## 6a. Power is much WORSE conditioned than torque

    quantity          p50      p90      p99    p99.9   crest (p99/p50)
    torque (N.m)    3.350   37.584  105.925  188.365        31.6
    omega (rad/s)   0.596    3.350    8.414   18.837        14.1
    power (W)       0.531   11.885   94.406  298.538       177.8

Power is a product of two skewed signals and inherits both skews: crest 178
against torque's 32 and ω's 14. **Power is not a drop-in better control signal.**
It carries information torque cannot, but it needs distinctly more aggressive
compression — log or cube-root rather than the softer curve torque wants.

## 6b. Whole-body power is physiologically credible

    |total power|   p50 = 18.8 W   p90 = 133.4   p99 = 530.9   p99.9 = 1333.5
    generating p99 = 421.7 W       absorbing p99 = 335.0 W

Sustained human work is hundreds of watts and peaks exceed a kilowatt, so these
land where physiology says they should. That is independent corroboration that
the torque pipeline is producing real quantities, not just self-consistent ones
— a validation the whole chain had not previously had.

## 6c. The generating/absorbing sign IS usable — at articulation timescales

Instantaneously the split is 51.8% generating / 48.2% absorbing, which looks
like no information at all. It isn't: it is structured alternation.

    mean run length of constant sign : 115 ms  (~8.7 Hz)
    rectification |mean P| / mean |P| over a window:
        50 ms  : 0.933
       200 ms  : 0.826
      1000 ms  : 0.589

A 50 ms window is 93% one-directional, 200 ms still 83%. So **integrate power
over 50–200 ms and the generating/absorbing sign becomes a clean, slow, signed
control** — and that is a genuinely new dimension: torque magnitude cannot
distinguish driving a movement from braking it, and 50–200 ms is exactly the
timescale of a musical articulation. Even at 1 s there is net directionality
(0.589), so it degrades gracefully rather than washing out.

## 6d. Per-joint: legs dominate, ankles are impulsive

    joint            p50     p90      p99   p99.9  crest  absorb%
    right/left_hip  3.758  42.170  188.36  473.15   50.1   48.7/50.6%
    right_knee      2.661  29.854  149.62  375.84   56.2   48.8%
    right_ankle     0.841  21.135  149.62  421.70  177.8   49.3%
    pelvis          4.217  33.497  149.62  421.70   35.5   51.2%
    spine1          1.189  13.335   84.14  266.07   70.8   49.2%
    shoulders       ~0.8    ~8.0    ~50     ~188    63.1   ~47%
    elbows          0.531   4.217   23.71   94.41   44.7   ~48%
    wrists          ~0.1    ~0.7    ~4      ~25     ~40    ~49%

Hips carry the most power by a wide margin; arms are an order of magnitude
below the legs. The ankles stand out with **crest ~178–200** — by far the most
impulsive joint, which is push-off, and makes them the natural driver for
percussive models rather than sustained ones.

Absorb% is strikingly uniform at ~45–51% almost everywhere. The exception is the
shoulder girdle: collars absorb only 45.7% and `shoulder_girdle|common` only
39.0%, i.e. it is net-generating. Worth a look — likely gravity-dominated
postural load rather than a real asymmetry.

## 6e. The important result: differential gangs are quiet in torque, loud in power

Top gang power (watts), against the same gangs' rank in net torque:

    gang                          P p99   crest   absorb%   net-torque rank
    hip_flex|differential         237.14  158.5    50.3%    14th
    leg_push|differential         211.35  125.9    50.0%     3rd
    leg_push|right/left           167.88   89–100  49–51%    1st/2nd
    hip_flex|right                167.88  100.0    49.3%     7th
    contralateral_swing|left      167.88   56.2    46.8%     5th
    spine_flex|none               133.35  112.2    49.8%     4th
    ...
    wrist_flex|right                1.88   28.2    49.6%    last

`hip_flex|differential` is 14th by net torque and **1st by power**. That is the
complementarity worth exploiting: the differential (gait, alternation) modes are
low-variance directions in torque — §4 scored them "no better than a random
direction" — precisely because their torque is small, but they are where the
fast alternating *motion* lives, so their power is the largest in the body.

**Practical consequence: net torque is the right signal for postural/common
modes, and power is the right signal for differential/gait modes.** Neither
subsumes the other, and choosing per gang is a real design lever rather than a
preference.

Best-conditioned power gangs (lowest crest, easiest to map):
`contralateral_swing|common` 31.6, `head_flex` 31.6, `wrist_flex|right` 28.2.
Worst: `shoulder_girdle|common` 224, `spine_twist` 200, `arm_reach|common` 178.

Gang power spans 237 W down to 1.88 W — a **126x spread**, wider than net
torque's 22x. Per-gang normalization is even more necessary here.

---

# 7. Quality filtering — do the numbers survive the noise verdicts?

Everything in §1–6 pooled all 20.4 M frames, including material the noise work
already judged unusable. `noise_results_lenses_2026_06_17` carries a verdict for
14,182 of the 14,279 archive files (the 97 without one are exactly
`TCD_handMocap` + `PosePrior`, absent from the local AMASS tree).

    classification   files    frames
    clean             3,241    2.58 M
    moderate          9,129   12.29 M
    problematic       1,812    5.37 M     <- 26% of the corpus

DanceDB is 151/153 problematic with **zero** clean files.

Three regimes, `filtered_compare.py`:

    ALL     every frame                                  20,405,738  100.0%
    KEEP    clean+moderate, minus unusable segments
            and excision zones, frame by frame           14,690,920   72.0%
    CLEAN   clean only, same frame-level excision          2,572,589   12.6%

## 7a. Torque is essentially immune. The conditioning results stand.

    quantity        regime     p50      p90      p99    p99.9   crest
    torque (N.m)    ALL      3.350   37.584  105.925  188.365    31.6
                    KEEP     2.985   37.584  105.925  188.365    35.5
                    CLEAN    2.985   33.497  105.925  188.365    35.5

**p99 and p99.9 are identical to four significant figures across all three
regimes.** I expected the tails to be glitch-driven; they are not. Gang net
torque behaves the same way — `leg_push|right` p99 is 0.9441 in all three,
`leg_push|differential` 0.5957 in all three. Crest factors move by at most a
few percent. Everything in §1 and §4 survives unchanged.

## 7b. Angular velocity and power WERE inflated, at the far tail

    omega (rad/s)   ALL      0.596    3.350    8.414   18.837    14.1
                    KEEP     0.531    2.985    7.499   13.335    14.1
                    CLEAN    0.376    2.371    6.683   11.885    17.8

    power (W)       ALL      0.531   11.885   94.406  298.538   177.8
                    KEEP     0.422    9.441   74.989  237.137   177.8
                    CLEAN    0.266    7.499   74.989  266.073   281.8

    whole-body      ALL     p99 = 530.9   p99.9 = 1333.5
    power (W)       KEEP    p99 = 473.2   p99.9 = 1059.3
                    CLEAN   p99 = 473.2   p99.9 = 1188.5

ω p99.9 falls 37% from ALL to CLEAN, power p99 falls 21%, whole-body power p99
falls 11%. So the glitch contamination is real but confined to velocity and
its derivatives — which is exactly right, since teleportation is a *position*
artifact and shows up in the derivative, not in the torque magnitude.

The §6 conclusions hold: power crest is 177.8 under both ALL and KEEP, so
"power is much worse conditioned than torque" is not an artifact.

## 7c. But filtering introduces a selection bias, and it is large

Look at what moves and what doesn't, ALL → CLEAN:

    torque p50        -11%        torque p99        0%
    omega  p50        -37%        omega  p99      -21%
    power  p50        -50%        power  p99      -21%
    body power p50    -44%        body power p99  -11%

**The medians collapse while the peaks barely move.** The clean subset is
systematically *quieter*, not merely cleaner. That is the over-filtering error
predicted in advance: the noise detector false-positives on dynamic motion, so
`clean` skews hard toward low-energy material — KIT alone supplies 1,863 of the
3,241 clean files, and it is largely walking, while all of DanceDB is excluded.

CLEAN's apparently higher crest (282 vs 178 for power) is therefore an artifact
of that bias, not a property of clean data.

**Recommended regime: KEEP.** It retains 72% of frames, matches ALL exactly on
the torque tails, removes the ω/power contamination, and does not collapse the
medians the way CLEAN does.

## 7d. The finding that matters most survives

`hip_flex|differential` is the top power gang under **all three** regimes, and
is actually 12% *higher* under CLEAN (266 W) than ALL (237 W) — the least
glitch-driven gang in the table. The differential-gangs-are-loud-in-power
result of §6e is not a noise artifact.

Most glitch-inflated gang powers (ALL/CLEAN ratio): `leg_twist|common` 1.78,
`spine_bend` 1.58, `leg_abduct|common` and `leg_push|common` 1.41. Least:
`hip_flex|differential` 0.89, `leg_push|differential` 1.00, `hip_flex|left` 1.00.

## 7e. A coverage gap worth naming

KEEP excludes essentially all of DanceDB, and `problematic` claims 26% of the
corpus concentrated in the most energetic material. For a project about
translating *performance* movement into sound, dance is the most relevant
material available and it is also the most corrupted. The characterization
therefore describes everyday and athletic movement well and dance barely at
all. That is a limit on how far these ranges should be trusted for the
material the system is actually meant to play.

---

# 8. Correction — problematic files have clean sections, and §7 discarded them

§7's `KEEP` regime excluded `problematic` files **entirely**, clean stretches
included. That was wrong. The noise records carry `clean_segments`, a per-frame
verdict that exists for problematic files too:

    class          total frames   inside clean_segments
    clean               2.58 M      2.47 M   (95.5%)
    moderate           12.29 M      8.44 M   (68.7%)
    problematic         5.37 M      1.73 M   (32.3%)   <- discarded by §7

Recoverable fraction varies enormously by subset: KIT 78.6%, MPI_HDM05 73.6%,
TotalCapture 69.2%, Eyes_Japan 63.0%, CMU 47.6% — against **DanceDB 8.3%**. So
DanceDB genuinely is mostly unusable, but several subsets were being thrown
away for no reason.

Note also that `clean_segments` covers only 68.7% of *moderate* frames, so it is
a **stricter and more uniform** criterion than §7's whole-file filter, not a
looser one.

Two new regimes:

    SEG    clean_segments from EVERY file regardless of classification
                                                   12,570,351   61.6%
    SEGP   only the SEG frames contributed by problematic files
                                                    1,719,431    8.4%

## 8a. The discarded material was the most energetic in the corpus

    quantity      regime      p50      p90       p99     p99.9   crest
    torque (N.m)  ALL       3.3497  37.5837  105.9254  188.3649   31.6
                  SEG       3.3497  33.4965  105.9254  188.3649   31.6
                  SEGP      3.3497  33.4965  105.9254  211.3489   31.6   <--
                  CLEAN     2.9854  33.4965  105.9254  188.3649   35.5

    omega         SEGP      0.4732   2.6607    7.4989   13.3352   15.8
                  CLEAN     0.3758   2.3714    6.6834   11.8850   17.8

    power (W)     SEGP      0.4217   8.4140   74.9894  266.0725  177.8
                  CLEAN     0.2661   7.4989   74.9894  266.0725  281.8

**`SEGP` torque p99.9 is 211.3 — the highest of any regime, higher than ALL.**
The clean sections of problematic files contain the largest torque peaks in the
corpus. Their ω median is 26% above the clean files' and their power median 58%
above.

This is the over-filtering error of §7c, measured: the most energetic material
is the most likely to be flagged problematic, so filtering by file-level
classification systematically removes real high-effort movement. `CLEAN`'s
inflated crest (281.8 for power against SEGP's 177.8) is confirmed as a
selection artifact.

## 8b. Nothing about the gangs changes

Across all five regimes, gang net torque p99 is stable to the bin width —
`leg_push|right` 0.9441 in every regime, `leg_push|differential` 0.5957 in
every regime, `spine_flex` 0.5957 in four of five. `hip_flex|differential`
remains the top power gang everywhere. The §1/§4/§6 conclusions are unaffected
by any filtering choice.

## 8c. Revised recommendation

**Use `SEG`** — `clean_segments` applied uniformly to every file regardless of
its file-level classification. It is a per-frame criterion rather than a
per-file one, it is stricter than §7's `KEEP` inside moderate files, and it
recovers the 1.72 M frames of genuinely high-energy movement that a
classification filter discards. §7c's recommendation of `KEEP` is superseded.

The dance coverage gap of §7e narrows but does not close: DanceDB contributes
125 k usable frames out of 1.50 M.

## 8d. Why torque was insensitive (§7a)

Torque carries noise rejection of its own, which is the better explanation for
§7a than the one given there. The recorded options show `acc_smooth_window = 7`
(Savitzky-Golay derivative window) active, along with passive-limit clamping,
so single-frame position glitches are already substantially attenuated before
they reach the torque field. Angular velocity in this archive receives no such
treatment, which is why it — and power through it — is where the residual
contamination shows up.

---

# 9. Correlation structure recomputed under SEG — §3 and §4 confirmed

The gap left open above is now measured rather than inferred. `characterize.py
--regime SEG` over 12,570,293 frames (1,630 files skipped: no clean_segments,
or unrated). Output in `seg/`.

## 9a. Dimensionality: unchanged

    stream=total          ALL          SEG
    PC1                  10.2%        10.6%
    PC2                   8.5%         8.5%
    PC3                   7.2%         7.4%
    PC1-5                38.0%        38.7%
    PC1-10               57.8%        58.5%
    participation ratio   30.3         29.6   (of 56 live channels)

The central structural finding of §3 — **no small fixed basis captures the
torque field** — is unaffected. Ten components still fall short of 60%.

## 9b. The data's own top gangs: same channels, same order

    PC1 (10.6%)  spine3.x +.34  spine2.x +.32  right_collar.x +.28
                 left_collar.x +.27  spine1.x +.26  ...
    PC2 ( 8.5%)  spine2.z +.39  spine1.z +.36  spine3.z +.35
                 neck.z +.32  head.z +.31
    PC3 ( 7.4%)  spine1.x +.30  neck.x +.30  head.x +.29
                 collar.y +/-.27  shoulder.y +/-.25

Identical to ALL in composition and ordering; loadings move by at most 0.03.
PC1 is still `spine_flex` with the collars attached, PC2 still `spine_bend`
extended into neck and head.

## 9c. Preset scores: stable to the second decimal

    gang                      coh ALL  coh SEG  var ALL  var SEG
    spine_flex|none             0.931    0.927     2.78     2.77
    spine_bend|none             0.861    0.848     2.67     2.65
    arm_elevate|differential    0.475    0.510     2.52     2.63
    spine_twist|none            0.739    0.767     2.34     2.38
    ...
    arm_reach|left             -0.392   -0.394     0.61     0.60
    counter_rotation|none      -0.654   -0.674     0.35     0.33

Every confirmed preset stays confirmed and every sign conflict stays a sign
conflict. Two gangs cross my arbitrary 1.2 verdict threshold on the dynamic
stream (`arm_elevate|common` 1.21→1.14, `contralateral_swing|differential`
1.20→1.25) — that is the cutoff being arbitrary, not the data changing.

The largest genuine movers are the `leg_push` family, all downward:
`leg_push|differential` 1.71→1.51, `leg_push|left` 1.46→1.34, `leg_push|right`
1.39→1.30. So a little of leg_push's apparent variance capture was
noise-driven, though it remains above the random baseline.

## 9d. Uncovered couplings: same list, same magnitudes

    pair                                   ALL       SEG
    right_collar.x  right_shoulder.x     +0.891    +0.892
    left_collar.x   left_shoulder.x      +0.885    +0.879
    neck.z          head.z               +0.832    +0.834
    left_collar.y   left_shoulder.y      +0.827    +0.831
    right_hip.z     right_knee.z         +0.757    +0.736

All three actionable findings of §5 stand: collar↔shoulder couples at r≈0.89 on
**x and y** while `arm_elevate` gangs them on **z**; neck↔head couples more
strongly on z than the x `head_flex` uses; hip↔knee couple on z at r≈0.74 with
no preset covering it.

## 9e. Conclusion

Quality filtering changes none of the structural conclusions. Combined with
§7a and §8b, **every finding in this document is independent of the filtering
regime.** The one thing filtering did change is the §7c/§8c recommendation
about which frames to use going forward, and the §8a discovery that the
highest-energy usable material lives inside problematic files.

---

# 10. Reframing — low variance is the signal, not the defect

**§4's `var_ratio` verdicts are written backwards and should be read inverted.**
Scoring gangs by how much corpus variance their direction captures implicitly
treats "this is what bodies usually do" as the goal. But a gang's activation
only says the body did the expected thing; its *violation* says something is
happening. Surprise is −log p, so the rare direction carries the most
information per event.

So `counter_rotation` (var_ratio 0.33), `arm_reach|common` (0.29),
`shoulder_girdle|common` (0.69) and most `common`/`differential` variants are
not "no better than a random direction" — they are **contradiction detectors**.
They are quiet precisely because what they detect is rare, and that is what
makes their firing informative. The §1 conditioning consequence is unchanged
(they still need much more gain than the loud gangs); only the interpretation
of their worth flips.

This does not retract the *sign-conflict* findings, which are a different
measurement: `coh_data` says the weights fight the data's own co-variation, and
that remains a reason to look at `arm_reach` and `leg_twist` regardless.

## 10a. The prior is already measured

"How much does this movement contradict the statistically likely gang" is
directly computable from what §9 produced. Project the current torque vector
onto the covariance eigenbasis, weight each component by 1/√λ, take the
magnitude: movement loading onto low-eigenvalue directions scores high by
construction. That is a whitened surprise / Mahalanobis departure, it is
data-derived rather than hand-specified, and it commits to no mapping.

`correlation.npz` holds the 66×66 covariance per stream and per subset over
12.57 M clean-segment frames. `torque_residual` in `gang_nodes.py` is the
existing node closest to this.

It should be a **companion output** alongside net / total / coherence, not a
replacement — a gang would then report both what it did and how unusual that
was.

## 10b. Two cautions

- A whitened metric amplifies exactly the low-variance directions where mocap
  noise also lives. The covariance here was computed on noise-filtered frames,
  but a live Shadow signal is dirtier; the surprise measure will need its own
  noise floor.
- Per-subset covariances are already stored. Whether the prior should be one
  global corpus statistic or conditioned on movement context is exactly the
  fluid/gesture-dependent question from the project scope — and it is now one
  slice away from being answerable.

---

# 11. Re-run on the rate-corrected archive (v3)

The archive was reprocessed with `acc_smooth_ms=70.0` (commit `8896ffa`), which
fixes the Savitzky-Golay acceleration window at 70 ms rather than 7 frames.
Everything in §1–§10 was recomputed. **No conclusion changes.**

## 11a. The reprocess is verified correct

`compare_fingerprint.py` against 36 files fingerprinted before the swap
(12 each at 60/100/120 Hz), checking a falsifiable pattern rather than merely
"something changed":

    stream                 60 Hz     100 Hz     120 Hz     verdict
    torque              12 diff    12 SAME    12 diff     as predicted
    torques_dyn_vec     12 diff    12 SAME    12 diff     as predicted
    torques_grav_vec    12 same    12 same    12 same     as predicted
    angular_velocity    12 same    12 same    12 same     as predicted
    recorded options:   acc_smooth_window=0  acc_smooth_ms=70.0

**The 100 Hz files are bit-identical**, confirming 70 ms resolves to exactly
the legacy 7 frames at Shadow's rate and that nothing else drifted.

Two predictions of mine were wrong, both my error rather than the data's:

- `torques_passive_vec` DID change at 60/120 Hz. Passive is not purely
  pose-based — `smpl_processor.py:6901` computes it from `t_net_all`, so it is
  downstream of the acceleration smoothing. It follows the same rate pattern as
  dynamic, which confirms rather than contradicts.
- `com_acc` did NOT change at any rate. The stored value comes from the
  One-Euro CoM path, not the SG window, and that filter already sets its
  `_freq` from `dt`. **`com_acc` was never subject to this bug.**

Corrected: the acceleration-derived set is `torque`, `torques_dyn_vec` and
`torques_passive_vec`. `com_acc` is on a separate chain.

## 11b. Exact magnitude of the change

Measured on the 36 fingerprinted files with real percentiles (the corpus
histograms have 0.05-dex bins = 12% granularity, too coarse to resolve this):

    stream                 60 Hz p50/p99      100 Hz      120 Hz p50/p99
    torques_dyn_vec       +3.32% / +2.92%   0.00%/0.00%   -8.10% / -2.93%
    torque                -0.56% / +0.20%   0.00%/0.00%   +0.09% / -0.12%
    torques_grav_vec        0.00% /  0.00%   0.00%/0.00%    0.00% /  0.00%
    angular_velocity        0.00% /  0.00%   0.00%/0.00%    0.00% /  0.00%
    com_acc                 0.00% /  0.00%   0.00%/0.00%    0.00% /  0.00%

60 Hz dynamic torque rose (window 7→5, less smoothing); 120 Hz fell (7→9, more
smoothing). **The two rates moved toward each other**, which is the entire
point. The controlled same-motion test in the commit measured the resulting
consistency directly: p99 60/120 from 0.909 to 0.959, high-frequency content
from 0.613 to 0.863.

## 11c. Why the corpus conclusions did not move

**Net torque is gravity-dominated, and gravity is untouched.** Dynamic torque
shifted by ~3% at the affected rates, but its contribution to `torque` is small
enough that net moved by ±0.2%. Since §1's conditioning and §3–§4's structure
are computed on net (and on gravity), they were always insulated from this bug.

    conditioning (ALL)      v2            v3
      median crest total    11.2          10.6
      median crest dynamic  22.4          22.4
      median crest gravity   8.9           8.9
      p99 span total        0.042-0.944   0.042-0.944

    structure (SEG, total)  v2            v3
      PC1/PC2/PC3           10.6/8.5/7.4  10.6/8.5/7.4
      PC1-10                58.5%         58.5%
      participation ratio   29.6          29.6

    power (ALL)             v2            v3
      torque/omega/power crest  31.6/14.1/177.8  identical
      whole-body p99        530.9 W       530.9 W
      generating fraction   51.8%         51.9%
      sign run length       115 ms        116 ms
      rectification 50/200/1000 ms  .933/.826/.589  .936/.830/.595

Preset scores move by at most 0.04 in `var_ratio`. Two `coh_data` sign flips
appear (`leg_abduct|common` −0.030→+0.006, `leg_abduct|differential`
+0.030→−0.006) but both are values sitting on zero crossing it by noise, not
findings changing.

Rectification improved very slightly at every window (.933→.936, .826→.830,
.589→.595), consistent with marginally cleaner acceleration.

## 11d. What the fix is actually for

Not the corpus statistics — those were insulated. It matters for
**comparability between capture rates**, and specifically between this archive
and live Shadow. Before the fix, dynamic torque from a 120 Hz AMASS file and a
100 Hz Shadow stream were smoothed over different durations, so conditioning
constants derived from one would be systematically wrong for the other. They
now agree to within the odd-window quantization (70 ms at 100 Hz, 75 ms at
120 Hz).

Minor: v3 skipped 8 files where v2 skipped 5, 20,405,685 frames against
20,405,738 — a 53-frame difference, negligible but noted.

---

# 12. Whitened surprise — built and validated

The §10 measure, implemented. `build_torque_prior.py` (corpus prior),
`surprise_core.py` (runtime), `validate_surprise.py` (the four checks below).

    z = Λ^(-1/2) Vᵀ (x − μ)        d = ‖z‖

Directions the corpus rarely uses have small λ, so movement loading onto them
produces large `z`. Mahalanobis distance is invariant under per-channel
scaling, so normalizing torque by max_torque first makes no difference and is
not done.

Prior: 56 live channels of 66, over the 12,570,293 SEG frames. Raw condition
number 1.5e5, capped at 1000 by flooring eigenvalues at `ridge × λ_max`.

## 12a. Surprise is the best-conditioned signal measured so far

    corpus distribution of d:  p50 = 4.52   p90 = 8.04   p99 = 18.41

Crest (p99/p50) is **4.1**, against 11 for gang net torque and 178 for power,
and it stays 3.5–3.8 across every ridge setting. Whitening normalizes by
construction, so the §1 conditioning problem largely does not apply here. A
`percentile()` mapping through the corpus distribution is also provided, giving
a value already in [0,1] — "more unusual than X% of recorded movement" — which
is what a patch should generally consume.

## 12b. Calibration

Decile occupancy of `percentile()` on held-out corpus frames, 0.100 being
perfect:

    0.093 0.097 0.093 0.093 0.092 0.113 0.098 0.090 0.116 0.115
    mean = 0.517 (expect 0.500)

Close to uniform, mildly top-heavy. Good enough to trust as a scale; not exact,
because per-file percentiles are pooled across files of very different lengths.

## 12c. Noise sensitivity — the §10b caution, quantified

The stated worry was that whitening amplifies the low-variance directions where
mocap noise also lives. Measured against the noise work's own verdicts:

    population        files   d p50   d p90   d p99
    clean               125    4.20    6.48    8.02   1.00x
    moderate            361    4.57    6.84    8.32   1.09x
    problematic          60    5.31    6.28    7.72   1.26x
    <excised>            72    6.62   10.09   12.19   1.58x

`<excised>` are the frames the noise work actually cut — known-bad data. They
score **1.58× clean**, so surprise does respond to mocap noise, but only
moderately: it is not simply a noise detector wearing a different name. Clean
sections of problematic files sit at 1.26×.

**Live use will need its own noise floor**, and the gap between 1.0 and 1.58 is
the size of the problem to be solved.

## 12d. Discrimination — it separates movement, interpretably

Median `d` by subset, clean_segments only:

    Transitions           6.42      <- highest
    ACCAD                 6.33
    BMLHandball           6.12
    MPI_HDM05             5.23
    HUMAN4D               5.18
    DanceDB               5.18
    BMLMovi               4.94
    CMU                   4.93
    BioMotionLab_NTroje   4.64
    Eyes_Japan_Dataset    4.64
    EKUT                  3.87
    KIT                   3.57      <- lowest

The ordering is meaningful, not arbitrary. **Transitions** — a subset that is
literally movement transitions — scores highest, which is exactly where a body
does something it does not usually do. ACCAD (crawling, cartwheels, martial
arts) and BMLHandball (throwing) follow. **KIT, overwhelmingly walking, scores
lowest.** Walking is what bodies usually do, so it should be unsurprising, and
it is.

## 12e. Decomposition — how much the gangs cannot say

A gang with weights g reads g·x, whose whitened direction is Λ^(1/2) Vᵀ g. The
span of all 42 declared gangs is a subspace; splitting z against it gives
`d_gang` (surprise the vocabulary can express) and `d_free` (surprise no gang
can express).

    42 gangs -> rank 24 of 56 whitened dimensions (43% of the space)
    median d_total = 4.50
    median d_gang  = 3.14
    median d_free  = 2.87
    share of surprise the gangs CANNOT express: 45.8%

The 42 gangs collapse to rank 24 — heavy redundancy, as expected once bilateral
variants share terms. They capture 54.2% of surprise using 43% of the
dimensions, so they do beat a random subspace of equal rank (which would leave
~57% free), but only modestly.

**Nearly half of all surprise lives in directions no declared gang can
express.** That is a direct measurement of the §10 point: the vocabulary has no
word for a large share of what makes movement unusual, and `d_free` is the
signal that says so.

## 12f. Ridge — the one tuning knob, and where it should sit

Whitening divides by √λ, so the floor on λ decides how much the rare directions
are amplified. That is the noise/sensitivity trade directly:

    ridge   floored   excised/clean   Transitions/KIT   free share   crest
    1e-2     37/56         1.39            1.72           41.2%       3.5
    1e-3     20/56         1.47            1.85           45.4%       3.6
    1e-4      8/56         1.58            1.82           47.6%       3.7
    1e-5      1/56         1.59            1.83           50.9%       3.8

**1e-3 is the sweet spot** and is the default. Discrimination peaks there
(1.85), while loosening further only buys more noise pickup for no gain, and
tightening to 1e-2 costs real discrimination. Conditioning is insensitive
throughout.

## 12g. CORRECTION — surprise is substantially magnitude in disguise

A check I should have run before writing §12d. Spearman correlation between `d`
and plain torque magnitude ‖x‖:

    stream     per-file median   p10     p90    pooled
    total          0.774        0.251   0.921   0.828
    dynamic        0.920        0.767   0.974   0.962

So a large `d` often just means "a lot of torque". §12d's subset ordering is
therefore not, on its own, evidence that the measure detects unusualness.

**It survives the control, but only when the control is applied.** Across
subsets, Spearman(median d, median ‖x‖) is **0.549** — moderate, not 1.0 — so
the ordering is not simply magnitude. And KIT and EKUT (‖x‖ ≈ 80) score
3.57/3.87 where Eyes_Japan and BMLMovi at the same magnitude (78, 79) score
4.64/4.94. At equal torque, locomotion is less surprising. That part is real.

**Dividing magnitude out gives a better answer than raw d.** Ranked by
`d/‖x‖` — surprise per unit torque:

    DanceDB              0.0782     <- highest
    Eyes_Japan_Dataset   0.0609
    MPI_HDM05            0.0584
    BMLMovi / DFAUST     0.0567
    ...
    CMU                  0.0489
    KIT                  0.0444
    BioMotionLab_NTroje  0.0435
    EKUT                 0.0433     <- lowest, all locomotion

**Dance comes top and the locomotion subsets come bottom** — which is the
answer one would want, and which the raw `d` ranking missed entirely because
DanceDB's usable frames carry the lowest torque magnitude in the corpus (61.9)
and so scored mid-table. Added as `shape_surprise()` in `surprise_core.py`.

Revised guidance: use `shape_surprise()` when the question is "is this
configuration unusual"; use `surprise()` when absolute departure is wanted.

## 12h. Not yet done

Built as standalone modules, not wired into `gang_nodes.py`. The intended shape
is a companion output — a gang reporting `net`/`total`/`coherence` alongside
`surprise` and `free` — plus a whole-field surprise node. The prior is a 56×56
matrix and one matvec per frame, so runtime cost is negligible.

---

# 13. Dynamic-stream prior — built, and NOT recommended

Built the same way: 12,570,293 SEG frames, ridge 1e-3. **60 live channels
against total's 56** — the elbow x/z channels are live here, because the hinge
zeroing applies to net torque and not to dynamic.

Calibration is better than the total prior's (deciles 0.091–0.110, mean 0.493
against 0.517). Everything else is worse.

    measure                          total        dynamic
    correlation with ||x|| (pooled)   0.828        0.962
    excised/clean noise ratio         1.58x        2.79x
    share of surprise gangs miss      45.8%        38.1%

**At rho = 0.962 the dynamic prior is very nearly plain dynamic-torque
magnitude.** Whitening adds almost nothing, so anything wanted from it is
available more cheaply and more legibly by taking ‖dynamic torque‖ directly.

**And it is far more noise-sensitive**: excised frames score 2.79x clean,
against 1.58x for total. That is expected — dynamic torque is acceleration-
derived and mocap glitches are acceleration spikes — but it makes the dynamic
prior the worse of the two on exactly the axis §10b warned about.

The discrimination table also reads wrongly: DanceDB scores **0.34**, lowest by
a wide margin, and BMLHandball falls from 6.12 on total to 1.89. The reason is
structural rather than a bug — dynamic torque is near zero at rest, so a
subset whose usable frames are mostly its quiet residue scores near zero. Since
`clean_segments` preferentially retains low-motion frames, the subsets with the
most flagged content get measured on their stillest material.

**Why total works and dynamic does not:** total torque contains gravity, which
is posture-dependent and non-zero at rest, so the prior's mean is meaningful
and deviations from it have shape. Dynamic torque has a near-zero mean and
grows in all channels together with motion intensity, so distance from the mean
is mostly just speed.

Keep `torque_prior.npz` (total). `dynprior_r1e-3.npz` is retained for the
record but should not be used as a surprise measure. A `gravity` prior was not
built; by the same argument it would likely behave like total, being the
posture-carrying half.

---

## Open / next

- Time-scale (`rate_hz`) is in `per_file.npz`, not yet reduced. Note it is not
  valid under SEG as computed — `np.diff` runs across segment boundaries, so a
  masked regime needs the derivative taken within contiguous runs.
- Per-subset correlation matrices are stored under both regimes but not
  compared; whether structure shifts by gesture type is still unanswered.
- Per-subset correlation matrices are stored but not compared — the question of
  whether co-variation structure shifts by gesture type is one slice away.
- The archive ran with rate limiting / jitter damping / KF / one-euro all OFF,
  so these ranges are **upper bounds** relative to a filtered live patch.
