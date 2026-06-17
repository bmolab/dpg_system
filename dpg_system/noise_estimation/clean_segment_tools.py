"""Offline re-derivation of clean segments from a serialized noise report.

The expensive pipeline (estimate_noise_torque.py) emits, per file, the location
AND severity of every artifact it found: kinematic lens clusters
(teleport/flicker/zigzag/ROM/reversal, each a LensCluster with a normalised
`severity`), spike clusters (with `spike_density` / `contaminated`), and the
structural anomalies (corruption zones, stream breaks, torque glitch clusters).

`rederive_clean_segments` rebuilds the clean-segment list from that data alone,
in milliseconds, at a tunable tolerance — no re-running of the pipeline.  This
lets a reviewer dial how much minor spiking is acceptable inside a "clean"
segment: artifacts whose severity falls under the tolerance are absorbed, and
the clean segments on either side join across them.

All inputs are plain dicts/lists as loaded from the result JSON, so this works
on any already-processed file.  Backward compatible with old reports whose lens
clusters are bare ``[start, end]`` pairs (no severity) — those are treated as
hard breaks, since there is no severity to threshold against.
"""

from typing import List, Dict, Any


# Hard lenses: genuine motion-invalid events that always break a clean segment
# — a non-physical arm jump (teleport), a candy-wrapper twist (ROM), an
# unbounded reversal-at-speed (reversal), and an isolated recovering glitch
# (excursion: validated high-precision, 0 on clean ballistic motion, catches
# the moderate-speed reversals the others miss).  The tolerance dial does NOT
# absorb these — for extracting motion-valid sequences they must fragment.
HARD_LENS_FIELDS = ('teleport_clusters', 'rom_clusters', 'reversal_clusters',
                    'sfglitch_clusters', 'instability_clusters')

# Soft lenses: graded "minor spiking" (synchronized marker jitter, zigzag buzz)
# whose severity the tolerance dial modulates — absorbed when at/under tol so
# the clean segments join across them.
SOFT_LENS_FIELDS = ('flicker_clusters', 'zigzag_clusters')

LENS_CLUSTER_FIELDS = HARD_LENS_FIELDS + SOFT_LENS_FIELDS

# Human-readable labels for the per-cut readout (which lens caused a break).
_LENS_LABEL = {
    'teleport_clusters': 'teleport', 'rom_clusters': 'ROM',
    'reversal_clusters': 'reversal', 'sfglitch_clusters': '1-frame',
    'instability_clusters': 'instability', 'flicker_clusters': 'flicker',
    'zigzag_clusters': 'zigzag', 'spike_clusters': 'spike',
    'excursion_clusters': 'excursion', 'corruption_zones': 'corruption',
    'glitch_clusters': 'torque-glitch', 'stream_breaks': 'stream-break',
}


def _cluster_span(c):
    """(start, end) of a cluster that may be a LensCluster dict or a tuple."""
    if isinstance(c, dict):
        return int(c.get('start', 0)), int(c.get('end', c.get('start', 0)))
    if isinstance(c, (list, tuple)) and len(c) >= 2:
        return int(c[0]), int(c[1])
    return None


def rederive_clean_segments(report: Dict[str, Any], *,
                            off_traj_cm_tol: float = 1.0,
                            lens_severity_tol: float = 0.0,
                            spike_density_tol: float = 1.0,
                            keep_contaminated_spikes: bool = True,
                            movement_rescue_prob: float = 0.89,
                            include_structural: bool = True,
                            join_gap_s: float = 0.0,
                            min_clean_s: float = 1.0,
                            margin: int = 3,
                            cuts_out: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Re-derive clean segments from a report dict at a chosen tolerance.

    Tolerances (raise to be MORE permissive — absorb more, join more):
      off_traj_cm_tol : the PRIMARY spike/excursion relevance dial.  A spike or
          excursion event fragments a clean segment only if its off-trajectory
          contextual residual (perceptibility.off_traj_cm) reaches this many cm
          — a real, perceptible glitch.  Below it = imperceptible, absorbed.
          Default 1.0 cm (validated: real glitches 2-16 cm, smooth/clean/
          imperceptible 0-0.7 cm; 0.7-1.5 is the fuzzy margin).  Falls back to
          the contaminated/severity rules when off_traj_cm is unavailable.
      lens_severity_tol : a lens cluster breaks a clean segment only if its
          `severity` exceeds this.  severity is normalised so ~1.0 == the lens
          flag threshold; e.g. 1.4 absorbs single-frame flickers (sev ~1.33)
          while keeping multi-joint bursts (sev ~1.67).  0.0 = break on every
          flagged cluster.  Old reports without severity always break.
      spike_density_tol : an *uncontaminated* spike cluster breaks only if its
          `spike_density` exceeds this (density is in [0,1]; default 1.0 absorbs
          all uncontaminated spikes — lower to fragment on dense jitter).
      movement_rescue_prob : the movement-vs-glitch combiner valve.  A spike or
          excursion cluster whose serialized `movement_prob` (0..1, from
          movement_combiner) is at or above this is ABSORBED as real movement —
          rescued from fragmentation even if its off-trajectory residual is
          perceptible.  This is the throw/landing/complex-motion FP fix.  Default
          0.89 is conservative (rescues only the clearest movements, ~0 glitch
          leak); lower it to rescue more aggressively, raise above 1.0 to disable.
          No-op on old reports (movement_prob unset / -1).
      keep_contaminated_spikes : OVERRIDE on the off-traj dial.  If True, a
          contaminated (dense-jitter) spike cluster fragments even when its
          off-trajectory residual is below off_traj_cm_tol — the dense jitter is
          treated as motion-invalid regardless of a perceptible excursion.  If
          False, off_traj alone governs (a contaminated-but-imperceptible region
          — e.g. dense low-amplitude jitter at 0.13 cm — is recovered/absorbed).
      include_structural : if True, corruption zones / stream breaks / torque
          glitch clusters always break (these are hard, non-tunable evidence).
      join_gap_s : merge clean segments separated by a SHORT, SOFT-ONLY bad gap
          (shorter than this many seconds AND containing no hard artifact).
          Hard artifacts — teleport/ROM/reversal, corruption zones, stream
          breaks, contaminated spike clusters — are NEVER bridged regardless of
          gap length, so a brief genuine glitch is never swallowed into a clean
          segment.
      min_clean_s : discard clean segments shorter than this.
      margin : expand each bad region by this many frames on each side.
      cuts_out : if a list is passed, it is populated (one dict per artifact that
          actually breaks a segment at the current settings) with
          {start, end, lens, tunable, dial, note} — `tunable` False = a HARD lens
          (always cuts), True = a re-derivation dial can absorb it (`dial` names
          which).  This is the per-cut "why was this cut, and can I tune it away?"
          readout; only contributing cuts are recorded (absorbed/rescued ones
          aren't), so it reflects the live dial state.

    Returns clean segments as dicts {start, end, n_frames, duration_s},
    sorted longest-first.
    """
    T = int(report.get('n_frames', 0))
    fps = float(report.get('fps') or 0.0) or 120.0
    if T <= 0:
        return []

    # Two masks: HARD frames are genuine glitches/corruption that may NEVER be
    # bridged (no matter how brief); SOFT frames are above-tolerance minor
    # spiking that join_gap is allowed to span.  bad = hard | soft.
    hard = bytearray(T)
    soft = bytearray(T)

    def mark(buf, s, e):
        a = max(0, int(s)); b = min(T - 1, int(e))
        for i in range(a, b + 1):
            buf[i] = 1

    def record(field, s, e, tunable, dial='', note=''):
        """Log a contributing cut for the per-cut readout (no-op if cuts_out
        wasn't requested)."""
        if cuts_out is None:
            return
        cuts_out.append({'start': int(s), 'end': int(e),
                         'lens': _LENS_LABEL.get(field, field),
                         'tunable': bool(tunable), 'dial': dial, 'note': note})

    # ── Hard lenses — physically-impossible / excised, always break ─────
    if include_structural:
        for fldname in HARD_LENS_FIELDS:
            for c in report.get(fldname, []) or []:
                span = _cluster_span(c)
                if span is not None:
                    mark(hard, *span)
                    sev = c.get('severity') if isinstance(c, dict) else None
                    record(fldname, span[0], span[1], False,
                           note=(f"sev={sev:.2f}" if isinstance(sev, (int, float)) else ''))

    # ── Soft lenses — graded minor spiking, severity-thresholded ────────
    for fldname in SOFT_LENS_FIELDS:
        for c in report.get(fldname, []) or []:
            span = _cluster_span(c)
            if span is None:
                continue
            if isinstance(c, dict):
                # absorb clusters at/under the tolerance (join across them)
                sev = float(c.get('severity', float('inf')))
                if sev <= lens_severity_tol:
                    continue
                mark(soft, *span)        # above-tol soft break: bridgeable
                record(fldname, span[0], span[1], True, dial='lens severity tol',
                       note=f"sev={sev:.2f}")
            else:
                mark(hard, *span)        # old tuple cluster: no severity → hard
                record(fldname, span[0], span[1], False, note='legacy (no severity)')

    # ── Spike + excursion clusters — off-trajectory perceptibility dial ──
    # These soft events fragment a clean segment only if their off-trajectory
    # contextual residual (cm) meets the perceptibility threshold — a real,
    # perceptible glitch.  Below it = imperceptible → absorbed.  A perceptible
    # one is a HARD boundary (never bridged).  When off_traj_cm is unavailable
    # (-1, model absent or old report), fall back to the prior rule: excursion
    # always breaks; spike breaks if contaminated (or dense-uncontaminated).
    for fldname in ('excursion_clusters', 'spike_clusters'):
        is_spike = fldname == 'spike_clusters'
        for c in report.get(fldname, []) or []:
            span = _cluster_span(c)
            if span is None:
                continue
            # Movement-rescue valve: if the combiner is confident this is real
            # movement, absorb it (skip fragmentation) regardless of off_traj or
            # contamination.  Conservative threshold → rescues clear movements
            # (throws/landings) without leaking glitches.
            mprob = c.get('movement_prob', -1) if isinstance(c, dict) else -1
            if mprob is not None and mprob >= 0 and mprob >= movement_rescue_prob:
                continue
            contaminated = (is_spike and isinstance(c, dict)
                            and bool(c.get('contaminated')))
            otc = c.get('off_traj_cm', -1) if isinstance(c, dict) else -1
            if otc is not None and otc >= 0:
                # CONTAMINATED OVERRIDE: when keep_contaminated_spikes is on, a
                # dense-jitter (contaminated) region fragments even if its
                # off-trajectory residual is sub-threshold — the dense jitter
                # itself is treated as motion-invalid regardless of perceptible
                # excursion.  Off = trust off_traj alone (recover such regions
                # if imperceptible, e.g. the contaminated-but-0.13cm cluster).
                if otc >= off_traj_cm_tol or (contaminated and keep_contaminated_spikes):
                    mark(hard, *span)          # perceptible OR overridden → fragment
                    note = f"off={otc:.2f}cm"
                    if contaminated:
                        note += " contaminated"
                    if mprob is not None and mprob >= 0:
                        note += f" mv={mprob:.2f}"
                    dial = ('keep contaminated spikes'
                            if (otc < off_traj_cm_tol and contaminated)
                            else 'off-traj cm tol / movement rescue prob')
                    record(fldname, span[0], span[1], True, dial=dial, note=note)
                # else: imperceptible → absorbed
            else:                              # fallback (no perceptibility)
                if not is_spike:
                    mark(hard, *span)          # excursion: always break
                    record(fldname, span[0], span[1], True,
                           dial='movement rescue prob', note='no perceptibility')
                elif contaminated and keep_contaminated_spikes:
                    mark(hard, *span)
                    record(fldname, span[0], span[1], True,
                           dial='keep contaminated spikes', note='contaminated')
                elif isinstance(c, dict) and float(c.get('spike_density', 0.0)) > spike_density_tol:
                    mark(soft, *span)
                    record(fldname, span[0], span[1], True, dial='spike density tol',
                           note=f"density={float(c.get('spike_density', 0.0)):.2f}")

    # ── Structural anomalies — hard, non-tunable ────────────────────────
    if include_structural:
        for z in report.get('corruption_zones', []) or []:
            sp = _cluster_span(z)
            if sp:
                mark(hard, *sp)
                record('corruption_zones', sp[0], sp[1], False)
        for c in report.get('glitch_clusters', []) or []:
            sp = _cluster_span(c)
            if sp:
                mark(hard, *sp)
                record('glitch_clusters', sp[0], sp[1], False)
        for b in report.get('stream_breaks', []) or []:
            f = b.get('frame') if isinstance(b, dict) else b
            if f is not None:
                mark(hard, f, f)
                record('stream_breaks', f, f, False)

    # ── Expand bad regions by margin (both masks) ───────────────────────
    def expand(buf):
        if margin <= 0:
            return
        src = bytes(buf)
        for i in range(T):
            if src[i]:
                for d in range(1, margin + 1):
                    if i - d >= 0:
                        buf[i - d] = 1
                    if i + d < T:
                        buf[i + d] = 1
    expand(hard); expand(soft)
    bad = bytearray(h or s for h, s in zip(hard, soft))

    # ── Collect clean runs ──────────────────────────────────────────────
    runs = []  # (start, end_inclusive)
    in_clean = False
    start = 0
    for f in range(T):
        if not bad[f]:
            if not in_clean:
                start = f
                in_clean = True
        else:
            if in_clean:
                runs.append((start, f - 1))
                in_clean = False
    if in_clean:
        runs.append((start, T - 1))

    # ── Join runs separated by a short SOFT-ONLY bad gap ────────────────
    # A gap is bridged only if it is short AND contains no HARD frame — so a
    # brief genuine glitch (teleport/corruption/contaminated spike) is never
    # swallowed into a clean segment, regardless of how short it is.
    join_gap_frames = int(round(join_gap_s * fps))
    if join_gap_frames > 0 and runs:
        merged = [runs[0]]
        for s, e in runs[1:]:
            ps, pe = merged[-1]
            gap = range(pe + 1, s)
            gap_has_hard = any(hard[i] for i in gap)
            if (s - pe - 1) <= join_gap_frames and not gap_has_hard:
                merged[-1] = (ps, e)
            else:
                merged.append((s, e))
        runs = merged

    # ── Keep long-enough runs ───────────────────────────────────────────
    min_frames = min_clean_s * fps
    segs = []
    for s, e in runs:
        n = e - s + 1
        if n >= min_frames:
            segs.append({'start': s, 'end': e, 'n_frames': n,
                         'duration_s': round(n / fps, 2)})
    segs.sort(key=lambda x: -x['n_frames'])
    return segs


def clean_fraction(report: Dict[str, Any], segs=None, **kw) -> float:
    """Fraction of the file covered by clean segments at the given tolerance."""
    T = int(report.get('n_frames', 0))
    if T <= 0:
        return 0.0
    if segs is None:
        segs = rederive_clean_segments(report, **kw)
    return round(sum(s['n_frames'] for s in segs) / T, 4)
