import glob
import json
import os
import sys

import numpy as np

from dpg_system.node import Node

# Representativeness flag thresholds: a file's clean section is "unrepresentative"
# when it holds far less of the file's motion than of its duration (e.g. a static
# T-pose survives while the dynamic motion is cut).  rep = (mean activity in clean)
# / (mean activity over file); flag when rep is below UNREP_REP_MAX, but only if
# the clean section is a non-trivial slice (>= UNREP_MIN_DUR) — else the file is
# simply mostly-rejected, a different condition.
UNREP_REP_MAX = 0.4
UNREP_MIN_DUR = 0.15

# clean_segment_tools lives in the (non-package) noise_estimation dir alongside
# estimate_noise_torque.py; add it to the path so the live clean-segment
# re-derivation dials work regardless of cwd.
_NOISE_EST_DIR = os.path.join(os.path.dirname(__file__), 'noise_estimation')
if os.path.isdir(_NOISE_EST_DIR) and _NOISE_EST_DIR not in sys.path:
    sys.path.insert(0, _NOISE_EST_DIR)
try:
    import clean_segment_tools as cst
except Exception:
    cst = None


def register_noise_review_node():
    Node.app.register_node('noise_review', NoiseReviewNode.factory)


class NoiseReviewNode(Node):
    """Step through noise-report issues across a batch of NPZ files.

    Load a JSON produced by estimate_noise_torque.py, then use Prev/Next to
    walk through every stream break, corruption zone, spike frame, and glitchcan we
    cluster.  Outputs the NPZ path and start frame so an OpenTakeNode can load
    the file and jump to the right section automatically.
    """

    @staticmethod
    def factory(name, data, args=None):
        return NoiseReviewNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.raw_data = []    # list of report dicts from JSON
        self.files = []       # ordered per-file entries: {filepath, filename, assessment, issues}
        self.positions = []   # flat nav order: (file_idx, issue_idx) with issue_idx == -1 for issue-free files
        self.current_idx = -1
        self.current_filepath = ''
        self._path_cache = {}  # orig filepath -> resolved on-disk path (or orig)
        self._activity_cache = {}  # resolved path -> per-frame activity (proxy); for representativeness
        self.filter_terms = []  # lowercased tokens; all must be substrings of a file's path

        # ── JSON loading ──────────────────────────────────────────────────
        self.json_path_input = self.add_input(
            'json path', widget_type='text_input', default_value='',
            callback=self.load_json)
        self.load_btn = self.add_input(
            'load', widget_type='button', callback=self.load_json)
        # Fallback AMASS root: used to re-root stored paths that no longer
        # exist (tree relocated, or JSON authored on another machine).  Empty
        # = trust the stored absolute paths as-is.
        self.amass_root_input = self.add_input(
            'amass root', widget_type='text_input', default_value='',
            callback=self._on_root_change)

        # ── Navigation ────────────────────────────────────────────────────
        self.prev_btn = self.add_input(
            'prev', widget_type='button', callback=self.go_prev)
        self.next_btn = self.add_input(
            'next', widget_type='button', callback=self.go_next)
        self.prev_file_btn = self.add_input(
            'prev file', widget_type='button', callback=self.go_prev_file)
        self.next_file_btn = self.add_input(
            'next file', widget_type='button', callback=self.go_next_file)
        self.section_input = self.add_input(
            'section', widget_type='input_int', default_value=0,
            callback=self.jump_to_section)
        self.filter_input = self.add_input(
            'filter', widget_type='text_input', default_value='',
            callback=self.apply_filter)

        # ── Status labels ─────────────────────────────────────────────────
        self.counter_label    = self.add_label('— / —')
        self.file_label       = self.add_label('')
        self.assessment_label = self.add_label('')
        self.issue_label      = self.add_label('')

        # ── Outputs ───────────────────────────────────────────────────────
        self.path_out      = self.add_output('npz path')
        self.frame_out     = self.add_output('frame')
        self.end_frame_out = self.add_output('end frame')
        self.bang_out      = self.add_output('bang')

        # ── Filter options ────────────────────────────────────────────────
        self.show_breaks     = self.add_option('stream breaks',    widget_type='checkbox', default_value=True,  callback=self.rebuild_issues)
        self.show_corruption = self.add_option('corruption zones', widget_type='checkbox', default_value=True,  callback=self.rebuild_issues)
        self.show_spikes     = self.add_option('spike frames',     widget_type='checkbox', default_value=True,  callback=self.rebuild_issues)
        self.show_glitches   = self.add_option('glitch clusters',  widget_type='checkbox', default_value=True,  callback=self.rebuild_issues)
        self.show_lenses     = self.add_option('lens flags',       widget_type='checkbox', default_value=True,  callback=self.rebuild_issues)
        self.show_clean      = self.add_option('clean sections',   widget_type='checkbox', default_value=False, callback=self.rebuild_issues)
        # Per-cut readout: one stop per artifact that breaks a clean segment at the
        # current dial settings, tagged HARD (always cuts) or tunable (and which
        # dial absorbs it) — so during review you can see at a glance whether a
        # given break is dial-tunable or a hard lens.
        self.show_cuts       = self.add_option('show cuts',         widget_type='checkbox', default_value=False, callback=self.rebuild_issues)
        # Per-folder clean YIELD: prints clean-segment duration / total duration
        # per dataset to the console on load and on every dial change, so you can
        # watch how much of each folder survives as you re-tune.  (Re-derives
        # every loaded file per rebuild — same cost as 'clean sections'; turn off
        # if dragging a dial feels laggy on a very large folder.)
        self.show_yield      = self.add_option('folder yield',      widget_type='checkbox', default_value=False, callback=self.rebuild_issues)
        # Flag files whose surviving clean section is NOT representative of the
        # file's motion — e.g. a static T-pose survives while the dynamic motion
        # is all cut.  Metric: (mean activity in clean) / (mean activity over file);
        # well below 1 means the clean part is the quiet part.  Loads each file's
        # npz once to get its activity (fast proxy, ~3 ms/file, cached), so the
        # first rebuild after enabling pays a one-time per-folder cost.
        self.flag_unrep      = self.add_option('flag unrepresentative', widget_type='checkbox', default_value=False, callback=self.rebuild_issues)
        self.class_filter    = self.add_option('classification', widget_type='combo', default_value='all',  callback=self.rescope)
        # Exact-class selection (so you can isolate ONLY clean, or only
        # moderate, etc.), plus 'flagged' = moderate+problematic (the old
        # ">= moderate" view of everything the detector flagged).
        self.class_filter.widget.combo_items = ['all', 'clean', 'moderate', 'problematic', 'flagged']
        # File ordering for Prev/Next File.  'file order' = as loaded (the batch
        # writes noise-score-descending); 'dev×speed' ranks worst-corruption
        # first (continuous gross-corruption severity); others as labelled.
        self.sort_by         = self.add_option('sort by', widget_type='combo', default_value='file order', callback=self.rebuild_issues)
        self.sort_by.widget.combo_items = ['file order', 'dev×speed', 'noise score', 'classification']

        # ── Clean-segment re-derivation dials ─────────────────────────────
        # Tune offline (no re-run): absorb sub-tolerance artifacts and join the
        # clean segments across them.  Higher = more permissive.  See
        # clean_segment_tools.rederive_clean_segments.
        # PRIMARY dial: spike/excursion fragment a clean segment only if their
        # off-trajectory contextual residual (cm) reaches this. Default 1.0 cm
        # (real glitches 2-16, imperceptible/clean 0-0.7; 0.7-1.5 fuzzy margin).
        self.clean_offtraj   = self.add_option('off-traj cm tol', widget_type='drag_float', default_value=1.0, callback=self.rebuild_issues)
        self.clean_lens_tol  = self.add_option('lens severity tol', widget_type='drag_float', default_value=1.25, callback=self.rebuild_issues)
        self.clean_spike_tol = self.add_option('spike density tol', widget_type='drag_float', default_value=1.0, callback=self.rebuild_issues)
        # OVERRIDE on the off-traj dial: On (default) = a contaminated (dense-
        # jitter) spike cluster fragments even if its off-traj residual is below
        # the dial (dense jitter is motion-invalid regardless).  Off = trust
        # off-traj alone, so a contaminated-but-imperceptible region is absorbed.
        self.keep_contam     = self.add_option('keep contaminated spikes', widget_type='checkbox', default_value=True, callback=self.rebuild_issues)
        # Movement-rescue valve: absorb a perceptible spike/excursion as real
        # movement when the combiner's movement_prob >= this (throw/landing FP
        # fix).  0.89 = conservative; lower = rescue more; >1.0 = disable.
        self.clean_movement  = self.add_option('movement rescue prob', widget_type='drag_float', default_value=0.89, callback=self.rebuild_issues)
        self.clean_join_gap  = self.add_option('clean join gap (s)', widget_type='drag_float', default_value=0.0, callback=self.rebuild_issues)
        self.clean_min_dur   = self.add_option('clean min dur (s)',  widget_type='drag_float', default_value=1.0, callback=self.rebuild_issues)

    # ── JSON loading ──────────────────────────────────────────────────────

    def load_json(self):
        path = self.json_path_input()
        if not path:
            return
        if not os.path.exists(path):
            print(f'NoiseReviewNode: file not found: {path}')
            return
        try:
            with open(path, 'r') as f:
                raw = json.load(f)
            self.raw_data = self._normalize_reports(raw)
            print(f'NoiseReviewNode: loaded {len(self.raw_data)} reports from {os.path.basename(path)}')
            self.rebuild_issues()
        except Exception as e:
            print(f'NoiseReviewNode: failed to load {path}: {e}')

    def _on_root_change(self):
        """AMASS-root edited: drop cached resolutions and rebuild so paths
        re-resolve against the new root."""
        self._path_cache = {}
        self._activity_cache = {}
        if self.raw_data:
            self.rebuild_issues()

    # ── Path resolution ────────────────────────────────────────────────────

    def _resolve_filepath(self, filepath, report):
        """Return an on-disk path for a report.  Stored paths are absolute; if
        one no longer exists and an `amass root` is set, re-root it (tree moved
        or JSON authored elsewhere).  Results are cached per original path."""
        if not filepath:
            return filepath
        if filepath in self._path_cache:
            return self._path_cache[filepath]

        resolved = filepath
        if not os.path.exists(filepath):
            root = self.amass_root_input().strip()
            if root:
                cand = self._reroot(filepath, root, report)
                if cand:
                    resolved = cand
        self._path_cache[filepath] = resolved
        return resolved

    def _reroot(self, orig, root, report):
        """Find `orig` under a new AMASS `root`.  Tries dataset-relative tail,
        then the segment after an 'AMASS' anchor, then a basename search
        disambiguated by frame count."""
        parts = orig.replace('\\', '/').split('/')
        dataset = report.get('dataset')

        # 1) <root>/<dataset>/<tail after the dataset folder>
        if dataset and dataset in parts:
            i = len(parts) - 1 - parts[::-1].index(dataset)  # last occurrence
            cand = os.path.join(root, dataset, *parts[i + 1:])
            if os.path.exists(cand):
                return cand

        # 2) <root>/<tail after the last 'AMASS' segment>
        low = [p.lower() for p in parts]
        if 'amass' in low:
            i = len(low) - 1 - low[::-1].index('amass')
            cand = os.path.join(root, *parts[i + 1:])
            if os.path.exists(cand):
                return cand

        # 3) basename search, disambiguated by recorded frame count
        base = parts[-1]
        search_root = root
        if dataset and os.path.isdir(os.path.join(root, dataset)):
            search_root = os.path.join(root, dataset)
        matches = glob.glob(os.path.join(search_root, '**', base), recursive=True)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            n_frames = report.get('n_frames')
            if n_frames:
                for m in matches:
                    if self._npz_n_frames(m) == n_frames:
                        return m
        return None

    @staticmethod
    def _npz_n_frames(path):
        """Frame count of an AMASS .npz (poses length), or None if unreadable."""
        try:
            import numpy as np
            with np.load(path, allow_pickle=True) as z:
                for k in ('poses', 'pose', 'trans'):
                    if k in z:
                        return z[k].shape[0]
        except Exception:
            return None
        return None

    @staticmethod
    def _normalize_reports(raw):
        """Accept either a flat list of reports or the new per-folder dict
        `{"dataset": ..., "files": {filepath: report, ...}}`, and return a
        list of report dicts each carrying a `filepath` field."""
        if isinstance(raw, list):
            return [r for r in raw if isinstance(r, dict)]
        if isinstance(raw, dict):
            dataset = raw.get('dataset')
            files = raw.get('files')
            if isinstance(files, dict):
                reports = []
                for filepath, report in files.items():
                    if not isinstance(report, dict):
                        continue
                    report = dict(report)
                    report.setdefault('filepath', filepath)
                    if dataset is not None:
                        report.setdefault('dataset', dataset)
                    reports.append(report)
                return reports
            if isinstance(files, list):
                return [r for r in files if isinstance(r, dict)]
        return []

    # ── Issue list construction ───────────────────────────────────────────

    @staticmethod
    def _folder_of(filepath):
        """Dataset/folder name for the per-folder yield tally — the path segment
        right after 'AMASS', else the immediate parent directory."""
        fp = (filepath or '').replace('\\', '/')
        if '/AMASS/' in fp:
            return fp.split('/AMASS/')[1].split('/')[0]
        parts = [p for p in fp.split('/') if p]
        return parts[-2] if len(parts) >= 2 else (parts[-1] if parts else '?')

    def _activity_for(self, filepath):
        """Per-frame whole-body activity for `filepath` (an already-resolved path),
        cached.  Fast axis-angle frame-diff proxy (median over body joints) — not
        geodesic, but for *relative* motion it tracks the exact metric perfectly
        (r=1.0) at ~15x the speed.  Returns a 1-D array (len n_frames-1), or None
        if the npz can't be loaded."""
        if filepath in self._activity_cache:
            return self._activity_cache[filepath]
        act = None
        try:
            Q = np.load(filepath, allow_pickle=True)['poses']
            P = Q.reshape(Q.shape[0], -1, 3)[:, 1:22, :]        # body joints (skip root)
            act = np.median(np.linalg.norm(np.diff(P, axis=0), axis=2), axis=1)
        except Exception:
            act = None
        self._activity_cache[filepath] = act
        return act

    def _representativeness(self, segs, activity):
        """(rep, dur_yield, mot_yield, worst_cut_frame) for the live clean segments
        against the file's per-frame `activity`.  rep = clean-section motion share
        / duration share; rep << 1 means the clean section under-represents the
        file's motion.  worst_cut_frame = the most-active *discarded* frame (a jump
        target to the lost motion).  None if activity is unavailable."""
        if activity is None or len(activity) == 0:
            return None
        n = len(activity)
        clean = np.zeros(n, dtype=bool)
        for s in segs:
            clean[max(0, int(s.get('start', 0))):min(n, int(s.get('end', 0)) + 1)] = True
        dur_yield = float(clean.mean())
        if dur_yield <= 0:
            return 0.0, 0.0, 0.0, 0
        mot_yield = float(activity[clean].sum()) / (float(activity.sum()) + 1e-9)
        rep = mot_yield / dur_yield
        worst = int(np.argmax(np.where(clean, -1.0, activity))) if (~clean).any() else 0
        return rep, dur_yield, mot_yield, worst

    def _clean_segments_for(self, report, cuts_out=None):
        """Clean segments re-derived live at the current dial settings.  Falls
        back to the report's stored clean_segments if the tool is missing.  If a
        `cuts_out` list is passed, it is populated with the per-cut readout."""
        if cst is None:
            return report.get('clean_segments', []) or []
        try:
            return cst.rederive_clean_segments(
                report,
                off_traj_cm_tol=float(self.clean_offtraj()),
                lens_severity_tol=float(self.clean_lens_tol()),
                spike_density_tol=float(self.clean_spike_tol()),
                keep_contaminated_spikes=bool(self.keep_contam()),
                movement_rescue_prob=float(self.clean_movement()),
                join_gap_s=float(self.clean_join_gap()),
                min_clean_s=float(self.clean_min_dur()),
                cuts_out=cuts_out,
            )
        except Exception as ex:
            print(f'NoiseReviewNode: clean re-derivation failed: {ex}')
            return report.get('clean_segments', []) or []

    @staticmethod
    def _format_assessment(classification, noise_score):
        if noise_score is None:
            return classification
        return f'{classification}  (score {noise_score:.3f})'

    def rescope(self):
        """Callback for scope changes (min classification): the set of visited
        files is changing, so don't try to stay parked on the current file —
        clear it so the rebuild lands on the first file of the new scope, making
        the change immediately visible even when the current file still
        qualifies (e.g. you're sitting on the worst file)."""
        self.current_filepath = ''
        self.rebuild_issues()

    def rebuild_issues(self):
        show_breaks     = self.show_breaks()
        show_corruption = self.show_corruption()
        show_spikes     = self.show_spikes()
        show_glitches   = self.show_glitches()
        show_lenses     = self.show_lenses()
        show_clean      = self.show_clean()

        cls_rank = {'clean': 0, 'moderate': 1, 'problematic': 2}
        # Which classifications to include.  'all' = no filter at all (any label,
        # including custom categories like a labeling set's DYN-CANDIDATE / GLITCH
        # tags); otherwise exact match per class, plus 'flagged' (= moderate +
        # problematic).
        allowed_classes = {
            'clean':       {'clean'},
            'moderate':    {'moderate'},
            'problematic': {'problematic'},
            'flagged':     {'moderate', 'problematic'},
        }.get(self.class_filter())          # None for 'all' → no filtering

        files = []
        total_issues = 0
        want_yield = bool(self.show_yield())
        want_unrep = bool(self.flag_unrep())
        yield_acc = {}      # folder -> [clean_s, total_s, n_files, n_segs, n_files_with_clean]
        unrep_acc = {}      # folder -> [n_unrepresentative, n_checked]
        for report in self.raw_data:
            if (allowed_classes is not None
                    and report.get('classification', 'clean') not in allowed_classes):
                continue

            filepath = report.get('filepath') or report.get('filename', '')
            filepath = self._resolve_filepath(filepath, report)
            # Path filter: every term in the `filter` widget must appear in the
            # file's path (AND substring match).  Narrows the whole review scope
            # — prev/next/file-nav only visit files that pass.
            if self.filter_terms:
                low = filepath.lower()
                if not all(term in low for term in self.filter_terms):
                    continue
            filename = os.path.basename(filepath)
            classification = report.get('classification', 'clean')
            noise_score = report.get('noise_score')
            assessment = self._format_assessment(classification, noise_score)
            detail = report.get('classification_detail', '')

            file_issues = []

            def make_issue(**kw):
                kw.update(filepath=filepath, filename=filename, assessment=assessment)
                return kw

            if show_breaks:
                for sb in report.get('stream_breaks', []):
                    f = sb.get('frame', 0)
                    file_issues.append(make_issue(
                        type='stream_break', frame=f, end_frame=f,
                        desc=(f"stream break  frame={f}"
                              f"  type={sb.get('break_type','')}"
                              f"  worst={sb.get('worst_joint','')}")))

            if show_corruption:
                surgery  = report.get('surgery') or {}
                excision = surgery.get('excision') or {}
                for z in excision.get('zones', []):
                    s, e = z.get('start', 0), z.get('end', 0)
                    joints = ', '.join((z.get('joints') or [])[:3])
                    file_issues.append(make_issue(
                        type='corruption_zone', frame=s, end_frame=e,
                        desc=(f"corruption  [{s}–{e}]"
                              f"  {z.get('duration_s', 0):.1f}s"
                              f"  mean={z.get('mean_vel', 0):.0f} max={z.get('max_vel', 0):.0f} rad/s"
                              f"  {joints}")))

            if show_spikes:
                # Attributed spike CLUSTERS (grouped events) rather than noisy
                # per-frame flags; navigate to the corrected peak frame.
                for c in report.get('spike_clusters', []):
                    if not isinstance(c, dict):
                        continue
                    s, e = int(c.get('start', 0)), int(c.get('end', 0))
                    region = c.get('region') or ''
                    pframe = c.get('peak_frame', -1)
                    nav = pframe if isinstance(pframe, int) and pframe >= 0 else s
                    contam = ' CONTAM' if c.get('contaminated') else ''
                    joints = ', '.join(c.get('joints', [])[:3])
                    otc = c.get('off_traj_cm', -1)
                    off_txt = f"  off={otc:.2f}cm" if isinstance(otc, (int, float)) and otc >= 0 else ''
                    mvp = c.get('movement_prob', -1)
                    mv_txt = f" mv={mvp:.2f}" if isinstance(mvp, (int, float)) and mvp >= 0 else ''
                    attr = f"  → {region}/{c.get('peak_joint','')} @f{pframe}{off_txt}{mv_txt}" if region else ''
                    file_issues.append(make_issue(
                        type='spike_cluster', frame=nav, end_frame=e,
                        desc=(f"spike{contam}  [{s}–{e}]  n={c.get('n_spike_frames', 0)}"
                              f" dens={c.get('spike_density', 0):.2f}  {joints}{attr}")))

            if show_glitches:
                for cluster in report.get('glitch_clusters', []):
                    if not (isinstance(cluster, (list, tuple)) and len(cluster) >= 2):
                        continue
                    s, e = int(cluster[0]), int(cluster[1])
                    file_issues.append(make_issue(
                        type='glitch_cluster', frame=s, end_frame=e,
                        desc=f"glitch  [{s}–{e}]  {e - s + 1} frames"))

            if show_lenses:
                # Kinematic pose-level lenses.  New runs export LensCluster
                # dicts (carrying `severity`); older runs export bare [start,
                # end] pairs.  Handle both.
                for lname, key in (('teleport',  'teleport_clusters'),
                                   ('reversal',  'reversal_clusters'),
                                   ('1-frame',   'sfglitch_clusters'),
                                   ('instability', 'instability_clusters'),
                                   ('excursion', 'excursion_clusters'),
                                   ('flicker',   'flicker_clusters'),
                                   ('zigzag',    'zigzag_clusters'),
                                   ('candy-wrapper', 'rom_clusters')):
                    for cluster in report.get(key, []):
                        nav = None; attr_txt = ''
                        if isinstance(cluster, dict):
                            s, e = int(cluster.get('start', 0)), int(cluster.get('end', 0))
                            sev = cluster.get('severity')
                            sev_txt = f"  sev {sev:.2f}" if isinstance(sev, (int, float)) else ''
                            # attribution: navigate to the corrected peak frame,
                            # show the limb region (+ source joint).
                            region = cluster.get('region') or ''
                            pframe = cluster.get('peak_frame', -1)
                            otc = cluster.get('off_traj_cm', -1)
                            off_txt = f" off={otc:.2f}cm" if isinstance(otc, (int, float)) and otc >= 0 else ''
                            mvp = cluster.get('movement_prob', -1)
                            mv_txt = f" mv={mvp:.2f}" if isinstance(mvp, (int, float)) and mvp >= 0 else ''
                            if region:
                                attr_txt = f"  → {region}/{cluster.get('peak_joint','')} @f{pframe}{off_txt}{mv_txt}"
                            if isinstance(pframe, int) and pframe >= 0:
                                nav = pframe
                        elif isinstance(cluster, (list, tuple)) and len(cluster) >= 2:
                            s, e = int(cluster[0]), int(cluster[1])
                            sev_txt = ''
                        else:
                            continue
                        file_issues.append(make_issue(
                            type='lens_flag', frame=(nav if nav is not None else s),
                            end_frame=e,
                            desc=f"{lname}  [{s}–{e}]  {e - s + 1} frames{sev_txt}{attr_txt}"))

            want_cuts = bool(self.show_cuts())
            if show_clean or want_cuts or want_yield or want_unrep:
                # Re-derive clean segments live from the report's serialized
                # artifact severities at the current dial settings, instead of
                # trusting the run-time clean_segments.  Falls back to the
                # stored segments if the tool is unavailable.  cuts_out captures,
                # in the same pass, every artifact that breaks a segment now.
                cuts = [] if want_cuts else None
                segs = self._clean_segments_for(report, cuts_out=cuts)
                if want_yield:
                    T = float(report.get('n_frames') or 0)
                    fps = float(report.get('fps') or 120.0)
                    if T > 0:
                        fld = report.get('dataset') or self._folder_of(filepath)
                        acc = yield_acc.setdefault(fld, [0.0, 0.0, 0, 0, 0])
                        acc[0] += sum(s.get('duration_s', 0.0) for s in segs)
                        acc[1] += T / fps
                        acc[2] += 1                        # files
                        acc[3] += len(segs)                # clean sections
                        acc[4] += 1 if segs else 0         # files with >=1 clean section
                if show_clean:
                    for seg in segs:
                        s, e = seg.get('start', 0), seg.get('end', 0)
                        extra = (f"  mean={seg.get('mean_score', 0):.2f} max={seg.get('max_score', 0):.2f}"
                                 if 'mean_score' in seg else '')
                        file_issues.append(make_issue(
                            type='clean_section', frame=s, end_frame=e,
                            desc=(f"clean  [{s}–{e}]"
                                  f"  {seg.get('duration_s', 0):.1f}s"
                                  f"  {seg.get('n_frames', e - s + 1)} frames{extra}")))
                if cuts:
                    # one stop per active cut, tagged HARD / tunable(<dial>), in
                    # frame order — the "why was this cut, can I tune it?" readout
                    for cut in sorted(cuts, key=lambda x: (x['start'], x['end'])):
                        s, e = cut['start'], cut['end']
                        tag = ('tunable: ' + cut['dial']) if cut['tunable'] else 'HARD lens'
                        note = f"  {cut['note']}" if cut.get('note') else ''
                        file_issues.append(make_issue(
                            type='cut', frame=s, end_frame=e,
                            desc=f"cut · {cut['lens']}  [{s}–{e}]  ({tag}){note}"))
                if want_unrep:
                    # Is the surviving clean section representative of the file's
                    # motion?  Loads the file's activity (cached) and compares the
                    # clean section's motion share to its duration share.
                    rinfo = self._representativeness(segs, self._activity_for(filepath))
                    if rinfo is not None:
                        rep, dy, my, worst = rinfo
                        ua = unrep_acc.setdefault(report.get('dataset') or self._folder_of(filepath), [0, 0])
                        ua[1] += 1
                        if rep < UNREP_REP_MAX and dy >= UNREP_MIN_DUR:
                            ua[0] += 1
                            file_issues.append(make_issue(
                                type='unrepresentative', frame=worst, end_frame=worst,
                                desc=(f"⚠ clean section unrepresentative — holds {100 * my:.0f}% of motion "
                                      f"in {100 * dy:.0f}% of duration (rep {rep:.2f}); "
                                      f"jump → most-active discarded frame {worst}")))

            files.append({
                'filepath': filepath,
                'filename': filename,
                'assessment': assessment,
                'detail': detail,
                'issues': file_issues,
                'noise_score': report.get('noise_score') or 0.0,
                'dev_speed_max': report.get('dev_speed_max') or 0.0,
                'cls_rank': {'clean': 0, 'moderate': 1, 'problematic': 2}.get(
                    report.get('classification', 'clean'), 0),
            })
            total_issues += len(file_issues)

        # Optional re-ordering of the file list (worst-first) by a chosen key,
        # so Prev/Next File walks files in severity order.  'file order' keeps
        # the JSON's order (already noise-score-descending from the batch).
        sort_key = self.sort_by()
        keymap = {
            'dev×speed':      lambda f: f['dev_speed_max'],
            'noise score':    lambda f: f['noise_score'],
            'classification': lambda f: (f['cls_rank'], f['dev_speed_max']),
        }
        if sort_key in keymap:
            files.sort(key=keymap[sort_key], reverse=True)

        # Flat navigation order: one stop per issue, plus a standalone stop for
        # any file with no flagged issues of the selected kinds, so Prev/Next
        # File still visits it (showing 0 / 0 issues alongside its rating).
        positions = []
        for fi, fentry in enumerate(files):
            if fentry['issues']:
                positions.extend((fi, ii) for ii in range(len(fentry['issues'])))
            else:
                positions.append((fi, -1))

        prev_filepath = self.current_filepath
        self.files = files
        self.positions = positions
        filt = f" (filter: {' '.join(self.filter_terms)})" if self.filter_terms else ''
        print(f'NoiseReviewNode: {len(files)} files, {total_issues} issues '
              f'built from {len(self.raw_data)} reports{filt}')
        if yield_acc:
            print('NoiseReviewNode: clean yield @ current dials '
                  '(seg/file & avg over surviving files) —')

            def _line(label, a):
                c, t, nf, ns, nfc = a
                if t <= 0:
                    return
                spf = ns / nfc if nfc else 0.0          # clean sections per surviving file
                avg = c / ns if ns else 0.0             # mean clean-section length (s)
                lost = 100 * (nf - nfc) / nf if nf else 0.0
                print(f'    {label:<20} {100 * c / t:5.1f}%   '
                      f'{spf:4.1f} seg/file   {avg:5.1f}s avg seg   '
                      f'{lost:3.0f}% files lost   ({nf} files)')

            tot = [0.0, 0.0, 0, 0, 0]
            for fld, a in sorted(yield_acc.items(),
                                 key=lambda kv: -(kv[1][0] / kv[1][1] if kv[1][1] else 0)):
                _line(fld, a)
                for i in range(5):
                    tot[i] += a[i]
            if len(yield_acc) > 1:
                _line('ALL', tot)
        if unrep_acc:
            tot_u = sum(a[0] for a in unrep_acc.values())
            tot_n = sum(a[1] for a in unrep_acc.values())
            print(f'NoiseReviewNode: {tot_u}/{tot_n} files flagged unrepresentative '
                  f'(clean section misses the motion, rep < {UNREP_REP_MAX}) —')
            for fld, (nu, nn) in sorted(unrep_acc.items(), key=lambda kv: -kv[1][0]):
                if nu:
                    print(f'    {fld:<20} {nu:4d} / {nn}')
        if self.filter_terms and not files:
            self.counter_label.set('0 / 0  (no files match filter)')
            self.file_label.set('')
            self.assessment_label.set('')
            self.issue_label.set('')
            self.current_idx = -1
            return

        # Stay on the file that was being viewed, landing on its first stop in
        # the new order, so filter toggles don't bounce back to the first file.
        if prev_filepath:
            for i, (fi, _ii) in enumerate(positions):
                if files[fi]['filepath'] == prev_filepath:
                    self.current_idx = i
                    self._emit_current()
                    return

        # The previously-viewed file is gone (e.g. min-classification was raised
        # and it no longer qualifies, or a filter excluded it) — land on AND
        # DISPLAY the first file of the new list, so the change is visible
        # instead of leaving the view frozen on a stale/blank state.
        if positions:
            self.current_idx = 0
            self._emit_current()
        else:
            self.current_idx = -1
            self.counter_label.set('0 / 0')
            self.file_label.set('')
            self.assessment_label.set('')
            self.issue_label.set('')
            self.section_input.set(0, propagate=False)

    # ── Navigation ────────────────────────────────────────────────────────

    def go_next(self):
        if not self.positions:
            print('NoiseReviewNode: no files loaded')
            return
        self.current_idx = (self.current_idx + 1) % len(self.positions)
        self._emit_current()

    def go_prev(self):
        if not self.positions:
            return
        self.current_idx = (self.current_idx - 1) % len(self.positions)
        self._emit_current()

    def go_next_file(self):
        if not self.positions:
            return
        cur_file = self.positions[self.current_idx][0]
        n = len(self.positions)
        # walk forward to the first stop whose file differs, wrapping around;
        # because each file's stops are contiguous, this lands on the file's
        # first stop (its first issue, or its lone issue-free stop)
        for step in range(1, n + 1):
            idx = (self.current_idx + step) % n
            if self.positions[idx][0] != cur_file:
                self.current_idx = idx
                self._emit_current()
                return

    def go_prev_file(self):
        if not self.positions:
            return
        cur_file = self.positions[self.current_idx][0]
        n = len(self.positions)
        # find a stop belonging to the previous file, wrapping around
        prev_file = None
        for step in range(1, n + 1):
            idx = (self.current_idx - step) % n
            if self.positions[idx][0] != cur_file:
                prev_file = self.positions[idx][0]
                break
        if prev_file is None:
            return
        # rewind to that file's first stop (stops are contiguous per file)
        self.current_idx = next(i for i, p in enumerate(self.positions) if p[0] == prev_file)
        self._emit_current()

    def jump_to_section(self):
        if not self.positions:
            return
        n = int(self.section_input())
        total = len(self.positions)
        n = max(1, min(total, n))
        self.current_idx = n - 1
        self._emit_current()

    def apply_filter(self):
        """Restrict the review to files whose path contains every whitespace-
        separated term in the `filter` widget (AND substring match).  Rebuilds
        the nav order so prev/next/file-nav only traverse matching files.
        Clearing the widget restores the full set."""
        self.filter_terms = str(self.filter_input()).lower().split()
        if self.raw_data:
            self.rebuild_issues()

    def _emit_current(self):
        file_idx, issue_idx = self.positions[self.current_idx]
        fentry  = self.files[file_idx]
        n_files = len(self.files)
        ni      = len(fentry['issues'])

        if issue_idx >= 0:
            issue     = fentry['issues'][issue_idx]
            desc      = issue['desc']
            frame     = issue['frame']
            end_frame = issue['end_frame']
            itype     = issue['type']
            issue_count = f'{issue_idx + 1} / {ni}'
        else:
            issue     = None
            desc      = '(no flagged issues of the selected kinds)'
            frame     = 0
            end_frame = 0
            itype     = None
            issue_count = '0 / 0'

        # Labels
        self.counter_label.set(f'file {file_idx + 1}/{n_files}   issue {issue_count}')
        self.file_label.set(fentry['filename'])
        self.assessment_label.set(fentry['assessment'])
        self.issue_label.set(desc)
        self.section_input.set(self.current_idx + 1, propagate=False)

        # Console — flag file changes clearly
        if fentry['filepath'] != self.current_filepath:
            print(f'\n  ── {fentry["filename"]} ──  {fentry["assessment"]}'
                  f'  [dev×speed {fentry.get("dev_speed_max", 0):.1f}]')
            if fentry.get('detail'):
                print(f'     {fentry["detail"]}')
            self.current_filepath = fentry['filepath']
        if issue is not None:
            print(f'  [{issue_idx + 1}/{ni}] {desc}')
            if itype in ('corruption_zone', 'glitch_cluster', 'clean_section', 'lens_flag', 'spike_cluster', 'cut', 'unrepresentative') and end_frame != frame:
                print(f'         end frame: {end_frame}')
        else:
            print(f'  {desc}')

        # Send outputs — path first so OpenTakeNode loads before seeking;
        # bang last so downstream knows all values are settled.
        self.path_out.send(fentry['filepath'])
        self.end_frame_out.send(end_frame)
        self.frame_out.send(frame)
        self.bang_out.send('bang')
