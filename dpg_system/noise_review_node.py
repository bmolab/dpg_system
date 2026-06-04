import json
import os

from dpg_system.node import Node


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

        # ── JSON loading ──────────────────────────────────────────────────
        self.json_path_input = self.add_input(
            'json path', widget_type='text_input', default_value='',
            callback=self.load_json)
        self.load_btn = self.add_input(
            'load', widget_type='button', callback=self.load_json)

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
        self.show_clean      = self.add_option('clean sections',   widget_type='checkbox', default_value=False, callback=self.rebuild_issues)
        self.min_class       = self.add_option('min classification', widget_type='combo', default_value='all',  callback=self.rebuild_issues)
        self.min_class.widget.combo_items = ['all', 'moderate', 'problematic']

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

    @staticmethod
    def _normalize_reports(raw):
        """Accept either a flat list of reports or the new per-folder dict
        `{"dataset": ..., "files": {filepath: report, ...}}`, and return a
        list of report dicts each carrying a `filepath` field."""
        if isinstance(raw, list):
            return [r for r in raw if isinstance(r, dict)]
        if isinstance(raw, dict):
            files = raw.get('files')
            if isinstance(files, dict):
                reports = []
                for filepath, report in files.items():
                    if not isinstance(report, dict):
                        continue
                    report = dict(report)
                    report.setdefault('filepath', filepath)
                    reports.append(report)
                return reports
            if isinstance(files, list):
                return [r for r in files if isinstance(r, dict)]
        return []

    # ── Issue list construction ───────────────────────────────────────────

    @staticmethod
    def _format_assessment(classification, noise_score):
        if noise_score is None:
            return classification
        return f'{classification}  (score {noise_score:.3f})'

    def rebuild_issues(self):
        show_breaks     = self.show_breaks()
        show_corruption = self.show_corruption()
        show_spikes     = self.show_spikes()
        show_glitches   = self.show_glitches()
        show_clean      = self.show_clean()

        cls_rank = {'clean': 0, 'moderate': 1, 'problematic': 2}
        min_rank = {'all': 0, 'moderate': 1, 'problematic': 2}.get(self.min_class(), 0)

        files = []
        total_issues = 0
        for report in self.raw_data:
            if cls_rank.get(report.get('classification', 'clean'), 0) < min_rank:
                continue

            filepath = report.get('filepath') or report.get('filename', '')
            filename = os.path.basename(filepath)
            classification = report.get('classification', 'clean')
            noise_score = report.get('noise_score')
            assessment = self._format_assessment(classification, noise_score)

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
                by_frame = {}
                for sf in report.get('spike_frames', []):
                    f = sf.get('frame', 0)
                    by_frame.setdefault(f, []).append(
                        f"{sf.get('joint_name','')} {sf.get('velocity', 0):.0f}r/s ×{sf.get('neighbor_ratio', 0):.1f}")
                for f, entries in sorted(by_frame.items()):
                    file_issues.append(make_issue(
                        type='spike_frame', frame=f, end_frame=f,
                        desc=f"spike  frame={f}  {', '.join(entries[:4])}"))

            if show_glitches:
                for cluster in report.get('glitch_clusters', []):
                    if not (isinstance(cluster, (list, tuple)) and len(cluster) >= 2):
                        continue
                    s, e = int(cluster[0]), int(cluster[1])
                    file_issues.append(make_issue(
                        type='glitch_cluster', frame=s, end_frame=e,
                        desc=f"glitch  [{s}–{e}]  {e - s + 1} frames"))

            if show_clean:
                for seg in report.get('clean_segments', []):
                    s, e = seg.get('start', 0), seg.get('end', 0)
                    file_issues.append(make_issue(
                        type='clean_section', frame=s, end_frame=e,
                        desc=(f"clean  [{s}–{e}]"
                              f"  {seg.get('duration_s', 0):.1f}s"
                              f"  {seg.get('n_frames', e - s + 1)} frames"
                              f"  mean={seg.get('mean_score', 0):.2f} max={seg.get('max_score', 0):.2f}")))

            files.append({
                'filepath': filepath,
                'filename': filename,
                'assessment': assessment,
                'issues': file_issues,
            })
            total_issues += len(file_issues)

        # Flat navigation order: one stop per issue, plus a standalone stop for
        # any file with no flagged issues of the selected kinds, so Prev/Next
        # File still visits it (showing 0 / 0 issues alongside its rating).
        positions = []
        for fi, fentry in enumerate(files):
            if fentry['issues']:
                positions.extend((fi, ii) for ii in range(len(fentry['issues'])))
            else:
                positions.append((fi, -1))

        self.files = files
        self.positions = positions
        self.current_idx = 0 if positions else -1
        self.counter_label.set('— / —')
        self.file_label.set('')
        self.assessment_label.set('')
        self.issue_label.set('')
        self.section_input.set(0, propagate=False)
        print(f'NoiseReviewNode: {len(files)} files, {total_issues} issues '
              f'built from {len(self.raw_data)} reports')

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
            print(f'\n  ── {fentry["filename"]} ──  {fentry["assessment"]}')
            self.current_filepath = fentry['filepath']
        if issue is not None:
            print(f'  [{issue_idx + 1}/{ni}] {desc}')
            if itype in ('corruption_zone', 'glitch_cluster', 'clean_section') and end_frame != frame:
                print(f'         end frame: {end_frame}')
        else:
            print(f'  {desc}')

        # Send outputs — path first so OpenTakeNode loads before seeking;
        # bang last so downstream knows all values are settled.
        self.path_out.send(fentry['filepath'])
        self.end_frame_out.send(end_frame)
        self.frame_out.send(frame)
        self.bang_out.send('bang')
