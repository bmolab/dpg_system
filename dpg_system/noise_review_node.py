import json
import os

from dpg_system.node import Node


def register_noise_review_node():
    Node.app.register_node('noise_review', NoiseReviewNode.factory)


class NoiseReviewNode(Node):
    """Step through noise-report issues across a batch of NPZ files.

    Load a JSON produced by estimate_noise_torque.py, then use Prev/Next to
    walk through every stream break, corruption zone, spike frame, and glitch
    cluster.  Outputs the NPZ path and start frame so an OpenTakeNode can load
    the file and jump to the right section automatically.
    """

    @staticmethod
    def factory(name, data, args=None):
        return NoiseReviewNode(name, data, args)

    def __init__(self, label: str, data, args):
        super().__init__(label, data, args)

        self.raw_data = []    # list of report dicts from JSON
        self.issues = []      # flat list of issue dicts
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

        # ── Status labels ─────────────────────────────────────────────────
        self.counter_label = self.add_label('— / —')
        self.file_label    = self.add_label('')
        self.issue_label   = self.add_label('')

        # ── Outputs ───────────────────────────────────────────────────────
        self.path_out  = self.add_output('npz path')
        self.frame_out = self.add_output('frame')

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
                self.raw_data = json.load(f)
            print(f'NoiseReviewNode: loaded {len(self.raw_data)} reports from {os.path.basename(path)}')
            self.rebuild_issues()
        except Exception as e:
            print(f'NoiseReviewNode: failed to load {path}: {e}')

    # ── Issue list construction ───────────────────────────────────────────

    def rebuild_issues(self):
        show_breaks     = self.show_breaks()
        show_corruption = self.show_corruption()
        show_spikes     = self.show_spikes()
        show_glitches   = self.show_glitches()
        show_clean      = self.show_clean()

        cls_rank = {'clean': 0, 'moderate': 1, 'problematic': 2}
        min_rank = {'all': 0, 'moderate': 1, 'problematic': 2}.get(self.min_class(), 0)

        issues = []
        for report in self.raw_data:
            if cls_rank.get(report.get('classification', 'clean'), 0) < min_rank:
                continue

            filepath = report.get('filepath') or report.get('filename', '')
            filename = os.path.basename(filepath)

            if show_breaks:
                for sb in report.get('stream_breaks', []):
                    f = sb.get('frame', 0)
                    issues.append({
                        'filepath': filepath, 'filename': filename,
                        'type': 'stream_break', 'frame': f, 'end_frame': f,
                        'desc': (f"stream break  frame={f}"
                                 f"  type={sb.get('break_type','')}"
                                 f"  worst={sb.get('worst_joint','')}"),
                    })

            if show_corruption:
                surgery  = report.get('surgery') or {}
                excision = surgery.get('excision') or {}
                for z in excision.get('zones', []):
                    s, e = z.get('start', 0), z.get('end', 0)
                    joints = ', '.join((z.get('joints') or [])[:3])
                    issues.append({
                        'filepath': filepath, 'filename': filename,
                        'type': 'corruption_zone', 'frame': s, 'end_frame': e,
                        'desc': (f"corruption  [{s}–{e}]"
                                 f"  {z.get('duration_s', 0):.1f}s"
                                 f"  mean={z.get('mean_vel', 0):.0f} max={z.get('max_vel', 0):.0f} rad/s"
                                 f"  {joints}"),
                    })

            if show_spikes:
                by_frame = {}
                for sf in report.get('spike_frames', []):
                    f = sf.get('frame', 0)
                    by_frame.setdefault(f, []).append(
                        f"{sf.get('joint_name','')} {sf.get('velocity', 0):.0f}r/s ×{sf.get('neighbor_ratio', 0):.1f}")
                for f, entries in sorted(by_frame.items()):
                    issues.append({
                        'filepath': filepath, 'filename': filename,
                        'type': 'spike_frame', 'frame': f, 'end_frame': f,
                        'desc': f"spike  frame={f}  {', '.join(entries[:4])}",
                    })

            if show_glitches:
                for cluster in report.get('glitch_clusters', []):
                    if not (isinstance(cluster, (list, tuple)) and len(cluster) >= 2):
                        continue
                    s, e = int(cluster[0]), int(cluster[1])
                    issues.append({
                        'filepath': filepath, 'filename': filename,
                        'type': 'glitch_cluster', 'frame': s, 'end_frame': e,
                        'desc': f"glitch  [{s}–{e}]  {e - s + 1} frames",
                    })

            if show_clean:
                for seg in report.get('clean_segments', []):
                    s, e = seg.get('start', 0), seg.get('end', 0)
                    issues.append({
                        'filepath': filepath, 'filename': filename,
                        'type': 'clean_section', 'frame': s, 'end_frame': e,
                        'desc': (f"clean  [{s}–{e}]"
                                 f"  {seg.get('duration_s', 0):.1f}s"
                                 f"  {seg.get('n_frames', e - s + 1)} frames"
                                 f"  mean={seg.get('mean_score', 0):.2f} max={seg.get('max_score', 0):.2f}"),
                    })

        self.issues = issues
        self.current_idx = 0 if issues else -1
        total = len(issues)
        self.counter_label.set(f'0 / {total}')
        self.file_label.set('')
        self.issue_label.set('')
        print(f'NoiseReviewNode: {total} issues built from {len(self.raw_data)} reports')

    # ── Navigation ────────────────────────────────────────────────────────

    def go_next(self):
        if not self.issues:
            print('NoiseReviewNode: no issues loaded')
            return
        self.current_idx = (self.current_idx + 1) % len(self.issues)
        self._emit_current()

    def go_prev(self):
        if not self.issues:
            return
        self.current_idx = (self.current_idx - 1) % len(self.issues)
        self._emit_current()

    def _emit_current(self):
        issue  = self.issues[self.current_idx]
        n      = self.current_idx + 1
        total  = len(self.issues)

        # Labels
        self.counter_label.set(f'{n} / {total}')
        self.file_label.set(issue['filename'])
        self.issue_label.set(issue['desc'])

        # Console — flag file changes clearly
        if issue['filepath'] != self.current_filepath:
            print(f'\n  ── {issue["filename"]} ──')
            self.current_filepath = issue['filepath']
        print(f'  [{n}/{total}] {issue["desc"]}')
        if issue['type'] in ('corruption_zone', 'glitch_cluster', 'clean_section') and issue['end_frame'] != issue['frame']:
            print(f'         end frame: {issue["end_frame"]}')

        # Send outputs — path first so OpenTakeNode loads before seeking
        self.path_out.send(issue['filepath'])
        self.frame_out.send(issue['frame'])
