"""Display entire radar CSV recordings immediately, without radar hardware.

Install:  py -m pip install matplotlib
Play:     py csv_playback.py --csv recordings\\e2_regular.csv
Animate:  py csv_playback.py --csv recordings\\e2_regular.csv --animate --speed 2
Summary:  py csv_playback.py --csv recordings\\e2_regular.csv --summary
No path:  opens a CSV file-selection dialog (requires tkinter).

Displays saved features, ANN probabilities, and recorded total/left/right counts.
This player does NOT rerun or modify the counting algorithm or the input file.
By default, the full recording is drawn at once with no timed playback.
With --animate, Space pauses/resumes and R restarts.
In animated mode, host timestamp differences control playback when available. They are NOT sensor
acquisition times. The plot uses received sample index, not physical time.
Without usable host times, playback uses --fps (default 30, display only).
Foot labels are algorithmic assignments, not validated anatomical foot labels.
"""
import argparse
import csv
import math
import time
from pathlib import Path

FEATURES = ['weightedDoppler', 'weightedPositiveDoppler', 'weightedNegativeDoppler',
            'weightedRange', 'numPoints', 'weightedAzimuthMean', 'weightedElevationMean',
            'azimuthDopplerCorrelation', 'weightedAzimuthDispersion', 'weightedElevationDispersion']

LABELS = ['NO_GESTURE', 'LEFT_TO_RIGHT', 'RIGHT_TO_LEFT', 'UP_TO_DOWN',
          'DOWN_TO_UP', 'TWIRL_CW', 'TWIRL_CCW', 'OFF', 'ON', 'SHINE']

PROBS = ['prob_'+name for name in LABELS]

VERTICAL = ('UP_TO_DOWN', 'DOWN_TO_UP')

HORIZONTAL = ('LEFT_TO_RIGHT', 'RIGHT_TO_LEFT')

class FeatureDashboard:
    """Bounded history of the selected subframe; never converts firmware units."""
    def __init__(self, window=200, period=None, full_history=False):
        from collections import deque
        import matplotlib.pyplot as plt
        self.plt, self.period = plt, period
        self.full_history = full_history
        self.x = deque(maxlen=window)
        self.history = {key: deque(maxlen=window) for key in FEATURES + ['estimated_steps', 'left_steps', 'right_steps']}
        self.last_draw = 0
        self.last_segment = None
        self.sample = 0
        self.closed = False
        self.latest = {}
        plt.ion()
        self.fig, axes = plt.subplots(4, 2, figsize=(14, 10), layout='constrained')
        self.fig.canvas.manager.set_window_title('Radar features and gesture step counter')
        self.axes = axes.ravel()
        self.title = self.fig.suptitle('Waiting for radar data...', fontsize=13)
        groups = [
            ('Doppler features', FEATURES[:3], ['Weighted', 'Positive', 'Negative']),
            ('Range feature', ['weightedRange'], ['Weighted range']),
            ('Detected points feature', ['numPoints'], ['Points']),
            ('Mean angles', ['weightedAzimuthMean', 'weightedElevationMean'], ['Azimuth', 'Elevation']),
            ('Angle dispersion', ['weightedAzimuthDispersion', 'weightedElevationDispersion'], ['Azimuth', 'Elevation']),
            ('Azimuth–Doppler correlation', ['azimuthDopplerCorrelation'], ['Correlation']),
        ]
        self.lines = {}
        for ax, (title, keys, labels) in zip(self.axes, groups):
            ax.set_title(title, fontsize=10)
            ax.set_ylabel('Firmware value', fontsize=8)
            for key, label in zip(keys, labels):
                self.lines[key], = ax.plot([], [], label=label, linewidth=1.3)
            if len(keys) > 1:
                ax.legend(loc='upper left', fontsize=7, ncol=len(keys))
        self.prob_ax = self.axes[6]
        self.prob_ax.set_title('Current ANN probabilities (not confirmed gestures)', fontsize=10)
        self.bars = self.prob_ax.bar(range(10), [0]*10,
                                    color=['#3185c5' if label in VERTICAL else '#d77a20' if label in HORIZONTAL else '#aebbc6' for label in LABELS])
        self.prob_ax.set_xticks(range(10), ['None', 'L→R', 'R→L', 'U→D', 'D→U', 'CW', 'CCW', 'Off', 'On', 'Shine'], fontsize=8)
        self.prob_ax.set_ylim(0, 1.05)
        self.prob_ax.set_ylabel('Probability', fontsize=8)
        ax = self.axes[7]
        ax.set_title('Estimated steps — validate with manual count', fontsize=10)
        for key, label, color in [('estimated_steps', 'Total', '#28794e'),
                                  ('left_steps', 'Left foot', '#3185c5'),
                                  ('right_steps', 'Right foot', '#d77a20')]:
            self.lines[key], = ax.step([], [], where='post', color=color, label=label)
        ax.legend(loc='upper left', fontsize=7, ncol=3)
        ax.set_ylabel('Count', fontsize=8)
        for i, ax in enumerate(self.axes):
            ax.grid(alpha=.2)
            if i != 6:
                ax.set_xlabel('Sensor time within segment (s)' if period else 'Received sample index', fontsize=8)
                ax.tick_params(labelsize=8)
        self.fig.canvas.mpl_connect('close_event', lambda event: setattr(self, 'closed', True))
        plt.show(block=False)

    def add(self, row):
        # Reset plot history at frame resets so the time axis cannot fold backward.
        if self.last_segment is not None and row['segment'] != self.last_segment:
            if self.full_history:
                self.x.append(self.sample + 1)
                for values in self.history.values():
                    values.append(float('nan'))
            else:
                self.x.clear()
                for values in self.history.values():
                    values.clear()
        self.last_segment = row['segment']
        self.sample += 1
        x = row['sensor_elapsed_s'] if self.period else self.sample
        if row['sequence_status'] == 'gap':
            # Break lines across missing packets rather than interpolate motion.
            self.x.append(x)
            for values in self.history.values():
                values.append(float('nan'))
        self.x.append(x)
        for key, values in self.history.items():
            values.append(row.get(key, float('nan')))
        self.latest = row.copy()

    def refresh(self, force=False):
        if self.closed:
            return False
        now = time.monotonic()
        if not force and now-self.last_draw < .15:
            self.fig.canvas.flush_events()
            return not self.closed
        self.last_draw = now
        for key, line in self.lines.items():
            line.set_data(list(self.x), list(self.history[key]))
        for i, ax in enumerate(self.axes):
            if i != 6:
                ax.relim()
                ax.autoscale_view()
        row = self.latest
        has_probs = bool(row.get('probabilities_present'))
        for key, bar in zip(PROBS, self.bars):
            value = row.get(key, 0)
            bar.set_height(value if has_probs and 0 <= value <= 1.00001 else 0)
        self.prob_ax.set_title('Raw ANN probabilities (may flicker)' if has_probs else 'Probabilities unavailable in this frame', fontsize=10)
        count = row.get('estimated_steps', 0)
        vertical, horizontal = row.get('vertical_steps', 0), row.get('horizontal_steps', 0)
        confirmed = row.get('last_confirmed_gesture') or 'NONE'
        foot = row.get('last_step_foot') or 'NONE'
        left, right = row.get('left_steps', 0), row.get('right_steps', 0)
        held = row.get('held_gesture') or 'NONE'
        self.title.set_text(f'Estimated steps: {count} | Mode: {row.get("mode", "walk")} | Left foot: {left} | Right foot: {right}\n'
                            f'Latest step: {foot} | Votes L: {row.get("left_votes", 0)} R: {row.get("right_votes", 0)} | Held: {held} | '
                            f'{row.get("counter_status", "")}')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        return not self.closed

    def finish(self):
        if not self.closed:
            self.refresh(force=True)
            self.title.set_text(self.title.get_text() + ' — Recording finished')
            self.plt.ioff()
            self.plt.show(block=True)

def load_recording(path, subframe=0):
    """Validate and convert CSV fields consumed by the original dashboard."""
    numeric = FEATURES + PROBS + ['top_probability']
    integers = ['segment', 'probabilities_present', 'estimated_steps',
                'left_steps', 'right_steps', 'vertical_steps', 'horizontal_steps',
                'left_votes', 'right_votes', 'step_id', 'foot_reclassified']
    required = {'subframe', 'sequence_status', 'estimated_steps', 'left_steps',
                'right_steps', 'probabilities_present', *FEATURES, *PROBS}
    rows = []
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError('Missing required CSV columns: ' + ', '.join(sorted(missing)))
        for line, row in enumerate(reader, start=2):
            try:
                if int(row['subframe']) != subframe or row['sequence_status'] == 'duplicate':
                    continue
                for key in numeric:
                    row[key] = float(row[key]) if row.get(key) else float('nan')
                for key in integers:
                    row[key] = int(row.get(key) or 0)
                if any(row[k] < 0 for k in ['estimated_steps', 'left_steps', 'right_steps']):
                    raise ValueError('negative step count')
                if row['estimated_steps'] != row['left_steps'] + row['right_steps']:
                    raise ValueError('left and right totals do not equal total steps')
                row['host_elapsed_s'] = float(row['host_elapsed_s']) if row.get('host_elapsed_s') else None
                rows.append(row)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'CSV line {line}: {exc}') from exc
    if not rows:
        raise ValueError(f'No nonduplicate rows for subframe {subframe}')
    return rows


def print_summary(path, rows):
    last = rows[-1]
    print(f'Recording: {path.name}')
    print(f'Playback rows: {len(rows)}')
    print(f"Saved final steps: {last['estimated_steps']} | "
          f"Left: {last['left_steps']} | Right: {last['right_steps']}")
    print('Counts are recorded outputs; no algorithm recalculation was performed.')


def show_instant(rows, path):
    """Load all samples, then draw once; probability bars show the final frame."""
    import matplotlib.pyplot as plt
    with plt.ioff():
        dashboard = FeatureDashboard(window=max(2, 2 * len(rows)), full_history=True)
        # Disable interactive redraws while filling the entire history.
        plt.ioff()
        for row in rows:
            dashboard.add(row)
        dashboard.refresh(force=True)
        dashboard.fig.canvas.manager.set_window_title('Full CSV recording: ' + path.name)
        dashboard.prob_ax.set_title('ANN probabilities — final recorded frame', fontsize=10)
        dashboard.title.set_text(dashboard.title.get_text() + ' — Full recording')
        dashboard.fig.canvas.draw_idle()
        plt.show(block=True)
    return dashboard


def playback(rows, path, speed, fps):
    dashboard = FeatureDashboard(window=max(2, 2 * len(rows)), full_history=True)
    dashboard.fig.canvas.manager.set_window_title('CSV playback: ' + path.name)
    dashboard.fig.text(.01, .005, 'Space: pause/resume | R: restart | Close window: exit', fontsize=9)
    state = {'paused': False, 'restart': False}

    def on_key(event):
        if event.key == ' ':
            state['paused'] = not state['paused']
            dashboard.fig.canvas.manager.set_window_title(
                ('PAUSED: ' if state['paused'] else 'CSV playback: ') + path.name)
        elif event.key and event.key.lower() == 'r':
            state['restart'] = True
            state['paused'] = False

    dashboard.fig.canvas.mpl_connect('key_press_event', on_key)
    i = 0
    deadline = time.monotonic()
    while not dashboard.closed:
        if state['restart']:
            dashboard.x.clear()
            for values in dashboard.history.values():
                values.clear()
            dashboard.sample = 0
            dashboard.last_segment = None
            dashboard.latest = {}
            i = 0
            deadline = time.monotonic()
            state['restart'] = False
            dashboard.fig.canvas.manager.set_window_title('CSV playback: ' + path.name)
        if state['paused'] or i == len(rows):
            dashboard.plt.pause(.03)
            # Reset scheduling while paused so resumption does not skip forward.
            deadline = time.monotonic()
            continue
        now = time.monotonic()
        if now < deadline:
            dashboard.plt.pause(min(.03, max(.001, deadline-now)))
            continue
        dashboard.add(rows[i])
        dashboard.refresh(force=(i == 0 or i == len(rows)-1))
        i += 1
        if i == len(rows):
            dashboard.fig.canvas.manager.set_window_title('FINISHED (R to replay): ' + path.name)
            dashboard.title.set_text(dashboard.title.get_text() + ' — Playback finished')
            dashboard.fig.canvas.draw_idle()
            continue
        previous, current = rows[i-1]['host_elapsed_s'], rows[i]['host_elapsed_s']
        delay = 1.0 / fps
        if previous is not None and current is not None:
            difference = current - previous
            if math.isfinite(difference) and difference >= 0:
                delay = difference
        deadline += delay / speed
        # Keep the GUI responsive even for very fast or zero-delay playback.
        dashboard.fig.canvas.flush_events()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', '--file', dest='csv', type=Path, help='Recorded CSV to play')
    ap.add_argument('--animate', action='store_true', help='Use timed playback instead of displaying everything immediately')
    ap.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier (default: 1)')
    ap.add_argument('--fps', type=float, default=30.0, help='Display rate when host timing is unavailable')
    ap.add_argument('--subframe', type=int, default=0, help='Subframe to display (default: 0)')
    ap.add_argument('--summary', action='store_true', help='Print final saved counts without opening graphs')
    args = ap.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in [args.speed, args.fps]):
        ap.error('--speed and --fps must be finite positive numbers')
    if args.subframe < 0:
        ap.error('--subframe must be nonnegative')
    if args.csv is None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            selected = filedialog.askopenfilename(title='Choose a radar recording CSV',
                        filetypes=[('CSV recordings', '*.csv'), ('All files', '*.*')])
            root.destroy()
        except Exception as exc:
            ap.error(f'File picker unavailable ({exc}). Supply --csv "path/to/recording.csv".')
        if not selected:
            return
        args.csv = Path(selected)
    try:
        rows = load_recording(args.csv, args.subframe)
        print_summary(args.csv, rows)
        if not args.summary:
            try:
                import matplotlib
            except ImportError:
                ap.error('Install plotting support: py -m pip install matplotlib')
            if matplotlib.get_backend().lower() in ('agg', 'pdf', 'ps', 'svg', 'cairo', 'template'):
                ap.error('An interactive Matplotlib backend is required. Use a desktop Python installation, or --summary.')
            if args.animate:
                playback(rows, args.csv, args.speed, args.fps)
            else:
                show_instant(rows, args.csv)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    except KeyboardInterrupt:
        print('Playback stopped. The CSV was not changed.')


if __name__ == '__main__':
    main()
