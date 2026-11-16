"""Radar gesture step counter: one held-state rule for all four directions.

Install: py -m pip install pyserial matplotlib
Record:  py gesture_step_counter.py --port COM5 --csv walk.csv --capture walk.bin
Live graphs: add --live-plot. Otherwise graphs open after Ctrl+C or --seconds.
Replay:  py gesture_step_counter.py --file walk.bin --csv replay.csv

--mode walk (default): UP_TO_DOWN and DOWN_TO_UP count.
--mode stairs: LEFT_TO_RIGHT and RIGHT_TO_LEFT count as stair candidates.
--mode both: all four count. These activity mappings need experimental validation.
After a counted gesture, any quick follow-up among ALL four directions
(including a repeated direction after release) belongs to the same step.
--merge-frames (default 12) measures the gap between confirmed transitions.
Each merged transition renews this window, keeping a quick chain in one step.
A sustained held label never counts later just because the window expires.
--release-frames confident NO_GESTURE frames release a held label; ambiguous,
missing, or unrelated gesture frames cannot release it.

Defaults: confidence .60, winner-minus-runner-up margin .15, confirmation 4
consecutive frames, release 2 frames. These are adjustable starting settings.
A long merge window may combine distinct fast steps. Validate with manual counts.
Foot labels follow your mapping: UP_TO_DOWN/LEFT_TO_RIGHT = LEFT;
DOWN_TO_UP/RIGHT_TO_LEFT = RIGHT. Foot identity uses the majority of confident frames in confirmed gesture runs
within the held step. Runs contribute their confirmation frames once, then one
vote per further frame. Only gestures enabled by --mode vote. Ties retain the
current foot. Live foot totals are provisional and can be corrected without
adding a step. CSV step_foot is the initial label; last_step_foot, step_id,
left_votes/right_votes and foot_reclassified track subsequent corrections. Stair classification is experimental.
CSV retains all ten firmware features and all ten ANN probabilities. Feature
units remain firmware units. Header num_detected_objects is not feature numPoints.
The firmware must already be running; this script sends no CLI commands.
Optional --frame-period-ms enables sensor-time and cadence estimates. Host receive
timestamps are not sensor acquisition times. Missing packets break confirmation.
Old sequence, burst, leg-mapping and alternation switches were removed.
"""
import argparse
import csv
import math
from pathlib import Path
import struct
import sys
import time

MAGIC = bytes([2, 1, 4, 3, 6, 5, 8, 7])
HEADER = ['version', 'packet_length', 'platform', 'frame', 'cpu_cycles',
          'num_detected_objects', 'num_tlvs', 'subframe']
FEATURES = ['weightedDoppler', 'weightedPositiveDoppler', 'weightedNegativeDoppler',
            'weightedRange', 'numPoints', 'weightedAzimuthMean', 'weightedElevationMean',
            'azimuthDopplerCorrelation', 'weightedAzimuthDispersion', 'weightedElevationDispersion']
LABELS = ['NO_GESTURE', 'LEFT_TO_RIGHT', 'RIGHT_TO_LEFT', 'UP_TO_DOWN',
          'DOWN_TO_UP', 'TWIRL_CW', 'TWIRL_CCW', 'OFF', 'ON', 'SHINE']
PROBS = ['prob_'+name for name in LABELS]
STATS = ['interframe_processing_us', 'transmit_output_us', 'interframe_margin_us',
         'interchirp_margin_us', 'active_frame_cpu_percent', 'interframe_cpu_percent']
FIELDS = HEADER + ['tlv_types', 'features_present', 'probabilities_present',
                  'sequence_status', 'missing_frames'] + FEATURES + PROBS + [
                  'top_probability_class', 'top_probability'] + STATS

class Parser:
    def __init__(self):
        self.buffer = bytearray()
        self.invalid_packets = 0
        self.discarded_bytes = 0

    def feed(self, data):
        self.buffer.extend(data)
        while True:
            index = self.buffer.find(MAGIC)
            if index < 0:
                n = max(0, len(self.buffer)-7)
                self.discarded_bytes += n
                del self.buffer[:n]
                return
            self.discarded_bytes += index
            del self.buffer[:index]
            if len(self.buffer) < 40:
                return
            header = struct.unpack_from('<8I', self.buffer, 8)
            size, count = header[1], header[6]
            if size < 40 or size > 1048576 or count > (size-40)//8:
                self.invalid_packets += 1
                del self.buffer[0]
                continue
            if len(self.buffer) < size:
                return
            packet = bytes(self.buffer[:size])
            try:
                row = self.decode(packet, header)
            except ValueError:
                self.invalid_packets += 1
                del self.buffer[0]
                continue
            del self.buffer[:size]
            yield row

    @staticmethod
    def decode(packet, header):
        row = dict(zip(HEADER, header))
        row.update(features_present=0, probabilities_present=0)
        offset, types = 40, []
        for _ in range(header[6]):
            if offset+8 > len(packet):
                raise ValueError('Truncated TLV header')
            kind, length = struct.unpack_from('<II', packet, offset)
            offset += 8
            if offset+length > len(packet):
                raise ValueError('Truncated TLV payload')
            if kind in (1050,1051,6) and kind in types:
                raise ValueError('Duplicate known TLV')
            types.append(kind)
            if kind in (1050,1051):
                if length != 40:
                    raise ValueError('Incorrect feature/probability length')
                values = struct.unpack_from('<10f', packet, offset)
                if not all(math.isfinite(v) for v in values):
                    raise ValueError('Nonfinite feature/probability')
                if kind == 1050:
                    row.update(zip(FEATURES, values))
                    row['features_present'] = 1
                else:
                    row.update(zip(PROBS, values))
                    row['probabilities_present'] = 1
                    best = max(range(10), key=lambda i: values[i])
                    row['top_probability_class'] = LABELS[best]
                    row['top_probability'] = values[best]
            elif kind == 6:
                if length != 24:
                    raise ValueError('Incorrect stats length')
                row.update(zip(STATS, struct.unpack_from('<6I', packet, offset)))
            # Unknown TLVs are skipped; remaining packet bytes are alignment padding.
            offset += length
        row['tlv_types'] = ';'.join(map(str, types))
        return row


VERTICAL = ('UP_TO_DOWN', 'DOWN_TO_UP')
HORIZONTAL = ('LEFT_TO_RIGHT', 'RIGHT_TO_LEFT')
GESTURES = VERTICAL + HORIZONTAL
FOOT = {'UP_TO_DOWN': 'LEFT', 'LEFT_TO_RIGHT': 'LEFT',
        'DOWN_TO_UP': 'RIGHT', 'RIGHT_TO_LEFT': 'RIGHT'}
MODES = {"walk": VERTICAL, "stairs": HORIZONTAL, "both": GESTURES}
COUNT_FIELDS = dict(zip(('LEFT_TO_RIGHT', 'RIGHT_TO_LEFT', 'UP_TO_DOWN', 'DOWN_TO_UP'),
                       ('left_to_right_count', 'right_to_left_count', 'up_to_down_count', 'down_to_up_count')))
EXTRA = ['step_id', 'left_votes', 'right_votes', 'foot_reclassified', 'step_foot', 'last_step_foot', 'left_steps', 'right_steps', 'mode', 'host_received_utc', 'host_elapsed_s', 'sensor_elapsed_s', 'segment',
         'counter_status', 'step_event', 'estimated_steps', 'vertical_steps',
         'horizontal_steps', *COUNT_FIELDS.values(), 'last_confirmed_gesture',
         'held_gesture', 'confirmation_streak', 'probability_margin',
         'step_interval_s', 'instant_cadence_spm', 'confidence_threshold',
         'margin_threshold', 'confirm_frames', 'release_frames', 'merge_frames',
         'frame_period_ms']


class StepCounter:
    def __init__(self, confidence=.6, margin=.15, confirm=4, release=2, merge=12, mode="walk"):
        self.confidence, self.margin = confidence, margin
        self.confirm, self.release, self.merge = confirm, release, merge
        self.allowed = MODES[mode]
        self.counts = dict.fromkeys(GESTURES, 0)
        self.feet = {'LEFT': 0, 'RIGHT': 0}
        self.reset()

    def reset(self, gap=False):
        self.candidate, self.streak, self.neutral = None, 0, 0
        self.observed_margin = None
        self.foot_reclassified = False
        if not gap:
            self.held = self.last_transition = self.last_count = None
            self.last_confirmed = ''
            self.group_counted = False
            self.group_foot = ''
            self.votes = {'LEFT': 0, 'RIGHT': 0}

    @property
    def total(self):
        return sum(self.counts.values())

    def foot_count(self, foot):
        return self.feet[foot]

    def add_votes(self, label, amount):
        if not self.group_counted or label not in self.allowed:
            return
        self.votes[FOOT[label]] += amount
        other = 'RIGHT' if self.group_foot == 'LEFT' else 'LEFT'
        if self.votes[other] > self.votes[self.group_foot]:
            self.feet[self.group_foot] -= 1
            self.feet[other] += 1
            self.group_foot = other
            self.foot_reclassified = True

    def update(self, row, tick):
        self.foot_reclassified = False
        self.observed_margin = None
        values = [row.get(p, float('nan')) for p in PROBS]
        if not row.get('probabilities_present') or not all(math.isfinite(v) and 0 <= v <= 1.00001 for v in values):
            self.candidate, self.streak, self.neutral = None, 0, 0
            return '', 'missing_or_invalid_probabilities', None
        best = max(range(len(values)), key=values.__getitem__)
        ordered = sorted(values, reverse=True)
        self.observed_margin = ordered[0]-ordered[1]
        label = LABELS[best]
        if values[best] < self.confidence or self.observed_margin < self.margin:
            self.candidate, self.streak, self.neutral = None, 0, 0
            return '', 'ambiguous', None
        if label not in GESTURES:
            self.candidate, self.streak = None, 0
            self.neutral = self.neutral + 1 if label == 'NO_GESTURE' else 0
            if self.neutral >= self.release:
                self.held = None
            return '', 'neutral' if label == 'NO_GESTURE' else 'other_gesture', None
        self.neutral = 0
        self.streak = self.streak+1 if label == self.candidate else 1
        self.candidate = label
        if self.streak < self.confirm:
            return '', 'confirming', None
        if label == self.held:
            self.add_votes(label, self.confirm if self.streak == self.confirm else 1)
            return '', 'foot_reclassified' if self.foot_reclassified else 'held', None
        quick = self.group_counted and self.last_transition is not None and tick-self.last_transition <= self.merge
        self.held = label
        self.last_transition = tick
        if quick:
            self.add_votes(label, self.streak)
            return '', 'foot_reclassified' if self.foot_reclassified else 'merged_and_held', None
        self.group_counted = False
        if label not in self.allowed:
            return '', 'direction_outside_mode', None
        self.group_counted = True
        self.group_foot = FOOT[label]
        self.votes = {'LEFT': 0, 'RIGHT': 0}
        self.votes[self.group_foot] = self.streak
        self.feet[self.group_foot] += 1
        interval = None if self.last_count is None else tick-self.last_count
        self.counts[label] += 1
        self.last_count, self.last_confirmed = tick, label
        return label, 'counted', interval


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


def show_recording(path, subframe, total_rows):
    """Read the saved CSV after the serial port/files are closed."""
    dashboard = FeatureDashboard(window=max(2, 2*total_rows), full_history=True)
    with Path(path).open(newline='', encoding='utf-8') as source:
        for row in csv.DictReader(source):
            if int(row['subframe']) != subframe or row['sequence_status'] == 'duplicate':
                continue
            for key in FEATURES + PROBS + ['top_probability']:
                if row.get(key, '') != '':
                    row[key] = float(row[key])
                else:
                    row.pop(key, None)
            for key in ('segment', 'probabilities_present', 'estimated_steps',
                        'left_to_right_count', 'right_to_left_count', 'vertical_steps',
                        'horizontal_steps', 'up_to_down_count', 'down_to_up_count', 'left_steps', 'right_steps'):
                row[key] = int(row.get(key) or 0)
            dashboard.add(row)
    dashboard.finish()
    return dashboard

def main():
    from contextlib import ExitStack
    from datetime import datetime, timezone

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--port', help='Data UART port, e.g. COM5')
    source.add_argument('--file', type=Path, help='Raw binary capture to replay')
    ap.add_argument('--csv', type=Path, required=True)
    ap.add_argument('--capture', type=Path, help='Save binary capture (live only)')
    ap.add_argument('--baud', type=int, default=921600)
    ap.add_argument('--seconds', type=float, help='Stop recording automatically')
    ap.add_argument('--mode', choices=tuple(MODES), default='walk',
                    help='walk: up/down; stairs: left/right candidates; both: all four')
    ap.add_argument('--confidence', type=float, default=.6)
    ap.add_argument('--margin', type=float, default=.15)
    ap.add_argument('--confirm-frames', type=int, default=4)
    ap.add_argument('--release-frames', type=int, default=2)
    ap.add_argument('--merge-frames', type=int, default=12)
    ap.add_argument('--frame-period-ms', type=float, help='Actual firmware frame period, for cadence')
    ap.add_argument('--subframe', type=int, default=0, help='Count this subframe; record all')
    ap.add_argument('--window-frames', type=int, default=200, help='History length for live graphs')
    ap.add_argument('--replay-delay', type=float, default=.05, help='Live replay display delay in seconds')
    plots = ap.add_mutually_exclusive_group()
    plots.add_argument('--live-plot', action='store_true')
    plots.add_argument('--no-plot', action='store_true')
    ap.add_argument('--quiet', action='store_true', help='Only print counted events')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    for name in ('confidence', 'margin'):
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= 1:
            ap.error(name+' must be between 0 and 1')
    for name in ('baud', 'confirm_frames', 'release_frames', 'merge_frames'):
        if getattr(args, name) < 1:
            ap.error(name+' must be positive')
    for name in ('seconds', 'frame_period_ms'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            ap.error(name+' must be finite and positive')
    if args.subframe < 0 or args.window_frames < 2:
        ap.error('Subframe must be nonnegative and window-frames at least 2')
    if not math.isfinite(args.replay_delay) or args.replay_delay < 0:
        ap.error('replay-delay must be finite and nonnegative')
    if args.file and (args.capture or args.seconds):
        ap.error('--capture and --seconds require --port')
    paths = [p.resolve() for p in (args.file, args.csv, args.capture) if p]
    if len(paths) != len(set(paths)):
        ap.error('Input and output paths must differ')
    for path in (args.csv, args.capture):
        if path and path.exists() and not args.overwrite:
            ap.error(f'{path} already exists; choose another filename or use --overwrite')
    if args.port:
        try:
            import serial
        except ImportError:
            ap.error('Install: py -m pip install pyserial')
    if not args.no_plot:
        try:
            import matplotlib.pyplot
        except ImportError:
            ap.error('Install: py -m pip install matplotlib')

    counter = StepCounter(args.confidence, args.margin, args.confirm_frames,
                          args.release_frames, args.merge_frames, args.mode)
    parser = Parser()
    period = args.frame_period_ms/1000 if args.frame_period_ms else None
    dashboard = FeatureDashboard(args.window_frames, period) if args.live_plot else None
    previous, ticks, segments = {}, {}, {}
    rows = selected_rows = probabilities = 0
    settings = dict(mode=args.mode, confidence_threshold=args.confidence, margin_threshold=args.margin,
                    confirm_frames=args.confirm_frames, release_frames=args.release_frames,
                    merge_frames=args.merge_frames, frame_period_ms=args.frame_period_ms or '')
    start = notice = time.monotonic()
    print(f'Mode: {args.mode}. Quick transitions among all four directions stay held with a counted step.')
    print('Ctrl+C stops and saves. Motion features retain firmware units.')
    try:
        with ExitStack() as stack:
            stream = stack.enter_context(serial.Serial(args.port, args.baud, timeout=.2)
                                         if args.port else args.file.open('rb'))
            output = stack.enter_context(args.csv.open('w' if args.overwrite else 'x', newline='', encoding='utf-8'))
            raw = stack.enter_context(args.capture.open('wb' if args.overwrite else 'xb')) if args.capture else None
            writer = csv.DictWriter(output, fieldnames=FIELDS+EXTRA)
            writer.writeheader()
            while args.seconds is None or time.monotonic()-start < args.seconds:
                if dashboard and not dashboard.refresh():
                    break
                chunk = stream.read(max(1, min(stream.in_waiting, 65536))) if args.port else stream.read(65536)
                if not chunk and args.file:
                    break
                now = time.monotonic()
                received = datetime.now(timezone.utc).isoformat() if args.port else ''
                if raw and chunk:
                    raw.write(chunk)
                    raw.flush()
                for row in parser.feed(chunk):
                    if dashboard and dashboard.closed:
                        break
                    frame, sub = row['frame'], row['subframe']
                    status, missing = 'first', 0
                    if sub not in previous:
                        ticks[sub], segments[sub] = 0, 0
                    else:
                        delta = (frame-previous[sub]) % 2**32
                        if delta == 0:
                            status = 'duplicate'
                        elif delta < 2**31:
                            status, missing = ('continuous' if delta == 1 else 'gap'), delta-1
                            ticks[sub] += delta
                        else:
                            status = 'reset_or_out_of_order'
                            ticks[sub] = 0
                            segments[sub] += 1
                    previous[sub] = frame
                    event, reason, interval = '', 'other_subframe', None
                    if sub == args.subframe:
                        selected_rows += 1
                        if status in ('gap', 'reset_or_out_of_order'):
                            counter.reset(gap=(status == 'gap'))
                        if status == 'duplicate':
                            reason = 'duplicate'
                        else:
                            event, reason, interval = counter.update(row, ticks[sub])
                    seconds = interval*period if interval is not None and period else None
                    row.update(settings)
                    row.update(step_foot=FOOT.get(event, ''), last_step_foot=counter.group_foot,
                               step_id=counter.total, left_votes=counter.votes['LEFT'], right_votes=counter.votes['RIGHT'],
                               foot_reclassified=int(counter.foot_reclassified and sub == args.subframe and status != 'duplicate'),
                               left_steps=counter.foot_count('LEFT'), right_steps=counter.foot_count('RIGHT'),
                               sequence_status=status, missing_frames=missing, segment=segments[sub],
                               host_received_utc=received, host_elapsed_s=round(now-start, 6) if args.port else '',
                               sensor_elapsed_s=round(ticks[sub]*period, 6) if period else '',
                               counter_status=reason, step_event=event, estimated_steps=counter.total,
                               vertical_steps=sum(counter.counts[g] for g in VERTICAL),
                               horizontal_steps=sum(counter.counts[g] for g in HORIZONTAL),
                               last_confirmed_gesture=counter.last_confirmed, held_gesture=counter.held or '',
                               confirmation_streak=counter.streak,
                               probability_margin=counter.observed_margin if sub == args.subframe and status != 'duplicate' and counter.observed_margin is not None else '',
                               step_interval_s=round(seconds, 6) if seconds else '',
                               instant_cadence_spm=round(60/seconds, 2) if seconds else '')
                    row.update({field: counter.counts[g] for g, field in COUNT_FIELDS.items()})
                    writer.writerow(row)
                    rows += 1
                    probabilities += row['probabilities_present']
                    if dashboard and sub == args.subframe and status != 'duplicate':
                        dashboard.add(row)
                        dashboard.refresh()
                        if args.file and args.replay_delay:
                            dashboard.plt.pause(args.replay_delay)
                    if event or row["foot_reclassified"] or (not args.quiet and not dashboard):
                        print(f'Frame {frame} | Steps: {counter.total} | Left: {counter.foot_count("LEFT")} | Right: {counter.foot_count("RIGHT")} | Last foot: {(counter.group_foot or "NONE")} | {reason}')
                if chunk:
                    output.flush()
                if args.port and now-notice >= 5:
                    if not probabilities or not selected_rows:
                        print('No usable probability stream yet; check Data UART, firmware startup and subframe.')
                    notice = now
    except KeyboardInterrupt:
        print('\nStopped.')
    print(f'Saved {rows} frames to {args.csv}. Estimated steps: {counter.total}; '
          f'left: {counter.foot_count("LEFT")}; right: {counter.foot_count("RIGHT")}.')
    print(f'Stream: {parser.invalid_packets} invalid packets; {parser.discarded_bytes} discarded bytes; {len(parser.buffer)} trailing bytes.')
    if dashboard:
        dashboard.finish()
    elif not args.no_plot and selected_rows:
        show_recording(args.csv, args.subframe, rows)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
