import numpy as np
import random
from collections import Counter
from beatcalc import BeatCalc
from util import make_onset_feature_context, np_pad


class Chart:
    def __init__(self, song_metadata, metadata, annotations):
        self.song_metadata = song_metadata
        self.metadata = metadata
        self.annotations = annotations
        self.label_counts = Counter()

        self.beat_calc = BeatCalc(song_metadata['offset'], song_metadata['bpms'], song_metadata['stops'])
        self.first_annotation_time = annotations[0][2]
        self.last_annotation_time = annotations[-1][2]
        self.time_annotated = self.last_annotation_time - self.first_annotation_time
        self.annotations_per_second = len(annotations) / self.time_annotated

        assert len(annotations) >= 2 and self.time_annotated > 0

    def get_metadata_field(self, index):
        return self.metadata[index]

    def get_nannotations(self):
        return len(self.annotations)


class OnsetChart(Chart):
    def __init__(self, song_metadata, song_features, frame_rate, metadata, annotations):
        super().__init__(song_metadata, metadata, annotations)
        self.song_features = song_features
        self.nframes = song_features.shape[0]
        self.dt = 1.0 / frame_rate

        self.onsets = {int(round(t / self.dt)) for _, _, t, _ in annotations if 0 <= int(round(t / self.dt)) < self.nframes}
        self.first_onset, self.last_onset = min(self.onsets), max(self.onsets)
        self.nframes_annotated = (self.last_onset - self.first_onset) + 1
        self.blanks = set(range(self.nframes)) - self.onsets
        self._blanks_memoized = {}

        assert all([self.first_onset >= 0, self.last_onset < self.nframes, len(self.onsets) > 0])

    def get_example(self, frame_idx, dtype, **kwargs):
        feats_audio = make_onset_feature_context(self.song_features, frame_idx, kwargs.get('time_context_radius', 1))
        feats_other = [np.zeros(0, dtype=dtype)]

        for key, to_id in [('diff_feet_to_id', 'get_foot_difficulty'), ('diff_coarse_to_id', 'get_metadata_field')]:
            if kwargs.get(key):
                value = to_id(kwargs[key])
                onehot = np.zeros(max(kwargs[key].values()) + 1, dtype=dtype)
                
                if kwargs.get('diff_dipstick'):
                    onehot[:value + 1] = 1.0
                else:
                    onehot[value] = 1.0
                feats_other.append(onehot)

        if kwargs.get('beat_phase') or kwargs.get('beat_phase_cos'):
            beat_phase = self.beat_calc.time_to_beat(frame_idx * self.dt) % 1
            feats_other.append(np.array([beat_phase], dtype=dtype))
            if kwargs.get('beat_phase_cos'):
                feats_other[-1] = np.cos(beat_phase * 2 * np.pi) * 0.5 + 0.5

        y = dtype(frame_idx in self.onsets)
        return np.array(feats_audio, dtype=dtype), np.concatenate(feats_other), y

    def sample(self, n, exclude_onset_neighbors=0, nunroll=0):
        if not self._blanks_memoized:
            valid = set(range(self.first_onset, self.last_onset + 1))
            if exclude_onset_neighbors:
                onset_neighbors = set.union(*[{x + i, x - i} for x in self.onsets for i in range(1, exclude_onset_neighbors + 1)])
                valid -= onset_neighbors
            if nunroll:
                valid -= set(range(self.first_onset, self.first_onset + nunroll))
            self._blanks_memoized = valid

        return random.sample(self._blanks_memoized, n)

    def sample_onsets(self, n):
        return random.sample(self.onsets, n)

    def sample_blanks(self, n, **kwargs):
        if kwargs not in self._blanks_memoized:
            blanks = self.blanks - {x + i for x in self.onsets for i in range(1, kwargs.get('exclude_onset_neighbors', 0) + 1)}
            if kwargs.get('exclude_pre_onsets'):
                blanks -= set(range(self.first_onset))
            if kwargs.get('exclude_post_onsets'):
                blanks -= set(range(self.last_onset, self.nframes))
            if kwargs.get('include_onsets'):
                blanks |= self.onsets
            self._blanks_memoized[kwargs] = blanks

        return random.sample(self._blanks_memoized[kwargs], n)


class SymbolicChart(Chart):
    def __init__(self, song_metadata, song_features, frame_rate, metadata, annotations, pre=1):
        super().__init__(song_metadata, metadata, annotations)
        self.song_features = song_features
        self.dt = 1.0 / frame_rate
        self.sequence = self._create_sequences(pre, annotations)

    def _create_sequences(self, pre, annotations):
        prepend = ['<-{}>'.format(i + 1) for i in range(pre)[::-1]]
        annotations = [annotations[0][:3] + [p] for p in prepend] + annotations
        sequences = []

        for i in range(len(annotations) - 1):
            pulse_last, beat_last, time_last, label_last = annotations[i]
            pulse, beat, time, label = annotations[i + 1]
            sequences.append((label_last, beat_last, time_last))

        sequences.append((label, beat, time))
        return sequences

    def get_subsequence(self, subseq_start, subseq_len, dtype=np.float32, **kwargs):
        syms = self.sequence[subseq_start:subseq_start + subseq_len + 1]
        feats = np.zeros((len(syms) - 1, 0), dtype=dtype)

        for key in ['meas_phase_cos', 'beat_phase_cos', 'beat_diff', 'time_diff']:
            if kwargs.get(key):
                feats = np.append(feats, np.array(getattr(self, f'seq_{key}')[subseq_start:subseq_start + len(syms) - 1], dtype=dtype)[:, np.newaxis], axis=1)

        feats_audio = np.zeros((len(syms) - 1, 0, 0, 0), dtype=dtype)
        if kwargs.get('audio_time_context_radius') >= 0 and self.song_features is not None:
            feats_audio = np.zeros((len(syms) - 1, kwargs['audio_time_context_radius'] * 2 + 1) + self.song_features.shape[1:], dtype=dtype)
            for i, frame_idx in enumerate(self.seq_frame_idxs[subseq_start:subseq_start + len(syms) - 1]):
                if kwargs.get('audio_deviation_max'):
                    frame_idx += random.randint(-kwargs['audio_deviation_max'], kwargs['audio_deviation_max'])
                feats_audio[i] = make_onset_feature_context(self.song_features, frame_idx, kwargs['audio_time_context_radius'])

        return syms, feats, feats_audio