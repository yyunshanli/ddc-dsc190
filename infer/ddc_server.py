import os
import shutil
import uuid
import zipfile
import tempfile
import pickle
import numpy as np
from flask import Flask, request, send_file, send_from_directory
from essentia.standard import MetadataReader
from scipy.signal import argrelextrema
from learn.extract_feats import extract_mel_feats, create_analyzers
from learn.onset_net import OnsetNet # step placement
from learn.sym_net import SymNet # step selection
from learn.util import make_onset_feature_context

# Flask app initialization
app = Flask(__name__, static_url_path='', static_folder='frontend')

# Constants and configuration
_DIFFS = ['Beginner', 'Easy', 'Medium', 'Hard', 'Challenge']
_DIFF_TO_COARSE_FINE_AND_THRESHOLD = {
    'Beginner': (0, 1, 0.15325437),
    'Easy': (1, 3, 0.23268291),
    'Medium': (2, 5, 0.29456162),
    'Hard': (3, 7, 0.29084727),
    'Challenge': (4, 9, 0.28875697)
}
_SUBDIV = 192
_DT = 0.01
_HZ = 1.0 / _DT
_BPM = 60 * (1.0 / _DT) * (1.0 / _SUBDIV) * 4.0
_PACK_NAME = 'DanceDanceConvolutionV1'
_CHART_TEMPL = """\
#NOTES:
    dance-single:
    DanceDanceConvolutionV1:
    {ccoarse}:
    {cfine}:
    0.0,0.0,0.0,0.0,0.0:
{measures};\
"""
_TEMPL = """\
#TITLE:{title};
#ARTIST:{artist};
#MUSIC:{music_fp};
#OFFSET:0.0;
#BPMS:0.0={bpm};
#STOPS:;
{charts}\
"""

class CreateChartException(Exception):
    pass

def weighted_pick(weights):
    return np.random.choice(len(weights), p=weights / np.sum(weights))

def load_sp_model(ckpt_fp, batch_size=128):
    """Load the step placement model."""
    model_sp = OnsetNet(
        mode='gen',
        batch_size=batch_size,
        audio_context_radius=7,
        audio_nbands=80,
        audio_nchannels=3,
        nfeats=5,
        cnn_filter_shapes=[(7, 3, 10), (3, 3, 20)],
        cnn_pool=[(1, 3), (1, 3)],
        dnn_sizes=[256, 128],
        dnn_nonlin='relu'
    )
    model_sp.load_weights(ckpt_fp)
    return model_sp

def load_ss_model(ckpt_fp):
    """Load the step selection model."""
    model_ss = SymNet(
        mode='gen',
        batch_size=1,
        nunroll=1,
        sym_in_type='bagofarrows',
        sym_embedding_size=0,
        sym_out_type='onehot',
        sym_narrows=4,
        sym_narrowclasses=4,
        rnn_cell_type='lstm',
        rnn_size=128,
        rnn_nlayers=2
    )
    model_ss.load_weights(ckpt_fp)
    return model_ss

def create_chart_dir(artist, title, audio_fp, norm, analyzers, sp_model, sp_batch_size, diffs, ss_model, idx_to_label, out_dir, delete_audio=False):
    """Generate a StepMania chart directory."""
    if not artist or not title:
        metadata = MetadataReader(filename=audio_fp)()
        artist = artist or metadata[1] or 'Unknown Artist'
        title = title or metadata[0] or 'Unknown Title'

    song_feats = (extract_mel_feats(audio_fp, analyzers, nhop=441) - norm[0]) / norm[1]
    song_len_sec = song_feats.shape[0] / _HZ

    diff_chart_txts = []
    for diff in diffs:
        coarse, fine, threshold = _DIFF_TO_COARSE_FINE_AND_THRESHOLD[diff]
        feats_other = np.zeros((sp_batch_size, 1, 5), dtype=np.float32)
        feats_other[:, :, coarse] = 1.0
        feats_audio = np.zeros((sp_batch_size, 1, 15, 80, 3), dtype=np.float32)

        predictions = []
        for start in range(0, song_feats.shape[0], sp_batch_size):
            for i, frame_idx in enumerate(range(start, start + sp_batch_size)):
                feats_audio[i] = make_onset_feature_context(song_feats, frame_idx, 7)
            prediction = sp_model(feats_audio, training=False).numpy()[:, 0]
            predictions.append(prediction)

        predictions = np.concatenate(predictions)[:song_feats.shape[0]]
        predictions_smoothed = np.convolve(predictions, np.hamming(5), 'same')
        maxima = argrelextrema(predictions_smoothed, np.greater_equal, order=1)[0]
        placed_times = [i * _DT for i in maxima if predictions[i] >= threshold]

        state = ss_model.reset_states()
        step_prev = '<-1>'
        times_arr = [placed_times[0]] + placed_times + [placed_times[-1]]
        selected_steps = []

        for i in range(1, len(times_arr) - 1):
            dt_prev, dt_next = times_arr[i] - times_arr[i - 1], times_arr[i + 1] - times_arr[i]
            scores = ss_model.predict_step(step_prev, dt_prev, dt_next, state)
            step_idx = weighted_pick(scores)
            step = idx_to_label[step_idx]
            selected_steps.append(step)
            step_prev = step

        time_to_step = {int(round(t * _HZ)): step for t, step in zip(placed_times, selected_steps)}
        max_subdiv = (max(time_to_step.keys()) + _SUBDIV - 1) // _SUBDIV * _SUBDIV
        full_steps = [time_to_step.get(i, '0000') for i in range(max_subdiv)]
        measures = [full_steps[i:i + _SUBDIV] for i in range(0, max_subdiv, _SUBDIV)]
        measures_txt = '\n,\n'.join(['\n'.join(measure) for measure in measures])

        chart_txt = _CHART_TEMPL.format(ccoarse=_DIFFS[coarse], cfine=fine, measures=measures_txt)
        diff_chart_txts.append(chart_txt)

    audio_out_name = os.path.basename(audio_fp)
    sm_txt = _TEMPL.format(title=title, artist=artist, music_fp=audio_out_name, bpm=_BPM, charts='\n'.join(diff_chart_txts))

    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(audio_fp, os.path.join(out_dir, audio_out_name))
    with open(os.path.join(out_dir, f"{os.path.basename(out_dir)}.sm"), 'w') as f:
        f.write(sm_txt)

    if delete_audio:
        os.remove(audio_fp)

    return True

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/choreograph', methods=['POST'])
def choreograph():
    uploaded_file = request.files.get('audio_file')
    if not uploaded_file:
        return 'Audio file required', 400

    song_artist = request.form.get('song_artist', '')[:1024]
    song_title = request.form.get('song_title', '')[:1024]
    diff_coarse = request.form.get('diff_coarse')

    if diff_coarse not in _DIFFS:
        return 'Invalid difficulty specified', 400

    out_dir = tempfile.mkdtemp()
    song_id = uuid.uuid4()
    song_fp = os.path.join(out_dir, f"{song_id}{os.path.splitext(uploaded_file.filename)[1]}")
    uploaded_file.save(song_fp)

    try:
        create_chart_dir(song_artist, song_title, song_fp, NORM, ANALYZERS, SP_MODEL, ARGS.sp_batch_size, [diff_coarse], SS_MODEL, IDX_TO_LABEL, out_dir)
    except CreateChartException as e:
        shutil.rmtree(out_dir)
        return str(e), 500
    except Exception as e:
        shutil.rmtree(out_dir)
        return 'Unknown error', 500

    with tempfile.NamedTemporaryFile(suffix='.zip') as z:
        with zipfile.ZipFile(z.name, 'w', zipfile.ZIP_DEFLATED) as f:
            for fn in os.listdir(out_dir):
                f.write(os.path.join(out_dir, fn), os.path.join(_PACK_NAME, str(song_id), fn))
        shutil.rmtree(out_dir)
        return send_file(z.name, as_attachment=True, attachment_filename=f'{song_id}.zip')

@app.after_request
def add_header(r):
    r.headers.update({
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': '*',
        'Access-Control-Allow-Methods': '*',
        'Access-Control-Expose-Headers': '*'
    })
    return r

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--norm_pkl_fp', type=str, default='server_aux/norm.pkl')
    parser.add_argument('--sp_ckpt_fp', type=str, default='server_aux/model_sp-56000.h5')
    parser.add_argument('--ss_ckpt_fp', type=str, default='server_aux/model_ss-23628.h5')
    parser.add_argument('--labels_txt_fp', type=str, default='server_aux/labels_4_0123.txt')
    parser.add_argument('--sp_batch_size', type=int, default=256)
    parser.add_argument('--max_file_size', type=int, default=None)
    ARGS = parser.parse_args()

    with open(ARGS.norm_pkl_fp, 'rb') as f:
        NORM = pickle.load(f)
    ANALYZERS = create_analyzers(nhop=441)
    with open(ARGS.labels_txt_fp, 'r') as f:
        IDX_TO_LABEL = {i + 1: l for i, l in enumerate(f.read().splitlines())}

    SP_MODEL = load_sp_model(ARGS.sp_ckpt_fp, ARGS.sp_batch_size)
    SS_MODEL = load_ss_model(ARGS.ss_ckpt_fp)

    app.config['MAX_CONTENT_LENGTH'] = ARGS.max_file_size
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 80)))