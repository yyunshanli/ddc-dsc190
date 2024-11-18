import argparse
import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score
from onset_net import OnsetNet
from util import open_dataset_fps, flatten_dataset_to_charts, stride_csv_arg_list, apply_z_norm, calc_mean_std_per_band

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate an OnsetNet model.")
    # Data arguments
    parser.add_argument('--train_txt_fp', type=str, default='', help='Training dataset txt file with a list of pickled song files')
    parser.add_argument('--valid_txt_fp', type=str, default='', help='Eval dataset txt file with a list of pickled song files')
    parser.add_argument('--z_score', action='store_true', help='If true, train and test on z-score of training data')
    parser.add_argument('--test_txt_fp', type=str, default='', help='Test dataset txt file with a list of pickled song files')
    parser.add_argument('--model_ckpt_fp', type=str, default='', help='File path to model checkpoint if resuming or eval')
    parser.add_argument('--experiment_dir', type=str, default='', help='Directory for temporary training files and model weights')
    
    # Features arguments
    parser.add_argument('--audio_context_radius', type=int, default=7, help='Past and future context per training example')
    parser.add_argument('--audio_nbands', type=int, default=80, help='Number of bands per frame')
    parser.add_argument('--audio_nchannels', type=int, default=3, help='Number of channels per frame')
    
    # Network params
    parser.add_argument('--cnn_filter_shapes', type=str, default='', help='CSV 3-tuples of filter shapes (time, freq, n)')
    parser.add_argument('--cnn_pool', type=str, default='', help='CSV 2-tuples of pool amounts (time, freq)')
    parser.add_argument('--rnn_cell_type', type=str, default='lstm', help='Type of RNN cell: rnn, gru, or lstm')
    parser.add_argument('--rnn_size', type=int, default=0, help='Size of RNN cell')
    parser.add_argument('--rnn_nlayers', type=int, default=0, help='Number of RNN layers')
    parser.add_argument('--dnn_sizes', type=str, default='', help='CSV sizes for dense layers')
    
    # Training params
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--nepochs', type=int, default=10, help='Number of training epochs')
    
    return parser.parse_args()

def load_and_preprocess_data(train_fp, valid_fp, test_fp, z_score, experiment_dir):
    train_data, valid_data, test_data = open_dataset_fps(train_fp, valid_fp, test_fp)
    if z_score:
        z_score_fp = os.path.join(experiment_dir, 'valid_mean_std.pkl')
        if valid_fp and not os.path.exists(z_score_fp):
            mean_per_band, std_per_band = calc_mean_std_per_band(valid_data)
            with open(z_score_fp, 'wb') as f:
                pickle.dump((mean_per_band, std_per_band), f)
        else:
            with open(z_score_fp, 'rb') as f:
                mean_per_band, std_per_band = pickle.load(f)
        for data in [train_data, valid_data, test_data]:
            apply_z_norm(data, mean_per_band, std_per_band)
    return train_data, valid_data, test_data

def create_tf_dataset(data, batch_size):
    # Assuming `data` is a list of tuples (feats_audio, feats_other, targets)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def main(args):
    train_data, valid_data, test_data = load_and_preprocess_data(
        args.train_txt_fp, args.valid_txt_fp, args.test_txt_fp, args.z_score, args.experiment_dir
    )
    charts_train = flatten_dataset_to_charts(train_data)
    charts_valid = flatten_dataset_to_charts(valid_data)
    
    # Prepare datasets
    train_dataset = create_tf_dataset(charts_train, args.batch_size)
    valid_dataset = create_tf_dataset(charts_valid, args.batch_size)

    # Model configuration
    model_config = {
        'audio_context_radius': args.audio_context_radius,
        'audio_nbands': args.audio_nbands,
        'audio_nchannels': args.audio_nchannels,
        'cnn_filter_shapes': stride_csv_arg_list(args.cnn_filter_shapes, 3, int),
        'cnn_pool': stride_csv_arg_list(args.cnn_pool, 2, int),
        'rnn_cell_type': args.rnn_cell_type,
        'rnn_size': args.rnn_size,
        'rnn_nlayers': args.rnn_nlayers,
        'dnn_sizes': stride_csv_arg_list(args.dnn_sizes, 1, int)
    }

    model = OnsetNet(mode='train', batch_size=args.batch_size, **model_config)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    # Callbacks for saving checkpoints and early stopping
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.experiment_dir, 'model_checkpoint.h5'),
        save_best_only=True
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    # Training
    model.fit(
        train_dataset,
        epochs=args.nepochs,
        validation_data=valid_dataset,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )

    # Save final model
    final_model_path = os.path.join(args.experiment_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f'Final model saved at {final_model_path}')

if __name__ == '__main__':
    args = parse_arguments()
    main(args)


