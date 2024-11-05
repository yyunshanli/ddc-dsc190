import os
import argparse
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from onset_cnn import OnsetCNN
from util import apply_z_norm

# Argument parser for command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Test script for OnsetCNN')
    parser.add_argument('--data_txt_fp', type=str, required=True, help='Path to the txt file listing pickled song files')
    parser.add_argument('--feats_dir', type=str, required=True, help='Directory containing the audio features')
    parser.add_argument('--train_dir', type=str, required=True, help='Directory for training files and model weights')
    parser.add_argument('--train_ckpt_fp', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--context_radius', type=int, default=7, help='Context radius per training example')
    parser.add_argument('--feat_dim', type=int, default=80, help='Number of features per frame')
    parser.add_argument('--nchannels', type=int, default=3, help='Number of channels per frame')
    parser.add_argument('--z_normalize_coeffs', action='store_true', help='Normalize coefficients to zero mean, unit variance per band per channel')
    parser.add_argument('--dense_layer_sizes', type=str, default='256', help='Comma-separated list of dense layer sizes')
    parser.add_argument('--export_feature_layer', type=int, default=0, help='Dense layer to use for feature export')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for exported features')
    return parser.parse_args()

BATCH_SIZE = 512

# Load and process data
def load_data(data_txt_fp):
    with open(data_txt_fp, 'r') as f:
        return f.read().split()

def normalize_data(train_dir, feats):
    with open(os.path.join(train_dir, 'test_mean_std.pkl'), 'rb') as f:
        mean_per_band, std_per_band = pickle.load(f)
    apply_z_norm([(None, feats, None)], mean_per_band, std_per_band)

# Test model
def test(args):
    print('Loading data...')
    pkl_fps = load_data(args.data_txt_fp)

    # Create and load model
    print('Creating and loading model...')
    dense_layer_sizes = list(map(int, args.dense_layer_sizes.split(',')))
    model = OnsetCNN(args.context_radius, args.feat_dim, args.nchannels, dense_layer_sizes, args.export_feature_layer)
    model.load_weights(args.train_ckpt_fp)
    print('Model weights restored from:', args.train_ckpt_fp)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Process song files
    for pkl_fp in tqdm(pkl_fps):
        with open(os.path.join(args.feats_dir, pkl_fp), 'rb') as f:
            song_features = pickle.load(f)

        if args.z_normalize_coeffs:
            print('Normalizing data...')
            normalize_data(args.train_dir, song_features)

        song_context, _ = model.prepare_test(song_features, 0)
        song_export = []

        # Run inference in batches
        for i in range(0, len(song_features), BATCH_SIZE):
            batch_features = song_context[i:i + BATCH_SIZE]
            batch_export = model(batch_features, training=False).numpy()
            song_export.append(batch_export)

        song_export = np.concatenate(song_export)

        # Save output
        out_pkl_fp = os.path.join(args.out_dir, os.path.basename(pkl_fp))
        with open(out_pkl_fp, 'wb') as f:
            pickle.dump(song_export, f)

        print('Saved output to:', out_pkl_fp)

if __name__ == '__main__':
    args = parse_arguments()
    test(args)