import os
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
from collections import defaultdict
from sym_net import SymNet
from util import open_dataset_fps, apply_z_norm, calc_mean_std_per_band, flatten_dataset_to_charts, load_id_dict

# Argument parser for command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train, evaluate, or generate using SymNet')
    parser.add_argument('--train_txt_fp', type=str, help='Path to training dataset')
    parser.add_argument('--valid_txt_fp', type=str, help='Path to validation dataset')
    parser.add_argument('--test_txt_fp', type=str, help='Path to test dataset')
    parser.add_argument('--model_ckpt_fp', type=str, help='Path to model checkpoint')
    parser.add_argument('--sym_rnn_pretrain_model_ckpt_fp', type=str, help='Path to pre-trained RNN model weights')
    parser.add_argument('--experiment_dir', type=str, required=True, help='Directory for experiment data and checkpoints')
    parser.add_argument('--audio_z_score', action='store_true', help='Z-score normalize audio data')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--nepochs', type=int, default=0, help='Number of training epochs')
    parser.add_argument('--opt', type=str, default='sgd', help='Optimizer type')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate')
    return parser.parse_args()

def load_and_prepare_data(args):
    train_data, valid_data, test_data = open_dataset_fps(args.train_txt_fp, args.valid_txt_fp, args.test_txt_fp)
    if args.audio_z_score and args.valid_txt_fp:
        z_score_fp = os.path.join(args.experiment_dir, 'valid_mean_std.pkl')
        mean_per_band, std_per_band = (calc_mean_std_per_band(valid_data) 
                                       if not os.path.exists(z_score_fp) 
                                       else pickle.load(open(z_score_fp, 'rb')))
        for data in [train_data, valid_data, test_data]:
            apply_z_norm(data, mean_per_band, std_per_band)
    return train_data, valid_data, test_data

def create_model_config(args, nfeats):
    return {
        'nunroll': 1,
        'sym_in_type': 'onehot',
        'sym_embedding_size': 32,
        'sym_out_type': 'onehot',
        'sym_narrows': 4,
        'sym_narrowclasses': 4,
        'other_nfeats': nfeats,
        'audio_context_radius': -1,
        'audio_nbands': 0,
        'audio_nchannels': 0,
        'cnn_filter_shapes': [],
        'cnn_init': tf.keras.initializers.VarianceScaling(scale=1.43, distribution="uniform"),
        'cnn_pool': [],
        'cnn_dim_reduction_size': -1,
        'cnn_dim_reduction_init': tf.keras.initializers.VarianceScaling(scale=1.0, distribution="uniform"),
        'cnn_dim_reduction_nonlin': None,
        'cnn_dim_reduction_keep_prob': 1.0,
        'rnn_proj_init': tf.keras.initializers.Zeros(),
        'rnn_cell_type': 'lstm',
        'rnn_size': 128,
        'rnn_nlayers': 1,
        'rnn_init': tf.random_uniform_initializer(-0.05, 0.05),
        'rnn_keep_prob': 1.0,
        'dnn_sizes': [256],
        'dnn_init': tf.keras.initializers.VarianceScaling(scale=1.15, distribution="uniform"),
        'dnn_keep_prob': 1.0,
        'grad_clip': 0.0,
        'opt': args.opt,
    }

def train_model(sess, model, charts_train, args, summary_writer):
    lr_summary = model.assign_lr(sess, args.lr)
    summary_writer.add_summary(lr_summary, 0)
    
    train_nexamples = sum(chart.get_nannotations() for chart in charts_train)
    examples_per_batch = args.batch_size * model.out_nunroll
    nbatches = args.nepochs * (train_nexamples // examples_per_batch)

    for batch_num in range(nbatches):
        batch_start_time = time.time()
        batch_data = model.prepare_train_batch(charts_train)
        feed_dict = {
            model.syms: batch_data[0],
            model.feats_other: batch_data[1],
            model.feats_audio: batch_data[2],
            model.targets: batch_data[3],
            model.target_weights: batch_data[4]
        }
        sess.run([model.avg_neg_log_lhood, model.train_op], feed_dict=feed_dict)

        if (batch_num + 1) % (train_nexamples // examples_per_batch) == 0:
            print(f'Completed epoch {batch_num // (train_nexamples // examples_per_batch) + 1}')

        if (batch_num + 1) % args.nbatches_per_ckpt == 0:
            save_path = os.path.join(args.experiment_dir, f'model_epoch_{batch_num // (train_nexamples // examples_per_batch) + 1}.ckpt')
            model.save_weights(save_path)
            print(f'Model checkpoint saved at {save_path}')

def main():
    args = parse_arguments()
    train_data, valid_data, test_data = load_and_prepare_data(args)
    charts_train = flatten_dataset_to_charts(train_data)
    model_config = create_model_config(args, nfeats=10)

    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        if args.train_txt_fp:
            model_train = SymNet(mode='train', batch_size=args.batch_size, **model_config)
            sess.run(tf.compat.v1.global_variables_initializer())
            summary_writer = tf.compat.v1.summary.FileWriter(args.experiment_dir, sess.graph)
            train_model(sess, model_train, charts_train, args, summary_writer)

if __name__ == '__main__':
    main()
