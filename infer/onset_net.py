import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import random
from functools import reduce

class OnsetNet(Model):
    def __init__(self,
                 batch_size,
                 audio_context_radius,
                 audio_nbands,
                 audio_nchannels,
                 nfeats,
                 cnn_filter_shapes,
                 cnn_pool,
                 rnn_cell_type,
                 rnn_size,
                 rnn_nlayers,
                 rnn_nunroll,
                 rnn_keep_prob,
                 dnn_sizes,
                 dnn_keep_prob,
                 dnn_nonlin,
                 target_weight_strategy='rect',
                 grad_clip=0.0,
                 opt='sgd',
                 dtype=tf.float32):
        
        super(OnsetNet, self).__init__()
        self.batch_size = batch_size
        self.rnn_nunroll = rnn_nunroll
        self.target_weight_strategy = target_weight_strategy

        # CNN layers
        self.cnn_layers = []
        for i, ((ntime, nband, nfilt), (ptime, pband)) in enumerate(zip(cnn_filter_shapes, cnn_pool)):
            self.cnn_layers.append(layers.Conv2D(filters=nfilt, kernel_size=(ntime, nband), padding='same', activation='relu'))
            self.cnn_layers.append(layers.MaxPooling2D(pool_size=(ptime, pband)))

        # RNN layers
        self.rnn_layers = []
        for _ in range(rnn_nlayers):
            if rnn_cell_type == 'lstm':
                self.rnn_layers.append(layers.LSTM(rnn_size, return_sequences=True, dropout=1 - rnn_keep_prob))
            elif rnn_cell_type == 'gru':
                self.rnn_layers.append(layers.GRU(rnn_size, return_sequences=True, dropout=1 - rnn_keep_prob))
            else:
                self.rnn_layers.append(layers.SimpleRNN(rnn_size, return_sequences=True, dropout=1 - rnn_keep_prob))

        # DNN layers
        self.dnn_layers = []
        for size in dnn_sizes:
            self.dnn_layers.append(layers.Dense(size, activation=dnn_nonlin))
            if dnn_keep_prob < 1.0:
                self.dnn_layers.append(layers.Dropout(1 - dnn_keep_prob))

        # Output layer
        self.logit_layer = layers.Dense(1, activation=None)

    def call(self, inputs, training=False):
        feats_audio, feats_other = inputs
        
        # Pass through CNN layers
        x = feats_audio
        for layer in self.cnn_layers:
            x = layer(x)

        # Flatten and concatenate with other features
        x = layers.Flatten()(x)
        x = layers.concatenate([x, feats_other], axis=-1)

        # Pass through RNN layers if defined
        if self.rnn_layers:
            x = tf.reshape(x, [self.batch_size, self.rnn_nunroll, -1])  # Reshape for RNN input
            for layer in self.rnn_layers:
                x = layer(x, training=training)
            x = tf.reshape(x, [self.batch_size * self.rnn_nunroll, -1])

        # Pass through DNN layers
        for layer in self.dnn_layers:
            x = layer(x, training=training)

        # Output logits and prediction
        logits = self.logit_layer(x)
        prediction = tf.nn.sigmoid(logits)

        return logits, prediction