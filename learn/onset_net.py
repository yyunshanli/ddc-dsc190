import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Sequential, layers

class OnsetNet(Model):
    def __init__(self,
                 mode,
                 batch_size,
                 audio_context_radius,
                 audio_nbands,
                 audio_nchannels,
                 nfeats,
                 cnn_filter_shapes,
                 cnn_init,
                 cnn_pool,
                 rnn_cell_type,
                 rnn_size,
                 rnn_nlayers,
                 rnn_keep_prob,
                 dnn_sizes,
                 dnn_nonlin,
                 target_weight_strategy,  # 'rect', 'last', 'pos', 'seq'
                 grad_clip,
                 opt,
                 export_feat_name=None):
        super(OnsetNet, self).__init__()
        self.mode = mode
        self.batch_size = batch_size
        self.target_weight_strategy = target_weight_strategy
        self.do_rnn = rnn_size > 0 and rnn_nlayers > 0

        # CNN layers
        self.cnn_layers = []
        for i, ((ntime, nband, nfilt), (ptime, pband)) in enumerate(zip(cnn_filter_shapes, cnn_pool)):
            self.cnn_layers.append(
                Sequential([
                    layers.Conv2D(filters=nfilt, kernel_size=(ntime, nband), padding='valid',
                                  activation='relu', kernel_initializer=cnn_init),
                    layers.MaxPooling2D(pool_size=(ptime, pband), strides=(ptime, pband), padding='same')
                ])
            )

        # RNN layers
        if self.do_rnn:
            rnn_cells = {
                'rnn': lambda: layers.SimpleRNNCell(rnn_size),
                'gru': lambda: layers.GRUCell(rnn_size),
                'lstm': lambda: layers.LSTMCell(rnn_size)
            }
            rnn_cell = rnn_cells.get(rnn_cell_type, lambda: None)()
            if rnn_keep_prob < 1.0:
                rnn_cell = layers.Dropout(rnn_keep_prob)(rnn_cell)
            self.rnn_layers = [rnn_cell] * rnn_nlayers

        # DNN layers
        self.dnn_layers = []
        for i, size in enumerate(dnn_sizes):
            self.dnn_layers.append(layers.Dense(size, activation=dnn_nonlin))

        # Final output layer
        self.logit_layer = layers.Dense(1, activation=None)

    def call(self, inputs, training=False):
        feats_audio, feats_other = inputs
        x = feats_audio

        # CNN forward pass
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)

        # Flatten and concatenate
        x = layers.Flatten()(x)
        x = tf.concat([x, feats_other], axis=1)

        # RNN forward pass
        if self.do_rnn:
            x = layers.RNN(self.rnn_layers, return_sequences=True)(x)

        # DNN forward pass
        for dnn_layer in self.dnn_layers:
            x = dnn_layer(x)
            if training:
                x = layers.Dropout(1 - self.rnn_keep_prob)(x)

        # Output layer
        logits = self.logit_layer(x)
        return tf.nn.sigmoid(logits)