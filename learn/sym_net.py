import math
import random
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class SymNet(Model):
    def __init__(self,
                 batch_size,
                 nunroll,
                 sym_in_type,
                 sym_embedding_size,
                 sym_out_type,
                 sym_narrows,
                 sym_narrowclasses,
                 other_nfeats,
                 audio_context_radius,
                 audio_nbands,
                 audio_nchannels,
                 cnn_filter_shapes,
                 cnn_pool,
                 cnn_dim_reduction_size,
                 rnn_cell_type,
                 rnn_size,
                 rnn_nlayers,
                 dnn_sizes,
                 dropout_rate=0.0):
        super(SymNet, self).__init__()

        self.batch_size = batch_size
        self.nunroll = nunroll
        self.sym_in_type = sym_in_type
        self.sym_embedding_size = sym_embedding_size
        self.sym_out_type = sym_out_type
        self.sym_narrows = sym_narrows
        self.sym_narrowclasses = sym_narrowclasses
        self.other_nfeats = other_nfeats
        self.audio_context_radius = audio_context_radius
        self.audio_nbands = audio_nbands
        self.audio_nchannels = audio_nchannels
        self.cnn_filter_shapes = cnn_filter_shapes
        self.cnn_pool = cnn_pool
        self.cnn_dim_reduction_size = cnn_dim_reduction_size
        self.rnn_cell_type = rnn_cell_type
        self.rnn_size = rnn_size
        self.rnn_nlayers = rnn_nlayers
        self.dnn_sizes = dnn_sizes
        self.dropout_rate = dropout_rate

        # Symbolic input embedding
        if sym_embedding_size > 0:
            self.embedding = layers.Embedding(input_dim=int(math.pow(sym_narrowclasses, sym_narrows)),
                                              output_dim=sym_embedding_size)
        else:
            self.embedding = None

        # CNN layers
        self.cnn_layers = []
        input_channels = audio_nchannels
        for i, ((ntime, nband, nfilt), (ptime, pband)) in enumerate(zip(cnn_filter_shapes, cnn_pool)):
            conv_layer = layers.Conv2D(filters=nfilt, kernel_size=(ntime, nband), padding='valid', activation='relu')
            pool_layer = layers.MaxPooling2D(pool_size=(ptime, pband), padding='same')
            self.cnn_layers.append((conv_layer, pool_layer))
            input_channels = nfilt

        # Dimensionality reduction layer
        if cnn_dim_reduction_size >= 0:
            self.dim_reduction = layers.Dense(cnn_dim_reduction_size, activation='relu')
        else:
            self.dim_reduction = None

        # RNN layers
        self.rnn_cells = []
        for _ in range(rnn_nlayers):
            if rnn_cell_type == 'lstm':
                self.rnn_cells.append(layers.LSTMCell(rnn_size))
            elif rnn_cell_type == 'gru':
                self.rnn_cells.append(layers.GRUCell(rnn_size))
            else:
                self.rnn_cells.append(layers.SimpleRNNCell(rnn_size))
        self.rnn_layer = layers.RNN(self.rnn_cells, return_sequences=True, return_state=True)

        # DNN layers
        self.dnn_layers = []
        for size in dnn_sizes:
            self.dnn_layers.append(layers.Dense(size, activation='relu'))
            self.dnn_layers.append(layers.Dropout(dropout_rate))

        # Output layer
        if sym_out_type == 'onehot':
            self.output_layer = layers.Dense(int(math.pow(sym_narrowclasses, sym_narrows)), activation='softmax')
        else:
            raise NotImplementedError("Currently only 'onehot' sym_out_type is implemented")

    def call(self, inputs, training=False):
        syms_input, feats_audio, feats_other = inputs
        batch_size = tf.shape(syms_input)[0]

        # Symbolic input embedding
        if self.embedding:
            sym_embedded = self.embedding(syms_input)
        else:
            sym_embedded = tf.one_hot(syms_input, depth=int(math.pow(self.sym_narrowclasses, self.sym_narrows)))

        # CNN feature extraction
        x = feats_audio
        for conv_layer, pool_layer in self.cnn_layers:
            x = conv_layer(x)
            x = pool_layer(x)
        cnn_features = layers.Flatten()(x)

        # Dimensionality reduction
        if self.dim_reduction:
            cnn_features = self.dim_reduction(cnn_features)

        # Combine features for RNN input
        combined_features = tf.concat([sym_embedded, cnn_features, feats_other], axis=-1)

        # RNN processing
        rnn_outputs, *rnn_states = self.rnn_layer(combined_features)

        # DNN layers
        x = rnn_outputs
        for layer in self.dnn_layers:
            x = layer(x, training=training)

        # Final output
        outputs = self.output_layer(x)
        return outputs