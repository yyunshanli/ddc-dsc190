import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class SymNet(keras.Model):
    def __init__(self, config):
        super(SymNet, self).__init__()
        self.config = config

        # Symbolic feature embedding
        if config['sym_embedding_size'] > 0:
            self.embedding_layer = layers.Embedding(input_dim=config['in_len'], output_dim=config['sym_embedding_size'])
        
        # CNN Layers
        self.cnn_layers = []
        if config['do_cnn']:
            for i, (filter_shape, pool_shape) in enumerate(zip(config['cnn_filter_shapes'], config['cnn_pool'])):
                self.cnn_layers.append(
                    keras.Sequential([
                        layers.Conv2D(filters=filter_shape[2], kernel_size=filter_shape[:2], padding='valid', activation='relu'),
                        layers.MaxPooling2D(pool_size=pool_shape, padding='same')
                    ])
                )
        
        # RNN Layers
        if config['do_rnn']:
            rnn_cells = {
                'rnn': layers.SimpleRNNCell,
                'gru': layers.GRUCell,
                'lstm': layers.LSTMCell
            }
            self.rnn_layers = [
                layers.RNN(rnn_cells[config['rnn_cell_type']](config['rnn_size']), return_sequences=True)
                for _ in range(config['rnn_nlayers'])
            ]
        
        # Dense Layers
        self.dnn_layers = []
        if config['do_dnn']:
            for size in config['dnn_sizes']:
                self.dnn_layers.append(layers.Dense(size, activation='sigmoid'))

        # Final output layer
        self.output_layer = layers.Dense(config['out_len'])

    def call(self, inputs, training=False):
        syms, feats_other, feats_audio = inputs
        
        # Embed symbolic features if embedding size is specified
        if hasattr(self, 'embedding_layer'):
            syms = self.embedding_layer(syms)
        
        # Apply CNN to audio features
        if hasattr(self, 'cnn_layers') and self.cnn_layers:
            for cnn_layer in self.cnn_layers:
                feats_audio = cnn_layer(feats_audio)
            feats_audio = layers.Flatten()(feats_audio)

        # Concatenate features
        combined_features = tf.concat([syms, feats_other, feats_audio], axis=-1)

        # Apply RNN layers
        if hasattr(self, 'rnn_layers') and self.rnn_layers:
            for rnn_layer in self.rnn_layers:
                combined_features = rnn_layer(combined_features)

        # Apply DNN layers
        for dnn_layer in self.dnn_layers:
            combined_features = dnn_layer(combined_features)

        # Output layer
        output = self.output_layer(combined_features)
        return output