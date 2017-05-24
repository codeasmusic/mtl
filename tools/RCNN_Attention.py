from keras.layers import LSTM, Conv1D, Dropout, TimeDistributed, Dense
from keras.layers import Reshape, concatenate

from keras.engine.topology import Layer
from keras import backend as K


class RCNN:
    def __init__(self, params):
        self.forward_lstm = LSTM(units=params['lstm_output_dim'], return_sequences=True,
                                 dropout=params['lstm_dropout'], recurrent_dropout=params['lstm_dropout'],
                                 name='forward_lstm')

        self.backward_lstm = LSTM(units=params['lstm_output_dim'], return_sequences=True,
                                  dropout=params['lstm_dropout'], recurrent_dropout=params['lstm_dropout'],
                                  name='backward_lstm', go_backwards=True)

        self.conv = Conv1D(filters=params['filters'], kernel_size=params['kernel_size'],
                           padding='same', activation='relu', strides=1, name='conv')

        self.conv_dropout = Dropout(params['conv_dropout'], name='conv_dropout')


    def handle(self, params, emb_layer, task_index):
        fw_lstm_out = self.forward_lstm(emb_layer)
        bw_lstm_out = self.backward_lstm(emb_layer)
        conv_out = self.conv_dropout(self.conv(emb_layer))

        # fw_lstm_out = TimeDistributed(Dense(params['lstm_output_dim']))(fw_lstm_out)
        fw_lstm_attr = AttLayer()(fw_lstm_out)
        fw_lstm_attr = Reshape((params['lstm_output_dim'], 1))(fw_lstm_attr)

        # conv_out = TimeDistributed(Dense(params['filters']))(conv_out)
        conv_attr = AttLayer()(conv_out)
        conv_attr = Reshape((params['filters'], 1))(conv_attr)

        # bw_lstm_out = TimeDistributed(Dense(params['lstm_output_dim']))(bw_lstm_out)
        bw_lstm_attr = AttLayer()(bw_lstm_out)
        bw_lstm_attr = Reshape((params['lstm_output_dim'], 1))(bw_lstm_attr)

        rcnn_out = concatenate([fw_lstm_attr, conv_attr, bw_lstm_attr], axis=2,
                               name='rcnn_out_'+str(task_index))
        return rcnn_out


class AttLayer(Layer):
    def __init__(self, **kwargs):
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.kernel = self.add_weight(shape=(input_shape[-1], 1),
                                      initializer='uniform',
                                      trainable=True)
        super(AttLayer, self).build(input_shape)

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.kernel))
        ai = K.exp(eij)

        ai_sum = K.expand_dims(K.sum(ai, axis=1), axis=2)
        weights = ai / ai_sum

        weighted_input = x * weights
        return K.sum(weighted_input, axis=1)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

