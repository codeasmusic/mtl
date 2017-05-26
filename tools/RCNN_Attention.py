from keras.layers import LSTM, Conv1D, Dropout, TimeDistributed, Dense
from keras.layers import Reshape, concatenate

from keras.engine.topology import Layer
from keras import backend as K


class RCNN:
    def __init__(self, params):
        self.params = params
        self.forward_lstm = LSTM(units=params['lstm_output_dim'], return_sequences=True,
                                 dropout=params['lstm_dropout'], recurrent_dropout=params['lstm_dropout'])

        self.backward_lstm = LSTM(units=params['lstm_output_dim'], return_sequences=True,
                                  dropout=params['lstm_dropout'], recurrent_dropout=params['lstm_dropout'],
                                  go_backwards=True)

        self.conv = Conv1D(filters=params['filters'], kernel_size=params['kernel_size'],
                           padding='same', activation='relu', strides=1)

        self.conv_dropout = Dropout(params['conv_dropout'])


    def global_handle(self, emb_layer, flag):

        fw_lstm_out = self.forward_lstm(emb_layer)
        bw_lstm_out = self.backward_lstm(emb_layer)
        conv_out = self.conv_dropout(self.conv(emb_layer))

        fw_lstm_out = TimeDistributed(Dense(self.params['attention_dim']), name='fw_tb_'+flag)(fw_lstm_out)
        fw_lstm_att = Attention()(fw_lstm_out)
        # fw_lstm_att = Reshape((self.params['lstm_output_dim'], 1))(fw_lstm_att)

        conv_out = TimeDistributed(Dense(self.params['attention_dim']), name='conv_tb_'+flag)(conv_out)
        conv_att = Attention()(conv_out)
        # conv_att = Reshape((self.params['filters'], 1))(conv_att)

        bw_lstm_out = TimeDistributed(Dense(self.params['attention_dim']), name='bw_tb_'+flag)(bw_lstm_out)
        bw_lstm_att = Attention()(bw_lstm_out)
        # bw_lstm_att = Reshape((self.params['lstm_output_dim'], 1))(bw_lstm_att)

        return concatenate([fw_lstm_att, conv_att, bw_lstm_att], axis=2)


    def local_handle(self, emb_layer):
        fw_lstm_out = self.forward_lstm(emb_layer)
        bw_lstm_out = self.backward_lstm(emb_layer)
        conv_out = self.conv_dropout(self.conv(emb_layer))

        return concatenate([fw_lstm_out, conv_out, bw_lstm_out], axis=2)


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.u_w = self.add_weight(shape=(input_shape[-1], 1),
                                   initializer='uniform', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        u_i = K.tanh(x)
        ai = K.exp(K.dot(u_i, self.u_w))

        ai_sum = K.expand_dims(K.sum(ai, axis=1), axis=2)
        weights = ai / ai_sum

        weighted_input = x * weights
        return weighted_input


    def compute_output_shape(self, input_shape):
        return input_shape

