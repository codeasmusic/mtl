
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import LSTM, Conv1D, Dropout, concatenate


class RCNN:
    def __init__(self, params, flag):
        self.params = params
        self.forward_lstm = LSTM(units=params['lstm_conv_dim'], return_sequences=True,
                                 dropout=params['lstm_dropout'], recurrent_dropout=params['lstm_dropout'],
                                 name='fw_lstm_'+flag)

        self.backward_lstm = LSTM(units=params['lstm_conv_dim'], return_sequences=True,
                                  dropout=params['lstm_dropout'], recurrent_dropout=params['lstm_dropout'],
                                  name='bw_lstm_'+flag, go_backwards=True)

        self.conv = Conv1D(filters=params['lstm_conv_dim'], kernel_size=params['kernel_size'],
                           padding='same', activation='relu', strides=1, name='conv_'+flag)

        self.conv_dropout = Dropout(params['conv_dropout'], name='conv_dropout_'+flag)


    def __call__(self, emb_layer):
        fw_lstm_out = self.forward_lstm(emb_layer)
        bw_lstm_out = self.backward_lstm(emb_layer)
        conv_out = self.conv_dropout(self.conv(emb_layer))

        return concatenate([fw_lstm_out, conv_out, bw_lstm_out], axis=2)


class Attention(Layer):
    def __init__(self, att_dim, to_compose=False, **kwargs):
        self.att_dim = att_dim
        self.to_compose = to_compose
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.pre_weight = self.add_weight(name='pre_weight', shape=(input_shape[-1], self.att_dim),
                                          initializer='glorot_uniform', trainable=True)
        self.pre_bias = self.add_weight(name='pre_weight', shape=(self.att_dim,),
                                        initializer='zeros', trainable=True)
        self.u_w = self.add_weight(name='att_weight', shape=(self.att_dim, 1),
                                   initializer='glorot_uniform', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        u_i = K.tanh(K.dot(x, self.pre_weight) + self.pre_bias)
        ai = K.exp(K.dot(u_i, self.u_w))

        ai_sum = K.expand_dims(K.sum(ai, axis=1), axis=2)
        weights = ai / ai_sum

        att_output = x * weights
        if self.to_compose:
            att_output = K.sum(att_output, axis=1)
        return att_output


    def compute_output_shape(self, input_shape):
        if self.to_compose:
            return input_shape[0], input_shape[-1]
        return input_shape


class LstmConvWeightedMerge(Layer):
    def __init__(self, to_compose=False, init='glorot_uniform', **kwargs):
        self.init = init
        self.to_compose = to_compose
        super(LstmConvWeightedMerge, self).__init__(**kwargs)

    def build(self, input_shape):
        self.fw_weight = self.add_weight(name='fw_weight', shape=(1,),
                                         initializer=self.init, trainable=True)
        self.conv_weight = self.add_weight(name='conv_weight', shape=(1,),
                                         initializer=self.init, trainable=True)
        self.bw_weight = self.add_weight(name='bw_weight', shape=(1,),
                                         initializer=self.init, trainable=True)
        self.merge_bias = self.add_weight(name='merge_bias', shape=(1,),
                                         initializer='zeros', trainable=True)
        super(LstmConvWeightedMerge, self).build(input_shape)

    def call(self, x, **kwargs):
        assert len(x.shape) == 3
        part_dims = int(x.shape[-1] / 3)

        merge_output = x[:, :, 0: part_dims] * self.fw_weight \
                       + x[:, :, part_dims: 2*part_dims] * self.conv_weight \
                       + x[:, :, 2*part_dims: 3*part_dims] * self.bw_weight \
                       + self.merge_bias

        if self.to_compose:
            merge_output = K.sum(merge_output, axis=1)

        return merge_output

    def compute_output_shape(self, input_shape):
        if self.to_compose:
            return input_shape[0], input_shape[2]/3
        return input_shape[0], input_shape[1], input_shape[2]/3



class WeightedLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightedLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weight = self.add_weight(name='global_weight', shape=(1,),
                                         initializer='ones', trainable=True)
        super(WeightedLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        #assert len(x.shape) == 3
        return x * self.weight

    def compute_output_shape(self, input_shape):
        return input_shape
