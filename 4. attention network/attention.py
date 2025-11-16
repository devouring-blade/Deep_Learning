from keras.layers import Dot, Activation, Concatenate, Reshape
from keras.layers import Dense, GRU, Layer
import tensorflow as tf

class Attention_Layer(Layer):
    def __init__(self, n_hidden):
        super().__init__()
        self.attentionFFN = Dense(n_hidden, activation= "tanh")

    def call(self, e, d):
        dot_product = Dot(axes= (2, 2))([d, e])
        score = Activation("softmax")(dot_product)
        value = Dot(axes= (2, 1))([score, e])
        concat = Concatenate()([value, d])
        h_attention = self.attentionFFN(concat)
        return h_attention

class Encoder(Layer):
    def __init__(self, n_hidden):
        super().__init__()
        self.encoder_GRU = GRU(units= n_hidden, return_sequences= True, return_state= True)

    def call(self, x):
        return self.encoder_GRU(x)

class Decoder(Layer):
    def __init__(self, n_hidden, n_feed):
        super().__init__()
        self.n_hidden = n_hidden
        self.decoder_GRU = GRU(units= n_hidden)
        self.input_feeding_FFN = Dense(units= n_feed, activation= "tanh")
        self.attention = Attention_Layer(n_hidden)

    def call(self, x, o_enc, h_enc):
        outputs = [] # output of decoder (many-to-many)
        i_feed = tf.zeros(shape= (tf.shape(x)[0], self.n_hidden))
        for t in range(x.shape[1]):
            i_cat = self.input_feeding_FFN(i_feed)
            i_cat = Concatenate()([i_cat, x[: , t, : ]])
            i_cat = Reshape([1, -1])(i_cat)
            h_dec = self.decoder_GRU(i_cat, initial_state= h_enc)

            # find attentional hidden state
            h_att = self.attention(o_enc, Reshape([1, -1])(h_dec))

            # update encoder's hidden state and the input-feeding vector for the next step
            h_enc = h_dec
            i_feed = Reshape([-1, ])(h_att)

            # collect outputs at all time steps
            outputs.append(Reshape((self.n_hidden, ))(h_att))

        outputs = tf.convert_to_tensor(outputs)
        outputs = tf.transpose(outputs, perm= [1, 0, 2])
        return outputs











