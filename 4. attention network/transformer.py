import tensorflow as tf
import keras
from keras.layers import Layer, Dense, Dropout, LayerNormalization


# ==================== POSITIONAL ENCODING ==================== #
class PositionalEncoding(Layer):
    def __init__(self, seq_len, num_feat):
        super().__init__()
        positions = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        dims = tf.range(num_feat, dtype=tf.float32)[tf.newaxis, :]

        angle_rates = 1 / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(num_feat, tf.float32))
        angle_rads = positions * angle_rates

        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        self.pos_encoding = tf.constant(pos_encoding[tf.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding


# ==================== MULTI-HEAD ATTENTION ==================== #
class MultiHeadAttention(Layer):
    def __init__(self, num_feat, num_head):
        super().__init__()
        assert num_feat % num_head == 0

        self.num_head = num_head
        self.depth = num_feat // num_head

        self.wq = Dense(num_feat)
        self.wk = Dense(num_feat)
        self.wv = Dense(num_feat)

        self.dense = Dense(num_feat)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_head, self.depth))
        return tf.transpose(x, perm=(0, 2, 1, 3))  # (batch, heads, seq_len, depth)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            logits += (mask * -1e9)
        weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(weights, v)
        return output, weights

    def call(self, q, k, v, mask= None):
        batch_size = tf.shape(q)[0]
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        attn_output, _ = self.scaled_dot_product_attention(q, k, v, mask)
        attn_output = tf.transpose(attn_output, perm=(0, 2, 1, 3))
        concat = tf.reshape(attn_output, (batch_size, -1, self.num_head * self.depth))
        output = self.dense(concat)
        return output

# ==================== FEED-FORWARD NETWORK ==================== #
class FeedForward(Layer):
    def __init__(self, num_feat, num_ff):
        super().__init__()
        self.dense1 = Dense(num_ff, activation='relu')
        self.dense2 = Dense(num_feat)

    def call(self, x):
        return self.dense2(self.dense1(x))

# ==================== ENCODER LAYER ==================== #
class EncoderLayer(Layer):
    def __init__(self, num_feat, num_head, num_ff, dropout_rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(num_feat, num_head)
        self.ffn = FeedForward(num_feat, num_ff)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x):
        attn_output = self.mha(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x

# ==================== DECODER LAYER ==================== #
class DecoderLayer(Layer):
    def __init__(self, num_feat, num_head, num_ff, dropout_rate=0.1):
        super().__init__()
        self.mha_first = MultiHeadAttention(num_feat, num_head)
        self.mha_second = MultiHeadAttention(num_feat, num_head)
        self.ffn = FeedForward(num_feat, num_ff)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.norm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, x, enc_output):
        seq_len = tf.shape(x)[1]
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = tf.cast(mask, tf.float32)[tf.newaxis, tf.newaxis, ...]


        # 1. Self-attention (masked)
        attn1 = self.mha_first(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn1))
        # 2. Encoder-Decoder attention
        attn2 = self.mha_second(x, enc_output, enc_output)
        x = self.norm2(x + self.dropout2(attn2))
        # 3. FeedForward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        return x

# ==================== ENCODER ==================== #
class Encoder(Layer):
    def __init__(self, num_layer, num_feat, num_head, num_ff, dropout_rate=0.1):
        super().__init__()
        self.num_layer = num_layer
        self.num_feat = num_feat
        self.num_head = num_head
        self.num_ff = num_ff
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.pos_encoding = PositionalEncoding(input_shape[1], input_shape[2])
        self.enc_layers = [EncoderLayer(self.num_feat, self.num_head, self.num_ff, self.dropout_rate) for _ in range(self.num_layer)]

    def call(self, x):
        x = self.pos_encoding(x)
        for layer in self.enc_layers:
            x = layer(x)
        return x

# ==================== DECODER ==================== #
class Decoder(Layer):
    def __init__(self, num_layer, num_feat, num_head, num_ff, dropout_rate=0.1):
        super().__init__()
        self.num_layer = num_layer
        self.num_feat = num_feat
        self.num_head = num_head
        self.num_ff = num_ff
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.pos_encoding = PositionalEncoding(input_shape[1], input_shape[2])
        self.dec_layers = [DecoderLayer(self.num_feat, self.num_head, self.num_ff, self.dropout_rate) for _ in range(self.num_layer)]

    def call(self, x, enc_output):
        x = self.pos_encoding(x)
        for layer in self.dec_layers:
            x = layer(x, enc_output)
        return x

# ==================== TRANSFORMER ==================== #
class Transformer(keras.Model):
    def __init__(self, num_layer, seq_len_enc, seq_len_dec, num_feat, num_head, num_ff, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layer, seq_len_enc, num_feat, num_head, num_ff, dropout_rate)
        self.decoder = Decoder(num_layer, seq_len_dec, num_feat, num_head, num_ff, dropout_rate)
        self.final_dense = Dense(num_feat)  # dự đoán giá trị số tại mỗi step

    def call(self, enc_input, dec_input, look_ahead_mask=None):
        enc_output = self.encoder(enc_input)
        dec_output = self.decoder(dec_input, enc_output, look_ahead_mask)
        return self.final_dense(dec_output)
