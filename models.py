import numpy as np
import tensorflow as tf


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

def point_wise_feed_forward_network(
  d_model,
  dff
  ):

  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # Shape `(batch_size, seq_len, dff)`.
      tf.keras.layers.Dense(d_model)  # Shape `(batch_size, seq_len, d_model)`.
  ])

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*,
               d_model,
               num_attention_heads,
               dff,
               dropout_rate=0.1
               ):
    super(EncoderLayer, self).__init__()



    self.mha = tf.keras.layers.MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=d_model,
        dropout=dropout_rate,
        )

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x, training, mask):

    if mask is not None:
      mask1 = mask[:, :, None]
      mask2 = mask[:, None, :]
      attention_mask = mask1 & mask2
    else:
      attention_mask = None

    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        attention_mask=attention_mask,
        training=training,
        )

    out1 = self.layernorm1(x + attn_output)

    ffn_output = self.ffn(out1)
    ffn_output = self.dropout1(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)

    return out2

class Encoder(tf.keras.layers.Layer):
  def __init__(self,
               *,
               num_layers,
               d_model,
               num_attention_heads,
               dff,
               max_len,
               input_vocab_size,
               dropout_rate=0.1,
               ):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, mask_zero=True)
    self.pos_encoding = positional_encoding(max_len, self.d_model)

    self.enc_layers = [
        EncoderLayer(
          d_model=d_model,
          num_attention_heads=num_attention_heads,
          dff=dff,
          dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  # Masking.
  def compute_mask(self, x, previous_mask=None):
    return self.embedding.compute_mask(x, previous_mask)

  def call(self, x, training):

    seq_len = tf.shape(x)[1]

    mask = self.compute_mask(x)
    x = self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    x = self.dropout(x, training=training)


    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training=training, mask=mask)

    return x


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_attention_heads,
               dff,
               dropout_rate=0.1
               ):
    super(DecoderLayer, self).__init__()

    self.mha_masked = tf.keras.layers.MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=d_model,
        dropout=dropout_rate
    )

    self.mha_cross = tf.keras.layers.MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=d_model,
        dropout=dropout_rate
    )

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x, mask, enc_output, enc_mask, training):

    self_attention_mask = None
    if mask is not None:
      mask1 = mask[:, :, None]
      mask2 = mask[:, None, :]
      self_attention_mask = mask1 & mask2

    attn_masked, attn_weights_masked = self.mha_masked(
        query=x,
        value=x,
        key=x,
        attention_mask=self_attention_mask,
        use_causal_mask=True,
        return_attention_scores=True,
        training=training
        )

    out1 = self.layernorm1(attn_masked + x)

    attention_mask = None
    if mask is not None and enc_mask is not None:
      mask1 = mask[:, :, None]
      mask2 = enc_mask[:, None, :]
      attention_mask = mask1 & mask2

    attn_cross, attn_weights_cross = self.mha_cross(
        query=out1,
        value=enc_output,
        key=enc_output,
        attention_mask=attention_mask,
        return_attention_scores=True,
        training=training
    )

    out2 = self.layernorm2(attn_cross + out1)

    ffn_output = self.ffn(out2)
    ffn_output = self.dropout1(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)

    return out3, attn_weights_masked, attn_weights_cross

class Decoder(tf.keras.layers.Layer):
  def __init__(self,
               *,
               num_layers,
               d_model,
               num_attention_heads,
               dff,
               max_len,
               target_vocab_size,
               dropout_rate=0.1
               ):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(
      target_vocab_size,
      d_model,
      mask_zero=True
      )
    self.pos_encoding = positional_encoding(max_len, d_model)

    self.dec_layers = [
        DecoderLayer(
          d_model=d_model,
          num_attention_heads=num_attention_heads,
          dff=dff,
          dropout_rate=dropout_rate)
        for _ in range(num_layers)
        ]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x, enc_output, enc_mask, training):

    seq_len = tf.shape(x)[1]
    attention_weights = {}

    mask = self.embedding.compute_mask(x)
    x = self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2  = self.dec_layers[i](x, mask=mask, enc_output=enc_output, enc_mask=enc_mask, training=training)

      attention_weights[f'decoder_layer{i+1}_block1'] = block1
      attention_weights[f'decoder_layer{i+1}_block2'] = block2

    return x, attention_weights


class Transformer(tf.keras.Model):
  def __init__(self,
               *,
               num_layers,
               d_model,
               num_attention_heads,
               dff,
               max_len,
               input_vocab_size,
               target_vocab_size,
               dropout_rate=0.1
               ):
    super().__init__()

    self.encoder = Encoder(
      num_layers=num_layers,
      d_model=d_model,
      num_attention_heads=num_attention_heads,
      dff=dff,
      input_vocab_size=input_vocab_size,
      dropout_rate=dropout_rate,
      max_len=max_len
      )

    self.decoder = Decoder(
      num_layers=num_layers,
      d_model=d_model,
      num_attention_heads=num_attention_heads,
      dff=dff,
      max_len=max_len,
      target_vocab_size=target_vocab_size,
      dropout_rate=dropout_rate
      )

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs, training=False):
    inp, tar = inputs

    enc_output = self.encoder(inp, training=training)
    enc_mask = self.encoder.compute_mask(inp)

    dec_output, attention_weights = self.decoder(
        tar, enc_output, enc_mask, training=training)

    final_output = self.final_layer(dec_output)

    return final_output

  class TransformerProbs(tf.keras.Model):
      def __init__(self,
                   *,
                   num_layers,
                   d_model,
                   num_attention_heads,
                   dff,
                   max_len,
                   first_vocab_size,
                   second_vocab_size,
                   gru_hidden=64,
                   dropout_rate=0.1
                   ):
          super().__init__()

          self.encoder = Encoder(
              num_layers=num_layers,
              d_model=d_model,
              num_attention_heads=num_attention_heads,
              dff=dff,
              input_vocab_size=first_vocab_size,
              dropout_rate=dropout_rate,
              max_len=max_len
          )

          self.decoder = Decoder(
              num_layers=num_layers,
              d_model=d_model,
              num_attention_heads=num_attention_heads,
              dff=dff,
              max_len=max_len,
              target_vocab_size=second_vocab_size,
              dropout_rate=dropout_rate
          )

          self.gru = tf.keras.layers.GRU(gru_hidden)
          self.prob_layer = tf.keras.layers.Dense(1)

      def call(self, inputs, training=False):
          inp, tar = inputs

          enc_output = self.encoder(inp, training=training)
          enc_mask = self.encoder.compute_mask(inp)

          dec_output, attention_weights = self.decoder(
              tar, enc_output, enc_mask, training=training)

          final_emb = self.gru_layer(dec_output)
          prob = self.prob_layer(final_emb)


          return prob