import numpy as np
import tensorflow as tf


class TransformerPredictor(tf.Module):
    def __init__(self, sp_input, sp_output, transformer):
        super().__init__()
        self.sp_input = sp_input
        self.sp_output = sp_output
        self.transformer = transformer

    def __call__(self, sentence, max_length=0):

        sentence = np.expand_dims(np.array([self.sp_input.bos_id()] + \
                                           self.sp_input.encode(sentence, out_type=int) + [self.sp_input.eos_id()],
                                           dtype=np.float32), axis=0)

        start = self.sp_output.bos_id()
        end = self.sp_output.eos_id()

        output_array = [start]
        for i in tf.range(max_length):
            output = np.expand_dims(np.array(output_array), axis=0)
            predictions = self.transformer((sentence, output), training=False)

            predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

            predicted_id = tf.argmax(predictions, axis=-1)

            output_array.append(int(predicted_id[0]))

            if predicted_id == end:
                print("EOS was predicted")
                break

        text = "".join([self.sp_output.decode(o) for o in output_array])
        return text, output_array