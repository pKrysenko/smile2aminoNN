import numpy as np
import tensorflow as tf

from tqdm import tqdm


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
        for i in tqdm(tf.range(max_length)):
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


class TransformerPredictorProb(tf.Module):
    def __init__(self, sp_first, sp_second, transformer, scaler_dict):
        super().__init__()
        self.sp_first = sp_first
        self.sp_second = sp_second
        self.transformer = transformer
        self.scaler_dict = scaler_dict

    def __call__(self, first, second, max_length=0):

        first_chem = np.expand_dims(np.array([self.sp_first.bos_id()] + \
                                           self.sp_first.encode(first, out_type=int) + [self.sp_first.eos_id()],
                                           dtype=np.float32), axis=0)
        second_chem = np.expand_dims(np.array([self.sp_second.bos_id()] + \
                                           self.sp_second.encode(second, out_type=int) + [self.sp_second.eos_id()],
                                           dtype=np.float32), axis=0)

        print(first_chem.shape)
        print(second_chem.shape)
        first_chem = tf.keras.utils.pad_sequences(first_chem, maxlen=max_length,
                                               padding='post', value=0)
        second_chem = tf.keras.utils.pad_sequences(second_chem, maxlen=max_length,
                                                padding='post', value=0)
        score = self.transformer.call((first_chem, second_chem), training=False)

        score = score * (self.scaler_dict["max"] - self.scaler_dict["min"]) + self.scaler_dict["min"]
        return score