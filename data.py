import os
import numpy as np
import sentencepiece as spm
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split

def generate_spm_model_name(vocab_size, column):
    return f"{column.lower()}_v{vocab_size}"

class DataLoader:
    def __init__(self, csv_path, vocab_size, max_len, input_column, output_column):
        df = pd.read_csv(csv_path)
        self.input_data = list(df[input_column])
        self.output_data = list(df[output_column])
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.sp_input, self.sp_output = self.prepare_spm(csv_path, input_column, output_column)

    def prepare_spm(self, csv_path, inp_column, out_column):


        os.system(f"python3 prepare_spm_models.py {csv_path} {self.vocab_size} {inp_column} {out_column}")

        sp_input_model_name = generate_spm_model_name(self.vocab_size, inp_column) + ".model"
        sp_out_model_name = generate_spm_model_name(self.vocab_size, out_column) + ".model"

        sp_input = spm.SentencePieceProcessor(model_file=sp_input_model_name)
        sp_output = spm.SentencePieceProcessor(model_file=sp_out_model_name)

        return sp_input, sp_output

    def prepare_batch(self, input_, output_):
        encoded_input = np.array([self.sp_input.bos_id()] + \
                        self.sp_input.encode(input_, out_type=int) + \
                        [self.sp_input.eos_id()], dtype=np.float32)

        encoded_output = np.array([self.sp_input.bos_id()] + \
                         self.sp_output.encode(output_, out_type=int) + \
                         [self.sp_input.eos_id()], dtype=np.float32)

        en_inputs = encoded_output[:-1]
        en_labels = encoded_output[1:]

        return (encoded_input, en_inputs), en_labels


    def pipeline(self):
        encoded_inputs, o_inputs, outputs = [], [], []

        for inp, out in zip(self.input_data, self.output_data):
            (encoded_input, en_inputs), output = self.prepare_batch(inp, out)
            encoded_inputs.append(encoded_input)
            o_inputs.append(en_inputs)
            outputs.append(output)


        encoded_inputs = tf.keras.utils.pad_sequences(encoded_inputs, maxlen=self.max_len,
                                     padding='post', value=0)
        o_inputs = tf.keras.utils.pad_sequences(o_inputs, maxlen=self.max_len,
                                                      padding='post', value=0)
        outputs = tf.keras.utils.pad_sequences(outputs, maxlen=self.max_len,
                                     padding='post', value=0)

        return encoded_inputs, o_inputs, outputs



