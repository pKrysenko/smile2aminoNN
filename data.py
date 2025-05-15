import os
import numpy as np
import sentencepiece as spm
import tensorflow as tf
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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

class DataLoaderProbaility:
    def __init__(self, csv_path, vocab_size, max_len_f, max_len_s, first_chemical, second_chemical, output):
        self.df = pd.read_csv(csv_path)
        self.first_chemical = list(self.df[first_chemical])
        self.second_chemical = list(self.df[second_chemical])
        self.probs = list(self.df[output])
        self.vocab_size = vocab_size
        self.max_len_f = max_len_f
        self.max_len_s = max_len_s
        self.sp_first, self.sp_second = self.prepare_spm(csv_path, first_chemical, second_chemical)
        self.first_col_name = first_chemical
        self.second_col_name = second_chemical
        self.out_col_name = output
        print("!!!!!", self.out_col_name)

    def prepare_spm(self, csv_path, inp_column, out_column):

        os.system(f"python3 prepare_spm_models.py {csv_path} {self.vocab_size} {inp_column} {out_column}")

        sp_input_model_name = generate_spm_model_name(self.vocab_size, inp_column) + ".model"
        sp_out_model_name = generate_spm_model_name(self.vocab_size, out_column) + ".model"

        sp_input = spm.SentencePieceProcessor(model_file=sp_input_model_name)
        sp_output = spm.SentencePieceProcessor(model_file=sp_out_model_name)

        return sp_input, sp_output

    def prepare_batch(self, input_, output_, prob):
        encoded_first = np.array([self.sp_first.bos_id()] + \
                        self.sp_first.encode(input_, out_type=int) + \
                        [self.sp_first.eos_id()], dtype=np.float32)

        encoded_second = np.array([self.sp_second.bos_id()] + \
                         self.sp_second.encode(output_, out_type=int) + \
                         [self.sp_second.eos_id()], dtype=np.float32)

        prob = np.float32(prob)



        return (encoded_first, encoded_second), prob

    def proc_data(self, first, second, scores):
        e_first, e_second, score = [], [], []
        for inp, out, prob in tqdm(zip(first, second, scores)):
            
            if len(inp) > self.max_len_f or len(out) > self.max_len_s:
                continue
            prob = 0 if prob == "no" else 1 
            (first_input, second_input), output = self.prepare_batch(inp, out, prob)
            e_first.append(first_input)
            e_second.append(second_input)
            score.append(output)

        return e_first, e_second, score

    def pipeline(self):
        self.df = shuffle(self.df)
        df_val = self.df.sample(frac=0.2)
        df_train = self.df[~self.df.index.isin(df_val.index)]
        
        print("Processing train: ")
        first_train, second_train, score_train = self.proc_data(list(df_train[self.first_col_name]),
                                                           list(df_train[self.second_col_name]),
                                                           list(df_train[self.out_col_name]))

        print("Processing val: ")
        first_val, second_val, score_val = self.proc_data(list(df_val[self.first_col_name]),
                                                          list(df_val[self.second_col_name]),
                                                          list(df_val[self.out_col_name]))


        e_first_train = tf.keras.utils.pad_sequences(first_train, maxlen=self.max_len_f,
                                     padding='post', value=0)
        e_first_val = tf.keras.utils.pad_sequences(first_val, maxlen=self.max_len_f,
                                               padding='post', value=0)
        e_second_train = tf.keras.utils.pad_sequences(second_train, maxlen=self.max_len_s,
                                                      padding='post', value=0)
        e_second_val = tf.keras.utils.pad_sequences(second_val, maxlen=self.max_len_s,
                                                      padding='post', value=0)

        score_train = np.array(score_train)
        score_val = np.array(score_val)

        return (e_first_train, e_second_train, score_train), (e_first_val, e_second_val, score_val), df_train, df_val



