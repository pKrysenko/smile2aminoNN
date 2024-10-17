import sys
import sentencepiece as spm
import pandas as pd

from data import generate_spm_model_name

def iter_pipeline(text):
    for line in text:
        line = line.rstrip()
        yield line

if __name__ == "__main__":
    csv_path, vocab_size, inp_column, out_column = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    df = pd.read_csv(csv_path)
    input_data = list(df[inp_column])
    output_data = list(df[out_column])

    smiles_input_it = iter_pipeline(input_data)
    smiles_output_it = iter_pipeline(output_data)

    inp_model_prefix = generate_spm_model_name(vocab_size, inp_column)
    out_model_prefix = generate_spm_model_name(vocab_size, out_column)

    spm.SentencePieceTrainer.train(sentence_iterator=smiles_input_it, model_prefix=inp_model_prefix,
                                   vocab_size=vocab_size, character_coverage=1.0, model_type='bpe')
    spm.SentencePieceTrainer.train(sentence_iterator=smiles_output_it, model_prefix=out_model_prefix,
                                   vocab_size=vocab_size, character_coverage=1.0, model_type='bpe')