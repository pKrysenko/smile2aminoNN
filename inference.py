import sys

import tensorflow as tf
from omegaconf import OmegaConf
import sentencepiece as spm

from models import Transformer
from predictor import TransformerPredictor


if __name__ == "__main__":
    cfg = OmegaConf.load(sys.argv[1])
    sp_input_filepath = sys.argv[2]
    sp_out_filepath = sys.argv[3]
    checkpoint_filepath = sys.argv[4]

    sp_input = spm.SentencePieceProcessor(model_file=sp_input_filepath)
    sp_output = spm.SentencePieceProcessor(model_file=sp_out_filepath)

    model = Transformer(num_layers=cfg.model.num_layers,
                        d_model=cfg.model.d_model,
                        num_attention_heads=cfg.model.num_heads,
                        dff=cfg.model.dff,
                        max_len=cfg.model.max_len,
                        input_vocab_size=cfg.model.input_vocab_size,
                        target_vocab_size=cfg.model.input_vocab_size,
                        dropout_rate=cfg.model.dropout_rate)

    model.load_weights(checkpoint_filepath)

    predictor = TransformerPredictor(
        transformer=model,
        sp_input=sp_input,
        sp_output=sp_output
    )

    while True:
        input_sequence = input("Write input sequence: \n")
        predicted_sequence, output_ids = predictor(input_sequence, max_length=cfg.model.max_len)
        print(predicted_sequence)