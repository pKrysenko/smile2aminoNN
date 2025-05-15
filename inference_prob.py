import sys
import json

import tensorflow as tf
import numpy as np
from omegaconf import OmegaConf
import sentencepiece as spm

from models import TransformerProbs
from predictor import TransformerPredictorProb


if __name__ == "__main__":
    cfg = OmegaConf.load(sys.argv[1])
    sp_first_filepath = sys.argv[2]
    sp_second_filepath = sys.argv[3]
    checkpoint_filepath = sys.argv[4]
    scaler_path = sys.argv[5]

    sp_input = spm.SentencePieceProcessor(model_file=sp_first_filepath)
    sp_output = spm.SentencePieceProcessor(model_file=sp_second_filepath)

    model = TransformerProbs(num_layers=cfg.model.num_layers,
                             d_model=cfg.model.d_model,
                             num_attention_heads=cfg.model.num_heads,
                             dff=cfg.model.dff,
                             max_len=cfg.model.max_len,
                             first_vocab_size=cfg.model.input_vocab_size,
                             second_vocab_size=cfg.model.input_vocab_size,
                             dropout_rate=cfg.model.dropout_rate)
    
    

    model((np.random.randn(1, cfg.model.max_len), np.random.randn(1, cfg.model.max_len)))
    print(model.summary())

    #model.load_weights(checkpoint_filepath)

    with open(scaler_path, 'r') as f:
        scaler_dict = json.load(f)


    predictor = TransformerPredictorProb(
        transformer=model,
        sp_first=sp_input,
        sp_second=sp_output,
        scaler_dict=scaler_dict
    )

    while True:
        first_sequence = input("Write SMILE sequence: \n")
        second_sequence = input("Write amino sequence: \n")
        score = predictor(first_sequence, second_sequence, max_length=cfg.model.max_len)
        print(f"Score: {score}")