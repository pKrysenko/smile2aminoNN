import os

import hydra

import tensorflow as tf
import numpy as np
from jupyterlab.browser_check import test_flags
from sklearn.model_selection import train_test_split

from omegaconf import OmegaConf, DictConfig
from data import DataLoader
from model import Transformer
from predictor import TransformerPredictor

import logging
log = logging.getLogger(__name__)


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)

@hydra.main(version_base=None, config_path='configs', config_name='transformer')
def app(cfg):
    print(OmegaConf.to_yaml(cfg))
    output_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']

    dataloader = DataLoader(csv_path=cfg.pathes.csv_path,
                            vocab_size=cfg.preprocessing.input_vocab_size,
                            max_len=cfg.preprocessing.max_len,
                            input_column=cfg.preprocessing.input_column,
                            output_column=cfg.preprocessing.output_column)

    encoded_inputs, add_inputs, labels = dataloader.pipeline()
    #encoded_inputs, add_inputs, labels = encoded_inputs[:4], add_inputs[:4], labels[:4]
    X_train, X_val, A_train, A_val, y_train, y_val = train_test_split(encoded_inputs, add_inputs, labels,
                                                                      test_size=0.2, random_state=42)

    model = Transformer(num_layers=cfg.model.num_layers,
                        d_model=cfg.model.d_model,
                        num_attention_heads=cfg.model.num_heads,
                        dff=cfg.model.dff,
                        max_len=cfg.model.max_len,
                        input_vocab_size=cfg.model.input_vocab_size,
                        target_vocab_size=cfg.model.input_vocab_size,
                        dropout_rate=cfg.model.dropout_rate)

    model((np.random.randn(1, cfg.model.max_len), np.random.randn(1, cfg.model.max_len)))
    model.summary()

    optimizer = tf.keras.optimizers.Adam(cfg.optimizer.lr, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)


    checkpoint_filepath = os.path.join(output_dir, "model", "best_model", "model.weights.h5")

    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_masked_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True,
        save_weights_only=True)

    model.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy])

    model.fit((X_train, A_train), y_train,
              epochs=cfg.train.epochs,
              batch_size=cfg.train.batch_size,
              validation_data=((X_val, A_val), y_val),
              callbacks=[model_checkpoint_cb])

    model.load_weights(checkpoint_filepath)
    predictor = TransformerPredictor(
        transformer=model,
        sp_input=dataloader.sp_input,
        sp_output=dataloader.sp_output
    )

    inp_sent = "".join([predictor.sp_input.decode(int(char)) for char in X_val[0]])
    orig_sent = "".join([predictor.sp_output.decode(int(char)) for char in y_val[0]])
    predicted_sent, output_ids = predictor(inp_sent, max_length=cfg.model.max_len)

    logging.info(f"Input sentence: {inp_sent}")
    logging.info(f"Target sequence: {orig_sent}")
    logging.info(f"Predicted sequence: {predicted_sent}")
    logging.info("="*80)
   # logging.info(" ".join([str(id_) for id_ in y_val]))
   # logging.info(" ".join([str(id_) for id_ in output_ids]))


if __name__ == "__main__":
    app()

