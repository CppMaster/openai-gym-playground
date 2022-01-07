import logging

from keras.callbacks import Callback


class EpochLogger(Callback):

    def __init__(self):
        super().__init__()
        self.log = logging.getLogger(__name__)

    def on_epoch_end(self, epoch, logs=None):
        self.log.debug(f"Epoch: {epoch+1},\tloss: {logs['loss']}" +
                       f",\tval_loss: {logs['val_loss']}" if 'val_loss' in logs else "")
