import numpy as np
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv1D
from keras.optimizers import Adam
from keras.utils import Sequence


class BitGenerator(Sequence):

    def __init__(self, n_bits: int = 8, batch_size: int = 8):
        self.n_bits = n_bits
        self.batch_size = batch_size

    def __len__(self):
        return 1

    def __getitem__(self, index):
        x = (np.random.random((self.batch_size, self.n_bits, 2)) > 0.5).astype(float)
        y = (np.expand_dims(np.logical_xor(x[:, :, 0], x[:, :, 1]), axis=2)).astype(float)
        return x, y


def get_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred == target).sum(axis=(1, 2)) == pred.shape[1]))


if __name__ == "__main__":

    batch_size = 100

    model = Sequential([
        Conv1D(4, kernel_size=(1,), activation="relu"),
        Conv1D(1, kernel_size=(1,), activation="relu"),
    ])
    model.compile(optimizer=Adam(lr=0.01), loss="mse")

    callbacks = [
        ReduceLROnPlateau(patience=10, monitor="loss"),
        EarlyStopping(patience=100, monitor="loss")
    ]

    for power in range(8):
        n_bits = int(2**power)
        print(f"Bits : {n_bits}")
        generator = BitGenerator(n_bits, batch_size)
        x, y = generator[0]
        pred_y = model.predict(x, batch_size=batch_size) > 0.5
        print(f"Accuracy before training: {get_accuracy(pred_y, y > 0.5)}")
        model.fit(x=generator, epochs=10000, verbose=2, callbacks=callbacks)
        x, y = generator[0]
        pred_y = model.predict(x, batch_size=batch_size) > 0.5
        print(f"Accuracy after training: {get_accuracy(pred_y, y > 0.5)}")



