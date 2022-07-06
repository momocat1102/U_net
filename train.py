from lib import *
from keras.callbacks import ModelCheckpoint

def main():
    file_img = 'images'
    file_label = 'annotations/trimaps'
    x, y = creat_data(file_img, file_label, 160)

    x_train = x[int(len(x) * 0.5):]
    y_train = y[int(len(x) * 0.5):]
    x_valid = x[:int(len(x) * 0.5)]
    y_valid = y[:int(len(x) * 0.5)]

    x_train = x_train.astype('float32') / 255.0
    y_train = y_train.astype('float32') / 255.0
    x_valid = x_valid.astype('float32') / 255.0
    y_valid = y_valid.astype('float32') / 255.0

    print(x_train.shape, y_train.shape)
    print(x_train.dtype, y_train.dtype)
    print(x_valid.shape, y_valid.shape)
    print(x_valid.dtype, y_valid.dtype)

    model = U_net()
    model.summary()
    checkpoint = ModelCheckpoint(
        filepath='unet_weight.h5',
        save_best_only=True,
        save_weights_only=True
    )
    model.fit(
        x_train, y_train,
        batch_size=16,
        validation_data=(x_valid, y_valid),
        callbacks=checkpoint,
        epochs=20)

if __name__ == "__main__":
    main()