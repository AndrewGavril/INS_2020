import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from image_marking import draw_masks
from sklearn.model_selection import train_test_split
from keras.regularizers import l2

l2_lambda = 0.0001

img_rows = 64
img_cols = 400


def load_data():
    data = np.load('imgs.npy')
    target = np.load('masks.npy')
    print(data.shape)
    total, w, h = data.shape
    data = data.reshape((total, w, h, 1))
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1, random_state=42)
    return (train_data, train_target, test_data, test_target)

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', activity_regularizer=l2(l2_lambda))(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', activity_regularizer=l2(l2_lambda))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', activity_regularizer=l2(l2_lambda))(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', activity_regularizer=l2(l2_lambda))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', activity_regularizer=l2(l2_lambda))(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', activity_regularizer=l2(l2_lambda))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', activity_regularizer=l2(l2_lambda))(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', activity_regularizer=l2(l2_lambda))(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(4, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.load_weights('start_weights.h5')

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def plot_dice_coef(history):
    dice_coef = history.history['dice_coef']
    val_dice_coef = history.history['val_dice_coef']
    plt.plot(dice_coef)
    plt.plot(val_dice_coef)
    plt.title('model Dice coef')
    plt.ylabel('Dice coef')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def train_and_predict():
    imgs_train, imgs_mask_train, imgs_test, imgs_mask_test = load_data()

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)
    std = np.std(imgs_train)
    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')

    model = get_unet()
    model_checkpoint = ModelCheckpoint('reserve/weights.h5', monitor='val_loss', save_best_only=True)

    history = model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=20, verbose=1, shuffle=True,
              validation_split=0.2, callbacks=[model_checkpoint])
    model.save('model.h5')

    plot_dice_coef(history)
    imgs_test = imgs_test.astype('float32')

    imgs_test -= mean
    imgs_test /= std

    imgs_mask_test_res = model.predict(imgs_test, verbose=1)

    indices = np.random.choice(imgs_test.shape[0], 5)

    for n, i in enumerate(indices):
        draw_masks((imgs_test[i, :, :, 0]*std + mean).astype('uint8'),
                   imgs_mask_test_res[i], '{}.png'.format(n))
        draw_masks((imgs_test[i, :, :, 0]*std + mean).astype('uint8'),
                   imgs_mask_test[i], '{}_true.png'.format(n))


if __name__ == '__main__':
    train_and_predict()
