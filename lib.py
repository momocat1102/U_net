import os, cv2, numpy as np
from keras.layers import concatenate, Conv2D, MaxPooling2D, Input, Conv2DTranspose
from keras.models import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 生成資料集
def creat_data(xfile, yfile, imgsize, istrain = True):
    x, y = list(), list()
    files = os.listdir(xfile)
    if istrain:
        np.random.seed(202105)
        np.random.shuffle(files)
    for file in files:
        name = file.split('.')[0]
        img = cv2.imread(xfile + '/' + file)
        try:
            img[0]
        except:
            continue
        mask = cv2.imread(yfile + '/' + name + '.png', cv2.IMREAD_UNCHANGED)
        mask = np.where((mask == 1)|(mask == 3), 255, 0).astype('uint8')
        img = cv2.resize(img, (imgsize, imgsize))
        mask = cv2.resize(mask, (imgsize, imgsize))
        # 小於閥值的灰度值設為0，其他設為最大灰度值
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[-1]
        mask = np.expand_dims(mask, axis=2)
        # cv2.imwrite('test3.jpg', mask)
        x.append(img)
        y.append(mask)
    x = np.array(x)
    y = np.array(y)
    # print(x.shape, y.shape)
    return x, y

# 去觀察每個layer所產生的圖片
def save_result(val_out, val_block_size, img_size, max_num, image_path):
    def preprocess(img):
        img = (img*255.0).astype(np.uint8)
        img = np.squeeze(img, axis=0)
        return img

    preprocesed = preprocess(val_out)
    # print(preprocesed.shape)
    final_image = np.array([])
    single_row = np.array([])
    if max_num > val_out.shape[-1]:
        num = val_out.shape[-1]
    else:
        num = max_num
    if val_block_size > num:
        print("val_block_size必須小於num")
        return
    for b in range(num):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[ :, :, b]
            single_row = cv2.resize(single_row, (img_size, img_size))
        else:
            single_row_now = preprocesed[:, :, b]
            single_row_now = cv2.resize(single_row_now, (img_size, img_size))
            single_row = np.concatenate((single_row, single_row_now), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])
    final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, final_image)

def layer_model(model, end):
    end_layer = model.get_layer(name=end)
    print("input to " + end_layer.name)
    conv1_layer = Model(inputs=model.input,outputs=end_layer.output)
    # conv1_layer.summary()
    return conv1_layer



def U_net():
    inputs = Input(shape=(160, 160, 3))
    conv1 = Conv2D(64, [3, 3], padding='same', activation='relu')(inputs)
    conv1 = Conv2D(64, [3, 3], padding='same', activation='relu')(conv1)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # print(conv1.shape)
    # cv2.imwrite('test1.jpg', maxpool1)

    conv2 = Conv2D(128, [3, 3], padding='same', activation='relu')(maxpool1)
    conv2 = Conv2D(128, [3, 3], padding='same', activation='relu')(conv2)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # print(conv2.shape)

    conv3 = Conv2D(256, [3, 3], padding='same', activation='relu')(maxpool2)
    conv3 = Conv2D(256, [3, 3], padding='same', activation='relu')(conv3)
    maxpool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # print(conv3.shape)

    conv4 = Conv2D(512, [3, 3], padding='same', activation='relu')(maxpool3)
    conv4 = Conv2D(512, [3, 3], padding='same', activation='relu')(conv4)
    maxpool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # print(conv4.shape)

    conv5 = Conv2D(1024, [3, 3], padding='same', activation='relu')(maxpool4)
    conv5 = Conv2D(1024, [3, 3], padding='same', activation='relu')(conv5)
    # print(conv5.shape)

    us1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = Conv2D(512, [3, 3], padding='same', activation='relu')(us1)
    # print(up6.shape, conv4.shape)
    marge1 = concatenate([conv4, up6], axis=3)
    # print(up6, conv4)
    up6 = Conv2D(512, [3, 3], padding='same', activation='relu')(marge1)
    up6 = Conv2D(512, [3, 3], padding='same', activation='relu')(up6)

    us2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(up6)
    up7 = Conv2D(256, [3, 3], padding='same', activation='relu')(us2)
    marge2 = concatenate([conv3, up7], axis=3)
    # print(up7.shape, conv3.shape)
    up7 = Conv2D(256, [3, 3], padding='same', activation='relu')(marge2)
    up7 = Conv2D(256, [3, 3], padding='same', activation='relu')(up7)

    us3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up7)
    up8 = Conv2D(128, [3, 3], padding='same', activation='relu')(us3)
    marge3 = concatenate([conv2, up8], axis=3)
    # print(up8.shape, conv2.shape)
    up8 = Conv2D(128, [3, 3], padding='same', activation='relu')(marge3)
    up8 = Conv2D(128, [3, 3], padding='same', activation='relu')(up8)

    us4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up8)
    up9 = Conv2D(64, [3, 3], padding='same', activation='relu')(us4)
    marge4 = concatenate([conv1, up9], axis=3)
    # print(up9.shape, conv1.shape)
    up9 = Conv2D(64, [3, 3], padding='same', activation='relu')(marge4)
    up9 = Conv2D(64, [3, 3], padding='same', activation='relu')(up9)

    out = Conv2D(1, [1, 1], padding='same', activation='sigmoid')(up9)

    model = Model(inputs, out)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

