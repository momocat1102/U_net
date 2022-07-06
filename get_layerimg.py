from lib import *

def main():
    # 照片處理
    img_size = 160
    layer_imgs_path = 'layer2_img'
    img = cv2.imread('img3.jpg')
    img = cv2.resize(img, (img_size, img_size))
    img_batch = np.expand_dims(img, axis=0)

    model = U_net()
    model.load_weights('unet_weight.h5')
    # model.summary()
    for i in [1, 3, 5, 7, 9, 10, 12, 13, 15, 16, 17, 18, 19, 21]:
        layer_name = 'conv2d_' + str(i)
        conv1_layer = layer_model(model, layer_name)
        conv_img = conv1_layer.predict(img_batch)
        save_result(conv_img, 8, 160, 64, './' + layer_imgs_path + '/layer' + layer_name + '.jpg')

    pre = model.predict(img_batch)
    save_result(pre, 1, 160, 1, './' + layer_imgs_path + '/predict.jpg')

if __name__ == "__main__":
    main()
    
    