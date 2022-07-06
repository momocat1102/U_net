import os, cv2, numpy as np
from lib import *


def main():
    file_img = 'test_img'
    file_label = 'test_label'

    x_test, y_test = creat_data(file_img, file_label, 160, False)

    x_test = x_test.astype('float32')/255.0
    y_test = y_test.astype('float32')/255.0
    print(x_test.shape, y_test.shape)

    model = U_net()
    model.load_weights('unet_weight.h5')
    # model.summary()
    import shutil
    try:
        shutil.rmtree('test_predict')
    except:
        pass
    os.mkdir('test_predict')

    predict = model.predict(x_test)

    for ind in range(len(predict)):
        img=(x_test[ind]*255.0).astype('uint8')
        true=(y_test[ind]*255.0).astype('uint8')
        true=np.pad(true, ((0,0),(0,0),(1,1)))
        pred=(predict[ind]*255.0).astype('uint8')
        pred=np.pad(pred, ((0,0),(0,0),(1,1)))
        show=np.concatenate((img, true, pred), axis=1)
        cv2.imwrite('test_predict/'+str(ind).zfill(4)+'.jpg', show)
        # cv2.imwrite('test_predict/I'+str(ind).zfill(4)+'.jpg', img)
        # cv2.imwrite('test_predict/T'+str(ind).zfill(4)+'.jpg', true)
        # cv2.imwrite('test_predict/P'+str(ind).zfill(4)+'.jpg', pred)


if __name__ == "__main__":
    main()