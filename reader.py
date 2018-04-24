from PIL import Image
import numpy as np

def pre_img(imagename):
    img = Image.open(imagename)
    reimg = img.resize((28,28),Image.ANTIALIAS)
    im_arr = np.array(reimg.convert('L'))
    threshold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if(im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    nm_arr = im_arr.reshape([1,784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)
    return img_ready

