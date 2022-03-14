import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from itertools import product

def cut_obj():
    path = '/data2/hhp/dataset/邵逸夫造口伤口/example/1587-3238760-王志坤/2014.8.22/IMG_6796.JPG'
    # 读取原图
    img_ori = cv2.imread(path)
    #img_ori = np.array(Image.open(path))
    img_gray = cv2.cvtColor(img_ori, cv2.COLOR_RGB2GRAY)

    # 全局阈值分割
    retval, img_global1 = cv2.threshold(img_gray, 160, 255, cv2.THRESH_BINARY)
    retval, img_global2 = cv2.threshold(img_gray, 170, 255, cv2.THRESH_BINARY)
    retval, img_global3 = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
    retval, img_global4 = cv2.threshold(img_gray, 190, 255, cv2.THRESH_BINARY)
    retval, img_global5 = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    retval, img_global6 = cv2.threshold(img_gray, 210, 255, cv2.THRESH_BINARY)
    # 自适应阈值分割
    img_ada_mean = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3)
    img_ada_gaussian = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
    imgs = [img_gray, img_global1, img_global2, img_global3, img_global4, img_global5, img_global6, img_ada_mean, img_ada_gaussian]
    titles = ['gray image', 'Global Thresholding(160)','Global Thresholding(170)','Global Thresholding(180)','Global Thresholding(190)','Global Thresholding(200)',
              'Global Thresholding(210)',
              'Adaptive Mean', 'Adaptive Guassian']
    # 显示图片
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(imgs[i], 'gray') #
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.savefig('test2.png')
    plt.show()





def cut01(bin_size=20):
    #基于某点像素值或者块像素值的方法计算并测量出标尺所在的位置，前者不可行，因确认大片黑色区域也存在白色像素点，因此尝试后者
    path = '/data2/hhp/dataset/邵逸夫造口伤口/example/1587-3238760-王志坤/2014.8.22/IMG_6796.JPG'
    # 读取原图
    img_ori = cv2.imread(path)
    img_shp = img_ori.shape
    print('shape:', img_shp)
    img_gray = cv2.cvtColor(img_ori, cv2.COLOR_RGB2GRAY)
    retval, img_global = cv2.threshold(img_gray, 210, 255, cv2.THRESH_BINARY)

    axis = []
    for m, n in product(range(math.ceil(img_shp[0]/bin_size)), range(math.ceil(img_shp[1]/bin_size))):
        x, y = m * bin_size, n * bin_size
        # print('b', img[x:x + bin_size, y:y + bin_size, 1].shape)
        mean = img_global[x:x + bin_size, y:y + bin_size].mean()
        # print('mean', mean,x,y)
        if (mean > 140):  #
            axis.append([x, y])

    axis = np.array(axis)
    left_up = [axis[:,0].min(),axis[:,1].min()]
    right_up = [axis[:, 0].min(), axis[:, 1].max()]
    left_down = [axis[:, 0].max(), axis[:, 1].min()]
    right_down = [axis[:, 0].max(), axis[:, 1].max()]
    width = (left_down[0]-left_up[0])
    height = (right_up[1]-left_up[1])
    print('ruler width,height:',width,height)



if __name__ == '__main__':
    cut_obj()
    #cut01()