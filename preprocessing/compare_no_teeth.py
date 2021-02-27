import numpy as np
from utils import *

for label in ['ABOUT', 'ABSOLUTELY', 'ABUSE', 'ACCESS', 'ACCORDING']:
    for i in range(1, 50):
        read_direc = "../datasets/LRW_h96w96_mouth_crop_gray_frontalized_no_teeth_compare"
        save_direc = "../datasets/LRW_h96w96_mouth_crop_gray_frontalized_no_teeth_compare"
        filename = "{}_{:05d}".format(label, i)
        srs_pathname = os.path.join(save_direc, label, 'test', filename + '.npz')
        dst_pathname = os.path.join(save_direc, label, 'test', filename + '.npz')
        a = np.load(srs_pathname)['data']
        for idx in range(29):
            a[idx, 5:15, 10:55] = 0
        save2npz(dst_pathname, data=a)