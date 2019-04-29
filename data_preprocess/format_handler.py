import cv2
# from PIL import Image
import sys
sys.path.insert(0, './')
from utils import get_config
import os


def format_handler():
    raw_imgs_root = config["raw_imgs_root"]
    p1 = os.path.join(raw_imgs_root, "train", '0', 'efc4299809b064a9870f5933a07ccb7b.png')
    p2 = os.path.join(raw_imgs_root, "train", '0', '77dc5f7a307eed10b00ce89069373fb5.jpg')
    img1 = cv2.imread(p1)
    img2 = cv2.imread(p2)
    print(img1.shape)
    print(img2.shape)
    # print(img1)
    # print(img2)




if __name__ == '__main__':
    config = get_config()
    format_handler()