from os import listdir
from os import makedirs
from os import path
import cv2
import numpy as np

matching_th = 0.70

size = [30, 30]
AFR = cv2.resize(cv2.imread('Templates/TestSetIcon/AFR_C.png', 0), size)
AFR_U = cv2.resize(cv2.imread('Templates/TestSetIcon/AFR_U.png', 0), size)
ZNR = cv2.resize(cv2.imread('Templates/SetIcons/ZNR_C.jpg', 0), size)
ZNR_R = cv2.resize(cv2.imread('Templates/TestSetIcon/InkedZNR_R_LI.jpg', 0), size)
w, h = ZNR.shape[::-1]


def set_check(color_template, draw_color, img_gray, img_to_check):
    results = cv2.matchTemplate(img_gray, color_template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(results >= matching_th)
    for pt in zip(*locations[::-1]):
        cv2.rectangle(img_to_check, pt, (pt[0] + w, pt[1] + h), draw_color, 2)
        print("set found")


def show():
    cv2.imshow("ref", ZNR_R)
    cv2.imshow("ref1", AFR_U)
    cv2.imshow("ref2", img_gray)
    cv2.imshow("img", img_to_check)
    cv2.waitKey(0)


for f in listdir("testCard"):
    filename = "testCard/" + f
    img_to_check = cv2.imread(filename)

    img_gray = cv2.cvtColor(img_to_check, cv2.COLOR_BGR2GRAY)

    set_check(AFR, (0, 255, 0), img_gray, img_to_check)
    set_check(AFR_U, (0, 255, 0), img_gray, img_to_check)
    set_check(ZNR, (0, 255, 0), img_gray, img_to_check)
    set_check(ZNR_R, (0, 255, 0), img_gray, img_to_check)
    if not path.isdir("debug/resultCards/"):
        makedirs("debug/resultCards/")

    filename2 = "debug/resultCards/" + f
    cv2.imwrite(filename2, img_to_check)
    print(f'checked color in image: {f}')
    show()
