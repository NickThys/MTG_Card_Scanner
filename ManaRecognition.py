from os import listdir
from os import makedirs
from os import path
import cv2
import numpy as np

matching_th = 0.8
green = cv2.imread('Templates/ManaColors/Green.jpg', 0)
blue = cv2.imread('Templates/ManaColors/Blue.jpg', 0)
red = cv2.imread('Templates/ManaColors/Red.jpg', 0)
white = cv2.imread('Templates/ManaColors/White.jpg', 0)
black = cv2.imread('Templates/ManaColors/Black.jpg', 0)

w, h = green.shape[::-1]


def color_check(color_template, draw_color, img_gray, img_to_check):
    results = cv2.matchTemplate(img_gray,color_template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(results >= matching_th)
    for pt in zip(*locations[::-1]):
        cv2.rectangle(img_to_check, pt, (pt[0] + w, pt[1] + h), draw_color, 2)


for f in listdir("perspective_fix"):
    filename = "perspective_fix/" + f
    img_to_check = cv2.imread(filename)

    img_gray = cv2.cvtColor(img_to_check, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("ref",green)
    #cv2.imshow("img",img_to_check)
    cv2.waitKey(0)
    color_check(green, (0, 255, 0), img_gray, img_to_check)
    color_check(red, (0, 0, 255), img_gray, img_to_check)
    color_check(black, (0, 0, 0), img_gray, img_to_check)
    color_check(blue, (255, 0, 0), img_gray, img_to_check)
    color_check(white, (255, 255, 255), img_gray, img_to_check)
    if not path.isdir("resultCards/"):
        makedirs("resultCards/")

    filename2 = "resultCards/" + f
    cv2.imwrite(filename2, img_to_check)
    print(f'checked color in image: {f}')
