import os
import shutil
import sys
import numpy as np
import cv2
from os import listdir
from os import makedirs
from os import path
from mtgsdk import Card
import pytesseract


def testContourValidity(contour, full_width, full_height):
    # Max countour width/height/area is 95% of whole image
    max_threshold = 0.95
    # Min contour width/height/area is 30% of whole image
    min_threshold = 0.3
    min_area = full_width * full_height * min_threshold
    max_area = full_width * full_height * max_threshold
    max_width = max_threshold * full_width
    max_height = max_threshold * full_height
    min_width = min_threshold * full_width
    min_height = min_threshold * full_height

    # Area
    size = cv2.contourArea(contour)
    if size < min_area:
        return False
    if size > max_area:
        return False

    # Width / Height
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    (tl, tr, br, bl) = sort_points(box)
    box_width = int(((br[0] - bl[0]) + (tr[0] - tl[0])) / 2)
    box_height = int(((br[1] - tr[1]) + (bl[1] - tl[1])) / 2)
    if box_width < min_width:
        return False
    if box_height < min_height:
        return False
    if box_width > max_width:
        return False
    if box_height > max_height:
        return False

    return True


def find_square(im, f):
    # Width and height for validity check
    h = np.size(im, 0)
    w = np.size(im, 1)
    # Grayscale and blur before trying to find contours
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 1000)
    debug_image(blur, '1_preprocess_gray', f)
    # Threshold and contours
    flag, thresh = cv2.threshold(blur, 115, 255, cv2.THRESH_BINARY)
    debug_image(thresh, '2_preprocess_thresh', f)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Debug
    im_debug = im.copy()
    # Find largest contour which does not take full image size.
    max = None
    for x in contours:
        if testContourValidity(x, w, h):
            im_debug = cv2.drawContours(im_debug, [x], -1, (0, 255, 0), 3)
            if max is None or cv2.contourArea(max) < cv2.contourArea(x):
                max = x
    # Debug
    debug_image(im_debug, '3_possible_contours', f)
    # Min area rectangle around that contour. This nicely finds corners as MTG cards are rounded
    rect = cv2.minAreaRect(max)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


def dpot(a, b):
    return (a - b) ** 2


def adist(a, b):
    return np.sqrt(dpot(a[0], b[0]) + dpot(a[1], b[1]))


def max_distance(a1, a2, b1, b2):
    dist1 = adist(a1, a2)
    dist2 = adist(b1, b2)
    if int(dist2) < int(dist1):
        return int(dist1)
    else:
        return int(dist2)


def sort_points(pts):
    ret = np.zeros((4, 2), dtype="float32")
    sumF = pts.sum(axis=1)
    diffF = np.diff(pts, axis=1)

    ret[0] = pts[np.argmin(sumF)]
    ret[1] = pts[np.argmin(diffF)]
    ret[2] = pts[np.argmax(sumF)]
    ret[3] = pts[np.argmax(diffF)]

    return ret


def fix_perspective(image, pts):
    (tl, tr, br, bl) = sort_points(pts)
    maxW = max_distance(br, bl, tr, tl)
    maxH = max_distance(tr, br, tl, bl)
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    transform = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
    fixed = cv2.warpPerspective(image, transform, (maxW, maxH))
    fixed_resized = cv2.resize(fixed, [550, 740])
    return fixed_resized


def debug_image(img, extra_path, filename):
    fpath = "debug/" + extra_path + "/"
    if not path.isdir(fpath):
        makedirs(fpath)
    cv2.imwrite(fpath + filename, img)


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

y1, y2 = 20, 72
x1, x2 = 30, 460

for f in listdir("TestCards"):
    print(f)
    filename = "TestCards/" + f
    img = cv2.imread(filename)
    square = find_square(img, f)
    # Debug
    im_debug = cv2.drawContours(img.copy(), [square], -1, (0, 255, 0), 3)
    debug_image(im_debug, "4_selected_contour", f)
    # Fix and save
    img = fix_perspective(img, square)

    debug_image(img, '5_perspective_fix', f)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    title = gray[y1:y2, x1:x2]

    flag, thresh = cv2.threshold(title, 100, 255, cv2.THRESH_BINARY)
    file_name = 'titles/' + f
    cv2.imwrite(file_name, thresh)
    debug_image(thresh, '5_title_thresh', f)
    debug_image(title, '6_title_img', f)
    print(f'thresh: {pytesseract.image_to_string(thresh).strip()}')
    title=pytesseract.image_to_string(thresh)
    cards = Card.where(name=title.strip()).all()

    for card in cards:
        print(card.name + ': set ' + card.set)
    val = input("continue to next card? y/n: ")
    if val=='y' or val=="Y":
        pass
    else:
        break

val = input("Delete debug images? y/n: ")
if val=='y' or val=="Y":
    shutil.rmtree('debug')
    print("debug images deleted...")
else:
    pass
