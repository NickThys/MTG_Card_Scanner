import os
import shutil
from urllib.request import urlopen
import mtgsdk
import numpy as np
import cv2
from os import listdir
from os import makedirs
from os import path
import pytesseract

from pytesseract import Output


def url_to_image(url, readFlag=cv2.IMREAD_GRAYSCALE):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)

    # return the image
    return image


class CardImage:
    # noinspection PyShadowingNames
    def __init__(self, is_debugging):
        self.is_debugging = is_debugging

    def __test_contour_validity(self, contour, full_width, full_height):
        # Max contour width/height/area is 95% of whole image
        max_threshold = 0.95
        # Min contour width/height/area is 30% of whole image
        min_threshold = 0.2
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
        # noinspection PyTypeChecker
        (tl, tr, br, bl) = self.__sort_points(box)
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

    def __find_square(self, im, file_name):
        # Width and height for validity check
        h = np.size(im, 0)
        w = np.size(im, 1)
        # Grayscale and blur before trying to find contours
        img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img_gray, (1, 1), 1000)
        if self.is_debugging is True:
            DebugImage.write_image(blur, '1_preprocess_gray', file_name)
        # Threshold and contours
        _, threshold = cv2.threshold(blur, 115, 255, cv2.THRESH_BINARY)
        if self.is_debugging is True:
            DebugImage.write_image(threshold, '2_preprocess_thresh', file_name)
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Debug
        img_debug = im.copy()
        # Find largest contour which does not take full image size.
        max_contour = None
        for x in contours:
            if self.__test_contour_validity(x, w, h):
                img_debug = cv2.drawContours(img_debug, [x], -1, (0, 255, 0), 3)
                if max_contour is None or cv2.contourArea(max_contour) < cv2.contourArea(x):
                    max_contour = x
        # Debug
        if self.is_debugging is True:
            DebugImage.write_image(img_debug, '3_possible_contours', file_name)
        # Min area rectangle around that contour. This nicely finds corners as MTG cards are rounded
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box

    @staticmethod
    def __d_pot(a, b):
        return (a - b) ** 2

    def __a_dist(self, a, b):
        return np.sqrt(self.__d_pot(a[0], b[0]) + self.__d_pot(a[1], b[1]))

    def __max_distance(self, a1, a2, b1, b2):
        dist1 = self.__a_dist(a1, a2)
        dist2 = self.__a_dist(b1, b2)
        if int(dist2) < int(dist1):
            return int(dist1)
        else:
            return int(dist2)

    @staticmethod
    def __sort_points(pts):
        ret = np.zeros((4, 2), dtype="float32")
        sum_f = pts.sum(axis=1)
        diff_f = np.diff(pts, axis=1)

        ret[0] = pts[np.argmin(sum_f)]
        ret[1] = pts[np.argmin(diff_f)]
        ret[2] = pts[np.argmax(sum_f)]
        ret[3] = pts[np.argmax(diff_f)]

        return ret

    def __fix_perspective(self, image, pts):
        (tl, tr, br, bl) = self.__sort_points(pts)
        max_w = self.__max_distance(br, bl, tr, tl)
        max_h = self.__max_distance(tr, br, tl, bl)
        dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype="float32")
        transform = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
        fixed = cv2.warpPerspective(image, transform, (max_w, max_h))
        fixed_resized = cv2.resize(fixed, [550, 740])
        return fixed_resized

    def filter_card_from_img(self, image):
        square = self.__find_square(image, f)
        im_debug = cv2.drawContours(image.copy(), [square], -1, (0, 255, 0), 3)
        if self.is_debugging is True:
            DebugImage.write_image(im_debug, "4_selected_contour", f)
        # noinspection PyTypeChecker
        image = self.__fix_perspective(image, square)
        # set the image right
        if self.is_debugging is True:
            DebugImage.write_image(image, '5_perspective_fix', f)
        return image


class DebugImage:

    @staticmethod
    def write_image(debug_img, extra_path, debug_filename):
        f_path = "debug/" + extra_path + "/"
        if not path.isdir(f_path):
            makedirs(f_path)
        cv2.imwrite(f_path + debug_filename, debug_img)


# <editor-fold desc="filter title">
def filter_title(texts):
    original_title = ''
    for i in range(len(texts)):
        if texts[i] != "" and texts[i] != " ":
            if not any(ext in texts[i] for ext in testStr):
                if original_title != '':
                    original_title += ' '
                original_title += texts[i]
    return original_title


def remove_word_from_title(full_title, removeFirstWord=True):
    split_title = full_title.split()
    for word in range(len(split_title)):
        if removeFirstWord is True and word == 0:
            split_title.pop(word)
        if removeFirstWord is False and word == len(split_title) - 1:
            split_title.pop(word)
    return ' '.join(map(str, split_title))


# </editor-fold>


# noinspection PyShadowingNames
def get_cards(title, mtg_set=''):
    return mtgsdk.Card.where(set=mtg_set).where(name=title).where(page=0).where(pageSize=10).all()


def get_exact_card(card_list, card_img):
    card_img = card_img
    if len(card_list) == 0:
        return None
    elif len(card_list) == 1:
        '''cv2.imshow("og", card_img)
        cv2.imshow("suggested", url_to_image(card_list[0].image_url))
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''
        return card_list[0]
    else:
        result = []
        for card_index in range(len(card_list)):
            if card_list[card_index].image_url is not None:
                img_gray = cv2.resize(cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY), [200, 300])
                template = cv2.resize(url_to_image(card_list[card_index].image_url), [200, 300])
                # cv2.imshow('test',template)
                # cv2.waitKey(0)

                res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
                result.append([card_index, res])
        index_highest_card = None
        highest_val = 0
        for single_result in range(len(result)):
            if result[single_result][1] >= highest_val:
                index_highest_card = result[single_result][0]
                highest_val = result[single_result][1]
            pass
        '''cv2.imshow("og", card_img)
        cv2.imshow("suggested", url_to_image(card_list[index_highest_card].image_url))
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        print(f'\nindex card: {index_highest_card}\n'
              f'card name: {card_list[index_highest_card].name}\n')
        return card_list[index_highest_card]


def card_searching_single(txt_title, card_img):
    card_list = []
    if txt_title != "":
        cards = get_cards(txt_title)
        for card_index in range(len(cards)):
            card_list.append(cards[card_index])
        if len(cards) == 0:
            print(txt_title + " not found in the db"
                              "\n try to remove last word ... ")
            title_r = remove_word_from_title(txt_title, False)
            cards = get_cards(title_r)

            for card_index in range(len(cards)):
                print(cards[card_index].name + " : " + cards[card_index].set)
                # if card == len(cards) - 1:
                card_list.append(cards[card_index])
            if len(cards) == 0:
                print(title_r + " not found in the db"
                                "\n try to remove First word ... ")
                title_r = remove_word_from_title(txt_title)
                cards = get_cards(title_r)

                for card_index in range(len(cards)):
                    print(cards[card_index].name + " : " + cards[card_index].set)
                    # if card == len(cards) - 1:
                    card_list.append(cards[card_index])
                if len(cards) == 0:
                    print(title_r + " not found in the db"
                                    "\n gonna try to search after the card by searching for the card word by word")
                    title_split = title_txt.split()
                    for word in title_split:
                        cards = get_cards(word)
                        if len(cards) <= 25:
                            for card_index in range(len(cards)):
                                print(cards[card_index].name + " : " + cards[card_index].set)
                                card_list.append(cards[card_index])
                            break
    return get_exact_card(card_list, card_img)


# constants
testStr = [":", "’", ";", "—", "$", "/", "_", "|"]
os.putenv('TESSDATA_PREFIX', 'C:\\Program Files\\Tesseract-OCR\\tessdata\\mtg.traineddata')
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# variables
y1, y2 = 15, 90
x1, x2 = 30, 410

amount_of_cards_not_found = 0

back_mtg_card = cv2.imread("Templates/Magic_card_back.jpg")
size_card = [back_mtg_card.shape[1], back_mtg_card.shape[0]]
cards_not_found = back_mtg_card

is_debugging = False
card_image = CardImage(is_debugging)

for f in listdir("TestCards"):
    print(f)
    filename = "TestCards/" + f
    img = cv2.imread(filename)
    # get fixed img from large image
    img = card_image.filter_card_from_img(img)
    # get the title from the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    title = gray[y1:y2, x1:x2]
    flag, thresh = cv2.threshold(title, 100, 255, cv2.THRESH_BINARY)
    if is_debugging is True:
        DebugImage.write_image(thresh, '6_title_thresh', f)
        DebugImage.write_image(title, '7_title_img', f)
    title_txt = filter_title(pytesseract.image_to_data(title, output_type=Output.DICT)['text'])
    print(title_txt)
    card = card_searching_single(title_txt, img)
    if card is None:
        img_resized = cv2.resize(img, size_card)
        cards_not_found = np.concatenate((cards_not_found, img_resized), axis=0)
        amount_of_cards_not_found += 1

amount_of_cards = len(os.listdir("TestCards"))
print(f'{amount_of_cards_not_found}/{amount_of_cards} not found')
cv2.imshow('cards not found', cards_not_found)
cv2.waitKey(0)
val = input("Delete debug images? y/n: ")
if val == 'y' or val == "Y":
    shutil.rmtree('debug')
    print("debug images deleted...")
else:
    pass
