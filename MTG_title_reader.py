import os
import shutil
from urllib.request import urlopen
import mtgsdk
import numpy as np
import cv2
from os import listdir
import pytesseract

from CardImage import CardImage
from TitleFilter import TitleFilter


def url_to_image(url, readFlag=cv2.IMREAD_GRAYSCALE):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)

    # return the image
    return image


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


def card_searching_single(txt_title, card_img, filter_text):
    card_list = []
    if txt_title != "":
        cards = get_cards(txt_title)
        for card_index in range(len(cards)):
            card_list.append(cards[card_index])
        if len(cards) == 0:
            print(txt_title + " not found in the db"
                              "\n try to remove last word ... ")
            title_r = filter_text.remove_word_from_title(txt_title, False)
            cards = get_cards(title_r)

            for card_index in range(len(cards)):
                print(cards[card_index].name + " : " + cards[card_index].set)
                # if card == len(cards) - 1:
                card_list.append(cards[card_index])
            if len(cards) == 0:
                print(title_r + " not found in the db"
                                "\n try to remove First word ... ")
                title_r = filter_text.remove_word_from_title(txt_title)
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
os.putenv('TESSDATA_PREFIX', 'C:\\Program Files\\Tesseract-OCR\\tessdata\\mtg.traineddata')
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# variables
y1, y2 = 15, 90
x1, x2 = 30, 410
testStr = [":", "’", ";", "—", "$", "/", "_", "|"]

amount_of_cards_not_found = 0

back_mtg_card = cv2.imread("Templates/Magic_card_back.jpg")
size_card = [back_mtg_card.shape[1], back_mtg_card.shape[0]]
cards_not_found = back_mtg_card

is_debugging = False
card_image = CardImage(is_debugging)
text_filter = TitleFilter(is_debugging)
for f in listdir("TestCards"):
    print(f)
    filename = "TestCards/" + f
    img = cv2.imread(filename)
    # get fixed img from large image
    img = card_image.filter_card_from_img(img, f)
    # get the title from the image

    title_txt = text_filter.get_title_from_image(img, f)

    card = card_searching_single(title_txt, img, text_filter)
    if card is None:
        img_resized = cv2.resize(img, size_card)
        cards_not_found = np.concatenate((cards_not_found, img_resized), axis=0)
        amount_of_cards_not_found += 1

amount_of_cards = len(os.listdir("TestCards"))
print(f'{amount_of_cards_not_found}/{amount_of_cards} not found')
cv2.imshow('cards not found', cards_not_found)
cv2.waitKey(0)
if is_debugging is True:
    val = input("Delete debug images? y/n: ")
    if val == 'y' or val == "Y":
        shutil.rmtree('debug')
        print("debug images deleted...")
    else:
        pass
