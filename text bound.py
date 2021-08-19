from os import listdir

import pytesseract
from pytesseract import Output
import cv2

testStr = [":", "’", ";", "—", "$", "/", "_"]
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def filterTitle(texts):
    title = ''
    for i in range(len(texts)):
        if texts[i] != "":
            if not any(ext in texts[i] for ext in testStr):
                title += d['text'][i] + ' '
    return title


for f in listdir('debug/6_title_img/'):
    print(f)
    file_name = 'debug/6_title_img//' + f
    img = cv2.imread(file_name)

    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(d['level'])

    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        title=filterTitle(d['text'])

    print(title)
    cv2.imshow('img', img)
    cv2.waitKey(0)
