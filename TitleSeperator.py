import sys
import numpy as np
import cv2
import pytesseract
from os import listdir, makedirs, path

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


y1, y2 = 20, 80
x1, x2 = 30, 580

if not path.isdir('titles'):
    makedirs('titles')
for f in listdir('resultCards'):
    print(f)
    file_name = 'resultCards/' + f
    img = cv2.imread(file_name)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    title = gray[y1:y2, x1:x2]

    flag, thresh = cv2.threshold(title, 100, 255, cv2.THRESH_BINARY)
    file_name = 'titles/' + f
    cv2.imwrite(file_name, thresh)
    print(pytesseract.image_to_string(thresh))
