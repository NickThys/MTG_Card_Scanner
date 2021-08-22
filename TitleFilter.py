import cv2
from pytesseract import pytesseract, Output

from DebugImage import DebugImage


class TitleFilter:
    # noinspection PyShadowingNames
    def __init__(self, is_debugging=False, filterCharacters=None, y1=15, y2=90, x1=30, x2=410):
        if filterCharacters is None:
            filterCharacters = [":", "’", ";", "—", "$", "/", "_", "|"]
        self.is_debugging = is_debugging
        self.filterCharacters = filterCharacters
        self.y1 = y1
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2

    def filter_title(self, texts):
        original_title = ''
        for i in range(len(texts)):
            if texts[i] != "" and texts[i] != " ":
                if not any(ext in texts[i] for ext in self.filterCharacters):
                    if original_title != '':
                        original_title += ' '
                    original_title += texts[i]
        return original_title

    @staticmethod
    def remove_word_from_title(full_title, removeFirstWord=True):
        split_title = full_title.split()
        for word in range(len(split_title)):
            if removeFirstWord is True and word == 0:
                split_title.pop(word)
            if removeFirstWord is False and word == len(split_title) - 1:
                split_title.pop(word)
        return ' '.join(map(str, split_title))

    def get_title_from_image(self, image, file):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        title = gray[self.y1:self.y2, self.x1:self.x2]
        flag, thresh = cv2.threshold(title, 100, 255, cv2.THRESH_BINARY)
        title_text = self.filter_title(pytesseract.image_to_data(title, output_type=Output.DICT)['text'])
        if self.is_debugging is True:
            DebugImage.write_image(thresh, '6_title_thresh', file)
            DebugImage.write_image(title, '7_title_img', file)
            print(title_text)
        return title_text
