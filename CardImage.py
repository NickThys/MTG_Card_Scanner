import cv2
import numpy as np
from DebugImage import DebugImage


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

    def filter_card_from_img(self, image, file):
        square = self.__find_square(image, file)
        im_debug = cv2.drawContours(image.copy(), [square], -1, (0, 255, 0), 3)
        if self.is_debugging is True:
            DebugImage.write_image(im_debug, "4_selected_contour", file)
        # noinspection PyTypeChecker
        image = self.__fix_perspective(image, square)
        # set the image right
        if self.is_debugging is True:
            DebugImage.write_image(image, '5_perspective_fix', file)
        return image
