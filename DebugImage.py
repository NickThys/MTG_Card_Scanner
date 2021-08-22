from os import makedirs
from sys import path

import cv2


class DebugImage:

    @staticmethod
    def write_image(debug_img, extra_path, debug_filename):
        f_path = "debug/" + extra_path + "/"
        if not path.isdir(f_path):
            makedirs(f_path)
        cv2.imwrite(f_path + debug_filename, debug_img)
