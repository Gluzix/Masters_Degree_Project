import cv2
from PyQt6.QtCore import QDir


class FramesFromVideoMaker:

    @staticmethod
    def make_frames(path_to_movie: str, count_gap: int = 1, max_count: int = 100):

        if count_gap < 1:
            count_gap = 1

        if max_count < 1:
            max_count = 1

        if not QDir('images').exists():
            QDir().mkdir('images')

        vid_cap = cv2.VideoCapture(path_to_movie)
        success, image = vid_cap.read()
        count = 0
        help_count = 1

        while success:
            if count % count_gap == 0:
                cv2.imwrite("images/img%d.png" % help_count, image)
                help_count += 1
            success, image = vid_cap.read()
            print('Read a new frame: ', success)
            count += 1

            if help_count >= max_count:
                break
