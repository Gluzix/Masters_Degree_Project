import cv2

if __name__ == '__main__':
    vidcap = cv2.VideoCapture('donald_trump_movie.mp4')
    success, image = vidcap.read()
    count = 0
    help_count = 1
    while success:
        cv2.imwrite("images/img%d.png" % help_count, image)
        help_count += 1
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
