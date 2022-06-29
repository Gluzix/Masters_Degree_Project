import cv2

if __name__ == '__main__':
    vidcap = cv2.VideoCapture('test_duda.mp4')
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("images/img%d.png" % count, image)     # save frame as JPEG file
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
        if count % 10 == 0:
            count += 1
