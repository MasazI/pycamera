# -*- coding: utf-8 -*-

import os
import datetime
import cv2

def capture_camera(size=(320, 240)):
    """Capture video from camera"""

    saveCapDirPath = makeSaveDir()

    cap = cv2.VideoCapture(0)

    n = 0
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    while True:
        ret, frame = cap.read()

        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        cv2.imshow('camera-capture', frame)

        key = cv2.waitKey(1)
        # key = cv2.waitKey(1) & 0xFF


        if key == 27 or key == ord('q'):  # [ESC][q]
            break
        if key == ord('s'):               # [s]
            fName = "cap" + now + "-" + str(n).zfill(4) + ".jpg"
            path = os.path.join(saveCapDirPath, fName)
            cv2.imwrite(path, frame)
            n = n + 1

    cap.release()
    cv2.destroyAllWindows()


def makeSaveDir(dirName='saveCapImage'):
    path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(path, dirName)

    if os.path.isdir(path) == False:
        os.mkdir(path)

    return path

def _main():
    print(os.path.abspath(__file__))
    print(os.path.abspath(os.path.dirname(__file__)))
    print(cv2.__version__)

    capture_camera()


if __name__ == '__main__':
    _main()
