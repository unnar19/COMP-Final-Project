import cv2
import cv2.aruco as aruco
from time import sleep

apriltag_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
param = aruco.DetectorParameters()

cap = cv2.VideoCapture("COMP-Final-Project\code\pics\cube.mp4")
(width,height) = (int(cap.get(3)), int(cap.get(4)))

while(True):
    sleep(1/40)

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco.detectMarkers(frame, apriltag_dict, parameters=param)

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
