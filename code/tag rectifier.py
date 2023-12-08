import cv2
import cv2.aruco as aruco
import numpy as np
from time import sleep


apriltag_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
param = aruco.DetectorParameters()
image = cv2.imread('COMP-Final-Project/code/pics/cube_texture.png', cv2.IMREAD_UNCHANGED)
(img_w,img_h) = (512,512)
img_corners = np.array([[(0,0), (img_w,0), (img_w, img_h), (0, img_h)]], dtype="float32")

path = "COMP-Final-Project\code\pics\cube2.mp4"
cap = cv2.VideoCapture(0)
(width,height) = (int(cap.get(3)), int(cap.get(4)))

while(True):
    sleep(1/20)

    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco.detectMarkers(frame, apriltag_dict, parameters=param)


    if ids is not None:
        idx = np.argmin(ids)
        corners = np.array([corners[idx]], dtype="float32")
        ids = np.array([ids[idx]])

        transformation_matrix = cv2.getPerspectiveTransform(img_corners, corners[0])
        rectified_image = cv2.warpPerspective(image, transformation_matrix, (640, 480))
        not_black = np.all(rectified_image != 0, axis=-1)
        frame[not_black] = rectified_image[not_black]

        for i, corner in enumerate(corners[0][0]):
            cv2.putText(frame,str(i),(int(corner[0]),int(corner[1])),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),1)

        #aruco.drawDetectedMarkers(frame, corners, ids)


    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
