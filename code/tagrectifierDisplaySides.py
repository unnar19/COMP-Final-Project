import cv2
import cv2.aruco as aruco
import numpy as np
import time


def shift(list_of_items, amount):
    for _ in range(amount):
        list_of_items = np.roll(list_of_items, -1, 0)
    return list_of_items

def scale(corners, scale):
    factor = scale-1

    center = (int(np.mean([corners[0][0][0][0],
                           corners[0][0][1][0],
                           corners[0][0][2][0],
                           corners[0][0][3][0]])),
              int(np.mean([corners[0][0][0][1],
                           corners[0][0][1][1],
                           corners[0][0][2][1],
                           corners[0][0][3][1]])))

    for p_i in range(len(corners[0][0])):
        x = corners[0][0][p_i][0]
        y = corners[0][0][p_i][1]

        corners[0][0][p_i][0] = int(x+(x-center[0])*factor)
        corners[0][0][p_i][1] = int(y+(y-center[1])*factor)

    return corners

def fix_corners(corners, ids):
    if ids[0] == 2:
        corners[0][0] = shift(corners[0][0], 2)
    elif ids[0] == 3:
        corners[0][0] = shift(corners[0][0], 3)
    elif ids[0] == 4:
        corners[0][0] = shift(corners[0][0], 1)
    elif ids[0] == 5:
        corners[0][0] = shift(corners[0][0], 1)

    return corners

def find_cube_corners(frame, corners):
    tag_points_3d = np.float32([
        [-0.5, -0.5, 0], [0.5, -0.5, 0], 
        [0.5, 0.5, 0], [-0.5, 0.5, 0]
    ])

    cube_corners_3d = np.float32([
        [-0.5, -0.5, 0], [0.5, -0.5, 0],
        [0.5, 0.5, 0], [-0.5, 0.5, 0],
        [-0.5, -0.5, 0 - 1.0], [0.5, -0.5, 0 - 1.0],
        [0.5, 0.5, 0 - 1.0], [-0.5, 0.5, 0 - 1.0]
    ])

    _, rvec, tvec = cv2.solvePnP(tag_points_3d, corners[0][0], camera_matrix, dist_coeffs)

    cube, _ = cv2.projectPoints(cube_corners_3d, rvec, -tvec, camera_matrix, dist_coeffs)
    cube = np.int32(cube).reshape(-1, 2)

    return cube

def get_holo_points(cube):
    v = [1.6*(cube[3][0]-cube[0][0]), 1.6*(cube[3][1]-cube[0][1])]
    u = [1.6*(cube[2][0]-cube[1][0]), 1.6*(cube[2][1]-cube[1][1])]

    p4 = [int((cube[2][0]+cube[6][0])/2), int((cube[2][1]+cube[6][1])/2)]
    p3 = [int((cube[3][0]+cube[7][0])/2), int((cube[3][1]+cube[7][1])/2)]
    p2 = [int(p3[0]+v[0]),int(p3[1]+v[1])]
    p1 = [int(p4[0]+u[0]),int(p4[1]+u[1])]

    return [p1,p2,p3,p4]




apriltag_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
param = aruco.DetectorParameters()


image = cv2.imread('cube_texture.png', cv2.IMREAD_UNCHANGED)
image = image[:, :, :3]
(img_w,img_h) = (512,512)
img_corners = np.array([[(0,0), (img_w,0), (img_w, img_h), (0, img_h)]], dtype="float32")

holo = cv2.imread('pump.jpg', cv2.IMREAD_UNCHANGED)
holo = holo[:, :, :3]
(holo_w,holo_h) = (500,500)
holo_corners = np.array([[(0,0), (holo_w,0), (holo_w, holo_h), (0, holo_h)]], dtype="float32")

path = "COMP-Final-Project\code\pics\cube2.mp4"
cap = cv2.VideoCapture(0)
(width,height) = (int(cap.get(3)), int(cap.get(4)))

# Camera parameters (assumed)
focal_length = width
center = (width / 2, height / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype="double"
)

# Assuming no lens distortion
dist_coeffs = np.zeros((4, 1))

while(True):
    time.sleep(1/20)

    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = aruco.detectMarkers(frame, apriltag_dict, parameters=param)


    if ids is not None:
        idx = np.argmax(ids)
        corners = np.array([corners[idx]], dtype="float32")
        ids = np.array([ids[idx]])

        corners = fix_corners(corners, ids)
        corners = scale(corners,1.7)
        cube = find_cube_corners(frame,corners)
        holo_points = get_holo_points(cube)

        for i in range(len(cube)):
            cv2.circle(frame,cube[i],4,(255),4)
            cv2.putText(frame,str(i),cube[i],cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),1)

        transformation_matrix = cv2.getPerspectiveTransform(holo_corners, np.array(holo_points, dtype="float32"))
        rectified_image = cv2.warpPerspective(holo, transformation_matrix, (640, 480))
        not_black = np.all(rectified_image != 0, axis=-1)
        frame[not_black] = rectified_image[not_black]

        transformation_matrix = cv2.getPerspectiveTransform(img_corners, corners[0])
        rectified_image = cv2.warpPerspective(image, transformation_matrix, (640, 480))
        not_black = np.all(rectified_image != 0, axis=-1)
        frame[not_black] = rectified_image[not_black]

        print(cube)
        
        transformation_matrix = cv2.getPerspectiveTransform(img_corners, np.array([cube[1],cube[5],cube[4],cube[0]],dtype="float32"))
        rectified_image = cv2.warpPerspective(image, transformation_matrix, (640, 480))
        not_black = np.all(rectified_image != 0, axis=-1)
        frame[not_black] = rectified_image[not_black]

        transformation_matrix = cv2.getPerspectiveTransform(img_corners, np.array([cube[0],cube[4],cube[5],cube[1]],dtype="float32"))
        rectified_image = cv2.warpPerspective(image, transformation_matrix, (640, 480))
        not_black = np.all(rectified_image != 0, axis=-1)
        frame[not_black] = rectified_image[not_black]

        transformation_matrix = cv2.getPerspectiveTransform(img_corners, np.array([cube[3],cube[7],cube[4],cube[0]],dtype="float32"))
        rectified_image = cv2.warpPerspective(image, transformation_matrix, (640, 480))
        not_black = np.all(rectified_image != 0, axis=-1)
        frame[not_black] = rectified_image[not_black]




        transformation_matrix = cv2.getPerspectiveTransform(img_corners, np.array([cube[1],cube[5],cube[6],cube[2]],dtype="float32"))
        rectified_image = cv2.warpPerspective(image, transformation_matrix, (640, 480))
        not_black = np.all(rectified_image != 0, axis=-1)
        frame[not_black] = rectified_image[not_black]

        transformation_matrix = cv2.getPerspectiveTransform(img_corners, np.array([cube[7],cube[4],cube[5],cube[6]],dtype="float32"))
        rectified_image = cv2.warpPerspective(image, transformation_matrix, (640, 480))
        not_black = np.all(rectified_image != 0, axis=-1)
        frame[not_black] = rectified_image[not_black]

        transformation_matrix = cv2.getPerspectiveTransform(img_corners, np.array([cube[3],cube[7],cube[4],cube[0]],dtype="float32"))
        rectified_image = cv2.warpPerspective(image, transformation_matrix, (640, 480))
        not_black = np.all(rectified_image != 0, axis=-1)
        frame[not_black] = rectified_image[not_black]

        transformation_matrix = cv2.getPerspectiveTransform(img_corners, np.array([cube[2],cube[3],cube[7],cube[6]],dtype="float32"))
        rectified_image = cv2.warpPerspective(image, transformation_matrix, (640, 480))
        not_black = np.all(rectified_image != 0, axis=-1)
        frame[not_black] = rectified_image[not_black]

        transformation_matrix = cv2.getPerspectiveTransform(img_corners, corners[0])
        rectified_image = cv2.warpPerspective(image, transformation_matrix, (640, 480))
        not_black = np.all(rectified_image != 0, axis=-1)
        frame[not_black] = rectified_image[not_black]



    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
