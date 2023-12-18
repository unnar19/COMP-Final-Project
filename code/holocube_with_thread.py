import cv2
import os
import time
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np
import cv2.aruco as aruco
import threading

def find_center_corner(cube):
    # Assuming 'cube' contains the 8 corners of the cube (0 to 7)
    # Calculate the average (geometric center) of all corners
    center_x = np.mean([corner[0] for corner in cube])
    center_y = np.mean([corner[1] for corner in cube])

    # Find which of the first four corners (0, 1, 2, 3) is closest to the average
    distances = [np.linalg.norm(cube[i] - [center_x, center_y]) for i in range(4)]
    center_corner_index = np.argmin(distances)

    return center_corner_index

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

def find_cube_corners(corners):
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

def get_holo_points(cube, tag_id):
    heig = 1
    cube2 = []
    if tag_id == 4: # Right
        cube2 = [cube[4],cube[0],cube[3],cube[7],cube[5],cube[1],cube[2],cube[6]]
    elif tag_id == 5: # Left
        cube2 = [cube[1],cube[5],cube[6],cube[2],cube[0],cube[4],cube[7],cube[3]]
    elif tag_id == 3: # Back
        cube2 = [cube[5],cube[4],cube[7],cube[6],cube[1],cube[0],cube[3],cube[2]]
    else:
        cube2 = cube

    if tag_id in [2,3,4,5]:

        v = [heig*(cube2[3][0]-cube2[0][0]), heig*(cube2[3][1]-cube2[0][1])]
        u = [heig*(cube2[2][0]-cube2[1][0]), heig*(cube2[2][1]-cube2[1][1])]

        p4 = [int((cube2[2][0]+cube2[6][0])/2), int((cube2[2][1]+cube2[6][1])/2)]
        p3 = [int((cube2[3][0]+cube2[7][0])/2), int((cube2[3][1]+cube2[7][1])/2)]
        p2 = [int(p3[0]+v[0]),int(p3[1]+v[1])]
        p1 = [int(p4[0]+u[0]),int(p4[1]+u[1])]
    else:
        if tag_id == 1:
            v = [heig*(cube2[3][0]-cube2[7][0]), heig*(cube2[3][1]-cube2[7][1])]
            u = [heig*(cube2[0][0]-cube2[4][0]), heig*(cube2[0][1]-cube2[4][1])]

            p4 = [int((cube2[1][0]+cube2[0][0])/2), int((cube2[1][1]+cube2[0][1])/2)]
            p3 = [int((cube2[3][0]+cube2[2][0])/2), int((cube2[3][1]+cube2[2][1])/2)]
            p2 = [int(p3[0]+v[0]),int(p3[1]+v[1])]
            p1 = [int(p4[0]+u[0]),int(p4[1]+u[1])]
        else:
            v = [heig*(cube2[2][0]-cube2[6][0]), heig*(cube2[2][1]-cube2[6][1])]
            u = [heig*(cube2[1][0]-cube2[5][0]), heig*(cube2[1][1]-cube2[5][1])]

            p4 = [int((cube2[7][0]+cube2[6][0])/2), int((cube2[7][1]+cube2[6][1])/2)]
            p3 = [int((cube2[4][0]+cube2[5][0])/2), int((cube2[4][1]+cube2[5][1])/2)]
            p2 = [int(p3[0]-v[0]),int(p3[1]-v[1])]
            p1 = [int(p4[0]-u[0]),int(p4[1]-u[1])]

    result = [p1,p2,p3,p4]
    if tag_id in [4,5]:
        result = [p2,p1,p4,p3]

    return result

def glow_effect(no_bg):
    # increase contrast
    gray = cv2.cvtColor(no_bg, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    no_bg = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    # add scan lines
    pattern = np.zeros_like(no_bg, dtype=np.uint8)
    pattern[::5, :, :] = no_bg[::5, :, :]/7
    no_bg = cv2.addWeighted(no_bg, 1, pattern, 0.6, 0)
    

    # make subject cyan
    no_bg[:, :, 1] = no_bg[:, :, 1]/2
    no_bg[:, :, 2] = 0

    # add glow
    glow = cv2.GaussianBlur(no_bg, (101, 101), 0)
    black = np.all(no_bg == 0, axis=-1)
    no_bg[black] = glow[black]

    return no_bg

def draw_cube_texture(corners):
    transformation_matrix = cv2.getPerspectiveTransform(img_corners, np.array(corners,dtype="float32"))
    rectified_image = cv2.warpPerspective(image, transformation_matrix, (640, 480))
    not_black = np.all(rectified_image != 0, axis=-1)
    frame1[not_black] = rectified_image[not_black]

class FrameCaptureThread:
    def __init__(self, cap):
        self.cap = cap
        self.latest_frame = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.update)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame

    def stop(self):
        self.running = False
        self.thread.join()

try:

    # laptop cam
    print("get pc camera")
    webcam = cv2.VideoCapture(0)
    (widthweb,heightweb) = (int(webcam.get(3)), int(webcam.get(4)))

    # Mobile cam
    print("get mobile camera")
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp;buffer_size;2'
    #cap = cv2.VideoCapture('rtsp://10.1.19.54/video', cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture('rtsp://10.1.19.9/video', cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    (width,height) = (int(cap.get(3)), int(cap.get(4)))

    # Start the frame capture thread
    print("start thread")
    frame_thread = FrameCaptureThread(cap)
    frame_thread.start()

    print("init april tag detector")
    apriltag_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    param = aruco.DetectorParameters()

    # Cube texture
    image = cv2.imread('COMP-Final-Project\code\pics\AI_texture.png', cv2.IMREAD_UNCHANGED)
    image = image[:, :, :3]
    (img_w,img_h) = (1024,1024)
    img_corners = np.array([[(0,0), (img_w,0), (img_w, img_h), (0, img_h)]], dtype="float32")

    #human_weights = cv2.CascadeClassifier('COMP-Final-Project\code\haarcascade_fullbody.xml')

    background_image = cv2.resize(cv2.imread("COMP-Final-Project/code/pics/black.png"), (640, 480)) 
    print("init background remover")
    segmentor = SelfiSegmentation(0)

    # Camera parameters (assumed)
    focal_length = widthweb
    center = (widthweb / 2, heightweb / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype="double"
    )

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))

    x = 0
    y = 0
    w = width
    h = height
    bounds = []
    avg = (0,0,width,height)
    running_avg = 5

    last_time = time.time()
    ret, frame = cap.read()

    # Measuring flicker
    box_in_frame = False
    total_frames = 0
    AR_active = 0

    while(True):

        now = time.time()
        # Read and discard frames in the buffer to get the most recent frame
        frame = frame_thread.get_latest_frame()
        if frame is None:
            continue  # Skip the rest of the loop if no frame is retrieved

        frame = cv2.resize(frame,(853,480))
        frame = frame[0:0 + 480, 106:106 + 640]

        cv2.imshow("extracam", frame)

        ret, frame1 = webcam.read()
        frame1 = cv2.resize(frame1, (640, 480))
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(frame1, apriltag_dict, parameters=param)

        # cut background
        no_bg = segmentor.removeBG(frame, background_image, cutThreshold=0.05)
        # add glow effect
        no_bg = glow_effect(no_bg)
        
        cv2.imshow("Cropped BG removed", no_bg)

        if box_in_frame:
            total_frames += 1
        
        if ids is not None:
            box_in_frame = True
            AR_active += 1

            idx = np.argmax(ids)
            corners = np.array([corners[idx]], dtype="float32")
            ids = np.array([ids[idx]])

            corners = fix_corners(corners, ids)
            corners = scale(corners,1.9)
            cube = find_cube_corners(corners)
            center_corner_index = find_center_corner(cube)
            
            holo_points = get_holo_points(cube,ids[0])
            (holo_w,holo_h) = (640,480)
            holo_corners = np.array([[(0,0), (holo_w,0), (holo_w, holo_h), (0, holo_h)]], dtype="float32")

            transformation_matrix = cv2.getPerspectiveTransform(holo_corners, np.array(holo_points, dtype="float32"))
            rectified_hologram = cv2.warpPerspective(no_bg, transformation_matrix, (640, 480))
            frame1 = cv2.add(rectified_hologram, frame1)
            
            # Draw only the faces connected to the center corner
            if center_corner_index == 0:
                draw_cube_texture([cube[3],cube[7],cube[4],cube[0]])
                frame1 = cv2.add(rectified_hologram, frame1)
                draw_cube_texture([cube[0],cube[4],cube[5],cube[1]])
            elif center_corner_index == 1:
                draw_cube_texture([cube[1],cube[5],cube[6],cube[2]])
                frame1 = cv2.add(rectified_hologram, frame1)
                draw_cube_texture([cube[1],cube[5],cube[4],cube[0]])
            elif center_corner_index == 2:
                draw_cube_texture([cube[2],cube[6],cube[7],cube[3]])
                frame1 = cv2.add(rectified_hologram, frame1)
                draw_cube_texture([cube[2],cube[1],cube[5],cube[6]])
            elif center_corner_index == 3:
                draw_cube_texture([cube[3],cube[7],cube[6],cube[2]])
                frame1 = cv2.add(rectified_hologram, frame1)
                draw_cube_texture([cube[3],cube[7],cube[4],cube[0]])
            draw_cube_texture(corners[0])
            
        last_time = time.time()
        cv2.imshow('frame',frame1)
        print(1/(now-last_time))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            flicker = (1- AR_active/total_frames)*100.0
            print('Flicker: ',flicker,'%')
            time.sleep(0.5)
            break

    cap.release()
    frame_thread.stop()
    cv2.destroyAllWindows()
except:
    cap.release()
    frame_thread.stop()
    cv2.destroyAllWindows()