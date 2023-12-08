#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
from pupil_apriltags import Detector

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--families", type=str, default='tag36h11')
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--quad_decimate", type=float, default=2.0)
    parser.add_argument("--quad_sigma", type=float, default=0.0)
    parser.add_argument("--refine_edges", type=int, default=1)
    parser.add_argument("--decode_sharpening", type=float, default=0.25)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--image_path", type=str, default='jungle.jpg')  # Path to the image to project

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    image_path = args.image_path

    # Load the image to be projected
    proj_image = cv.imread(image_path)
    if proj_image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    at_detector = Detector(
        families=args.families,
        nthreads=args.nthreads,
        quad_decimate=args.quad_decimate,
        quad_sigma=args.quad_sigma,
        refine_edges=args.refine_edges,
        decode_sharpening=args.decode_sharpening,
        debug=args.debug,
    )

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        tags = at_detector.detect(
            gray_frame,
            estimate_tag_pose=False,
            camera_params=None,
            tag_size=None,
        )

        tag_size = 1.0  # Size of the tag in your desired units
        scale_factor = 1.7  # Adjust this to change the size of the cube

        for tag in tags:
            if tag.tag_id == 0:
                #print(tag)
                overlay_image_over_tag(frame, proj_image, tag)
                draw_cube(frame, np.array(tag.corners, dtype='float32'), tag_size, scale_factor)
        
        cv.imshow('AprilTag Image Projection', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def overlay_image_over_tag(frame, proj_image, tag):
    # Define corners of the image to project
    h, w = proj_image.shape[:2]

    # Get the corners of the detected tag
    corners = tag.corners
    top_left, top_right, bottom_right, bottom_left = corners

    # Calculate the height of the tag (average of the two sides)
    tag_height = np.mean([np.linalg.norm(top_left - bottom_left), np.linalg.norm(top_right - bottom_right)])

    # Calculate new corners above the tag
    shift_up = tag_height * 2  # Adjust this factor as needed to move the image above the tag
    new_top_left = top_left + np.array([0, -shift_up])
    new_top_right = top_right + np.array([0, -shift_up])
    new_bottom_right = bottom_right + np.array([0, -shift_up])
    new_bottom_left = bottom_left + np.array([0, -shift_up])
    pts_dst = np.array([new_top_left, new_top_right, new_bottom_right, new_bottom_left])

    # Define source points from the image
    pts_src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=float)

    # Compute homography
    homography, _ = cv.findHomography(pts_src, pts_dst)

    # Warp the image
    warped_image = cv.warpPerspective(proj_image, homography, (frame.shape[1], frame.shape[0]))

    # Prepare mask for overlay
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv.fillConvexPoly(mask, np.int32(pts_dst), (255, 255, 255), cv.LINE_AA)

    # Erode mask to avoid edge artifacts
    mask = cv.erode(mask, np.ones((3, 3), np.uint8))

    # Overlay the image
    frame[mask > 0] = warped_image[mask > 0]
    

def draw_cube(frame, corners, tag_size, scale_factor):
    # Define the 3D points of the AprilTag (assuming it lies in the XY plane with Z=0)
    tag_half = tag_size / 2.0
    tag_points_3d = np.float32([
        [-tag_half, -tag_half, 0], [tag_half, -tag_half, 0], 
        [tag_half, tag_half, 0], [-tag_half, tag_half, 0]
    ])

    # Define the 3D points of the cube (scaled)
    cube_size = tag_size * scale_factor
    half_size = cube_size / 2.0
    cube_corners_3d = np.float32([
        [-half_size, -half_size, 0], [half_size, -half_size, 0], [half_size, half_size, 0], [-half_size, half_size, 0],
        [-half_size, -half_size, -cube_size], [half_size, -half_size, -cube_size], [half_size, half_size, -cube_size], [-half_size, half_size, -cube_size]
    ])

    # Camera parameters (assumed)
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))

    # Estimate the pose of the AprilTag
    ret, rvec, tvec = cv.solvePnP(tag_points_3d, corners, camera_matrix, dist_coeffs)

    # Project the 3D points of the cube to the image plane
    imgpts, _ = cv.projectPoints(cube_corners_3d, rvec, tvec, camera_matrix, dist_coeffs)

    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw the base
    cv.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), -3)

    # Draw the pillars
    for i, j in zip(range(4), range(4, 8)):
        cv.line(frame, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # Draw the top
    cv.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 3)

if __name__ == '__main__':
    main()