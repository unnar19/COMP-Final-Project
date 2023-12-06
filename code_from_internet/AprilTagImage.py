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
    parser.add_argument("--image_path", type=str, default='WIN_20231206_12_05_28_Pro.jpg')  # Path to the image to project

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

        for tag in tags:
            overlay_image_onto_tag(frame, proj_image, tag.corners)

        cv.imshow('AprilTag Image Projection', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def overlay_image_onto_tag(frame, proj_image, corners):
    # Define corners of the image to project
    h, w = proj_image.shape[:2]
    pts_dst = np.array(corners)
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

if __name__ == '__main__':
    main()
