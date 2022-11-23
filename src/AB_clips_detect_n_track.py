""" Check the face detector's performance on LaSOT dataset.
Find outliers/failed cases using the human bounding box.
"""
import os
import numpy as np
import time
import cv2
import json

from tracker.byte_tracker import BYTETracker, STrack
from tracker.byte_tracker_original import BYTETracker_original
from tracker.options import TrackOptions
from tracker.visualize import plot_tracking_cues
from tracker.basetrack import TrackState

from ego4d_utils import *
from ADAP_json_utils import json_dump

from blur_helper import FaceBlurModel


CLI_OUTPUT_DIR = "/home/cyxu/hdd/ego4d_data"
VERSION = "v1"
MANIFEST_PATH = os.path.join(CLI_OUTPUT_DIR, VERSION, 'manifest_clips.csv')
TRAIN_PATH = os.path.join(CLI_OUTPUT_DIR, VERSION, 'annotations', "av_train.json")
VAL_PATH = os.path.join(CLI_OUTPUT_DIR, VERSION, 'annotations', "av_train.json")
video_dir = os.path.join(CLI_OUTPUT_DIR, VERSION, 'clips')

# base_dir = '/home/cyxu/hdd/ego4d_eval/clips_for_AB_testing_finally_selected/shared_tasks'
base_dir = '/home/cyxu/hdd/ego4d_eval/clips_for_AB_testing_finally_selected/grouped_tasks'


if __name__ == "__main__":
    dataset = "Ego4D"
    dataset_stats = []
    candidate_clips = collect_AB_clips(base_dir)
    fps = 30

    # options for tracker
    args = TrackOptions()
    args.original_bytetracker = True
    args.no_tracker = False

    if args.original_bytetracker:
        args.track_thresh = 0.3 # was 0.6?
        args.model_thresh_low = 0.1 # hard coded to 0.1 in code
    else:
        args.model_thresh_high = 0.1
        args.model_thresh_low = 0.01

    # manual parameters
    args.debug_boxes = False
    args.min_track_lifespan = 10
    args.cluster_iou = 0.3
    args.tiny_ratio = 0.05
    # args.tiny_ratio = 0.0

    # used in Appen models, 0.4x scaling for 1080p video
    target_size, max_size = 768, 768
    # retinaFace default, 0.94x scaling for 1080p video
    # target_size, max_size = 1024, 1980

    # segment limits to reduce ADAP loading time
    fps = 30 # frames per second
    frame_count = 900 # frames
    args.detect_faces = True # only read GT if False
    args.plot_boxes = True # plot boxes on output images

    # init face detection model
    face_detector = FaceBlurModel()
    face_detector.load_model()
    root_dir = os.getcwd()

    # test each clip
    for clip_name, info in candidate_clips.items():

        # thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        thresholds = [0.3]
        for threshold in thresholds:

            if args.original_bytetracker:
                args.track_thresh = threshold
            else:
                args.model_thresh_high = threshold

            start_frame = 0  # skip the first N frames
            end_frame = start_frame + frame_count

            # path for each clip segment
            tracker_name = 'BYTE' if args.original_bytetracker else 'APPEN'

            if args.no_tracker:
                tracker_name = 'NONE'
                args.model_thresh_low = 0.8

            base_dir = f"/home/cyxu/hdd/ego4d_eval/clips_for_AB_testing_finally_selected/byte_tracker_0.3"
            os.makedirs(base_dir, exist_ok=True)

            # check if clip name already exists in current folder
            if os.path.exists(os.path.join(base_dir, clip_name+f'_{tracker_name}.json')):
                continue

            # read annotated clips instead of whole video
            av_video_path = info['video']

            pre_annotate_video(face_detector, args, clip_name, av_video_path, frame_count, start_frame, end_frame, 0, base_dir, root_dir, fps=fps, read_ego4d_gt=False, target_size=target_size, max_size=max_size)

    print('Done!')