""" Use all results from contributors to merge the matching boxes
and outlier boxes, so the expert can review the results.
"""

import sys
import os
import json
import cv2
import glob
import numpy as np
import pandas as pd

from tracker.byte_tracker import BYTETracker, STrack
from tracker.options import TrackOptions
from ADAP_json_utils import ADAP_json_to_Tracks, json_dump, batch_download_judgements
from tracker.visualize import plot_tracking_cues
from tracker.yolox_matching import ious


def collect_all_judgements(judgement_paths, frms_total):
    """
    Collect all the judgements from all the raters
    """
    judgements = []

    # load tracks from the json file
    for p in judgement_paths:
        tracks = ADAP_json_to_Tracks(p, total_frames=frms_total)
        judgements.append(tracks)

    # collect all boxes from all judgements
    all_boxes_per_frame = []
    for i in range(frms_total):

        this_frame = []
        for j in range(len(judgements)):
            this_frame.extend(judgements[j][i]) 
    
        all_boxes_per_frame.append(this_frame)

    return judgements, all_boxes_per_frame

def merge_judgements(consensus_threshold, iou_thres, clip_name, judgement_paths, frms_total, img_shape, base_dir):
    # options for tracker
    args = TrackOptions()
    args.tiny_ratio = 0.0
    args.cluster_iou = 0.0
    args.bypass_kalman = True
    tracker = BYTETracker(args, 0.1, 0.5, frame_rate=30)
    print(f'Merging judgements')

    judgements, all_boxes_per_frame = collect_all_judgements(judgement_paths, frms_total)

    """
    find matching boxes between many judgements, export as expert_json
    if half of the boxes agree with each other with IOU > 0.5, they're combined as one box (mean)
    otherwise, they're kept separate as outliers
    """

    expert_json = {"ableToAnnotate": "true", "annotation": {}}
    expert_json['annotation']['shapes'] = {}
    expert_json['annotation']['frames'] = {}

    reject_thres = 0.1

    for i in range(len(all_boxes_per_frame)):
        boxes = all_boxes_per_frame[i]
        boxes = np.array([b.tlbr for b in boxes])

        # output matching boxes and outlier boxes
        cluster_centroids = set()
        optional_boxes = []
        boxes_combined = []

        # cython_bbox, calculate the IOU between all boxes
        _ious = np.array(ious(boxes, boxes))
        _ious_copy = _ious.copy() # manipulate this copy to avoid repeated matches

        # first find the centroid box of each cluster
        for j in range(len(_ious)):

            init_cluster = np.where(_ious_copy[j] > iou_thres)[0]
            if len(init_cluster) / len(judgements) < reject_thres: continue

            # the centriod box has the largest IOU sum
            max_idx = np.argmax(np.sum(_ious_copy[init_cluster], axis=1))
            centroid_idx = init_cluster[max_idx]
            updated_cluster = np.where(_ious_copy[centroid_idx] > iou_thres)[0]

            consensus_ratio = len(updated_cluster) / len(judgements)
 
            # if more than certain ratio people agree, merge the boxes
            if consensus_ratio > consensus_threshold:
                cluster_centroids.add(centroid_idx)
            else:
                # automatically find the optional boxes
                if consensus_ratio > reject_thres:
                    print(f'frm: {i}, optional box: ', consensus_ratio)
                    optional_boxes.append(all_boxes_per_frame[i][centroid_idx])

            # mark these rows and columns as 0 in IOU matrix
            _ious_copy[updated_cluster] = 0
            _ious_copy[:, updated_cluster] = 0

        # find the matching boxes
        for c in cluster_centroids:
            matching_rows = np.where(_ious[c] > iou_thres)[0]

            # merge the matching boxes as mean and update b
            b = np.mean(boxes[matching_rows], axis=0).astype(np.int32)
            boxes_combined.append([*b, 1.0])

        # use tracker for consistent ID
        boxes_combined = np.array(boxes_combined)

        tracker.update(boxes_combined, img_shape, img_shape, include_FP=True)
        json_dump(expert_json, 0, frm_id=i, tracks=tracker.tracks_per_frame[i])
        
        # if len(optional_boxes) > 0 :
        #     json_dump(expert_json, 0, frm_id=i, tracks=optional_boxes, optional_attribute=True)

    # tracker.save_tracks(base_dir)

    # save the bounding box to json for adap tool
    json_out = base_dir + f'{clip_name}_merged.json'
    with open(json_out, 'w') as f:
        json.dump(expert_json, f, indent=2)

    return json_out

def read_video_imgs_metadata(video_path):
    """
    Read the video and return the images and metadata
    """
    # read all the frames for visualization
    vidcap = cv2.VideoCapture(video_path)
    success, img = vidcap.read()
    img_shape = img.shape[:2]
    imgs = []
    while success:
        imgs.append(img)
        success, img = vidcap.read()
    vidcap.release()
    return imgs, img_shape

def visualize_judgements(clip_name, imgs, merged_json, base_dir):

    # create out dirs for images and videos
    video_out_dir = base_dir + f'{clip_name}_merged_video/'
    os.makedirs(video_out_dir, exist_ok=True)

    # plot the merged boxes
    model_tracks = ADAP_json_to_Tracks(merged_json, GT=False)

    # draw the GT boxes
    print(f'exporting vidoe frames')
    for i in range(len(imgs)):
        img = imgs[i]
        m_tracks = model_tracks[i]

        frame_name = f'{i:05d}'+'.jpg'
        img_bbox = plot_tracking_cues(img, m_tracks, [], frame_id=i)
        cv2.imwrite(os.path.join(video_out_dir, frame_name), img_bbox)

    video_out_path = os.path.join(base_dir, f'{clip_name}_merged.mp4')
    os.chdir(video_out_dir)
    os.system(f"ffmpeg -framerate 30 -pattern_type glob -i '*.jpg' -c:v libx264 -pix_fmt yuv420p {video_out_path}")
    os.system(f"rm *.jpg")
    os.chdir(base_dir)
    os.system(f"rm -r {video_out_dir}")


if __name__ == "__main__":
    api_key = sys.argv[1]
    # api_key = "AXZngREC8oBJS-Hy14Wy"

    iou_thres = 0.5
    consensus_threshold = 0.33
    frms_total = 900

    base_dir = '/home/cyxu/hdd/ego4d_eval/AB_testing_results/'
    df_path = base_dir + 'part_1_progress_check/df_combined.csv'
    df_path = base_dir + 'part_2_progress_check/df2_combined.csv'

    part1_dir = base_dir + 'part_2_merged_judgements/'
    part1_merged = part1_dir + 'part_2_judgements_merged_v2_0.33/'
    os.makedirs(part1_dir, exist_ok=True)
    os.makedirs(part1_merged, exist_ok=True)

    df = pd.read_csv(df_path)
    clips = df['Clip_name'].unique()

    # main loop to process each video clip
    for clip in clips:
        # if clip != 'f55f2dfd-704f-42cf-a39e-72e2560314b3_segment6':
        #     continue

        print(f'processing {clip}')
        video_path = df[df['Clip_name']==clip]['Video_path'].values[0]
        imgs, img_shape = read_video_imgs_metadata(video_path)

        clip_dir = os.path.join(part1_dir, 'rater_judgements')
        # download judgement jsons if not already existed
        if not os.path.isdir(part1_dir):
            json_paths = batch_download_judgements(df, clip, clip_dir, api_key)
            continue
        else:
            # glob all json files in the clip directory
            json_paths = glob.glob(clip_dir + f'/{clip}*.json')
            # json_paths = [p for p in json_paths if "/B_" not in p]

        merged_json = merge_judgements(consensus_threshold, iou_thres, clip, json_paths, frms_total, img_shape, part1_merged)
        visualize_judgements(clip, imgs, merged_json, part1_merged)
