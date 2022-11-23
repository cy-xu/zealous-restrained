""" Check the face detector's performance on LaSOT dataset.
Find outliers/failed cases using the human bounding box.
"""
import os
import cv2
from tracker.visualize import plot_tracking, plot_tracking_cues
from ego4d_utils import *
from ADAP_json_utils import ADAP_json_to_Tracks


if __name__ == "__main__":

    base_dir = '/home/cyxu/hdd/ego4d_eval/AB_testing_results/'
    part1_df = base_dir + 'part_1_merged_judgements/AB_testing_precision_recall_IOU_0.5.csv'
    df = pd.read_csv(part1_df)

    vis_dir = base_dir + 'vis_for_CHI_video/'
    os.makedirs(vis_dir, exist_ok=True)

    clips = df['Clip_name'].unique()
    # video_paths = df['Video_path'].unique()
    # zealous_jsons = df['Appen_json'].unique()
    # restrained_jsons = df['Byte_json'].unique()
    # gt_jsons = df['GT_json'].unique()

    for clip in clips:
        # create out dirs for images and videos
        video_out_dir = os.path.join(vis_dir, clip)
        os.makedirs(video_out_dir, exist_ok=True)

        video_path = df[df['Clip_name'] == clip]['Video_path'].values[0]
        zealous_json = df[df['Clip_name'] == clip]['Appen_json'].values[0]
        restrained_json = df[df['Clip_name'] == clip]['Byte_json'].values[0]
        gt_json = df[df['Clip_name'] == clip]['GT_json'].values[0]

        zealous_tracks = ADAP_json_to_Tracks(zealous_json)
        restrained_tracks = ADAP_json_to_Tracks(restrained_json)
        gt_tracks = ADAP_json_to_Tracks(gt_json, GT=True)

        all_tracks = [zealous_tracks, restrained_tracks, gt_tracks]
        all_names = ['zealous', 'restrained', 'gt']

        vidcap = cv2.VideoCapture(video_path)
        success, img = vidcap.read()

        fps = 30 # frames per second
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        imgs = []
        while success:
            imgs.append(img)
            success, img = vidcap.read()
        vidcap.release()

        for i in range(3):
            tracks = all_tracks[i]
            name = all_names[i]

            for i in range(len(imgs)):
                print(f'frame {i}')
                img = imgs[i].copy()

                frame_name = f'{i:05d}'+'.jpg'
                img_bbox = plot_tracking_cues(img, tracks[i], [], frame_id=i, name=name)
                cv2.imwrite(os.path.join(video_out_dir, frame_name), img_bbox)
        
            # create video
            video_out_path = os.path.join(video_out_dir, f'{clip}_{name}.mp4')
            os.chdir(video_out_dir)
            os.system(f"ffmpeg -framerate 30 -pattern_type glob -i '*.jpg' -c:v libx264 -pix_fmt yuv420p {video_out_path}")
            os.system(f"rm *.jpg")

    print('Done!')
