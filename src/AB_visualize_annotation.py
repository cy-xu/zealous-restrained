""" Check the face detector's performance on LaSOT dataset.
Find outliers/failed cases using the human bounding box.
"""
import os
import cv2
from tracker.visualize import plot_tracking, plot_tracking_cues
from ego4d_utils import *
from ADAP_json_utils import ADAP_json_to_Tracks


if __name__ == "__main__":

    cwd = os.getcwd()
    base_dir = os.path.join(cwd, 'user_study_results')
    part1 = True
    bad_annotation_dict = {}
    breakpoint()

    part_dir = 'part_1_merged_judgements' if part1 else 'part_2_merged_judgements'
    
    df = os.path.join(base_dir, part_dir, 'AB_testing_precision_recall_IOU_0.5.csv')
    df = pd.read_csv(df)

    vis_dir = os.path.join(base_dir, part_dir, 'vis_clip_annotation')
    os.makedirs(vis_dir, exist_ok=True)

    # iterate through df and visualize clips with Recall lower than 0.5
    for i, row in df.iterrows():
        recall = round(row['Recall'])
        group = row['Group']
        temp_dir = os.path.join(vis_dir, 'temp_dir')
        os.makedirs(temp_dir, exist_ok=True)

        if recall < 75 and group == 'C':
            # create out dirs for images and videos
            worker = row['_worker_id']
            duration = row['Duration']
            clip = row['Clip_name']
            video_path = row['Video_path']
            rator_json = row['Rater_json']
            tenure = row['tenure']
            F1 = round(row['F1'])

            if worker not in bad_annotation_dict:
                bad_annotation_dict[worker] = 1
            else:
                bad_annotation_dict[worker] += 1
            
            rator_tracks = ADAP_json_to_Tracks(rator_json)

            video_out_path = os.path.join(vis_dir, f'Recall-{recall}_F1-{F1}_Group-{group}_tenure-{tenure}_worker-{worker}_Duration-{duration}_{clip}.mp4')

            vidcap = cv2.VideoCapture(video_path)
            success, img = vidcap.read()
            frame_counter = 0

            while success:
                frame_name = f'{frame_counter:05d}'+'.jpg'
                img_bbox = plot_tracking_cues(img, rator_tracks[frame_counter], [], frame_id=i, name='gt')
                cv2.imwrite(os.path.join(temp_dir, frame_name), img_bbox)

                frame_counter += 1
                success, img = vidcap.read()

            vidcap.release()

            # create video
            os.chdir(temp_dir)
            os.system(f"ffmpeg -framerate 30 -pattern_type glob -i '*.jpg' -c:v libx264 -pix_fmt yuv420p {video_out_path}")
            os.system(f"rm *.jpg")
            # change back to root dir
            os.chdir(cwd)

    print(bad_annotation_dict)

    print('Done!')
