""" Check the face detector's performance on LaSOT dataset.
Find outliers/failed cases using the human bounding box.
"""
import os
import cv2
import glob
from tracker.visualize import plot_tracking, plot_tracking_cues
from ego4d_utils import *
from ADAP_json_utils import ADAP_json_to_Tracks
from AB_expert_review import collect_all_judgements
from tracker.yolox_matching import ious

if __name__ == "__main__":

    review_part2 = True
    IOU_thresh = 0.3
    Consensus_thresh = 0.75
    fps = 30 # frames per second

    base_dir = '/home/cyxu/hdd/ego4d_eval/AB_testing_results/'
    part_dir = os.path.join(base_dir, 'part_2_merged_judgements')
    # df_path = base_dir + 'part_1_progress_check/df_combined.csv'
    df_path = os.path.join(base_dir, 'part_2_progress_check', 'df2_combined.csv')
    vis_dir = os.path.join(part_dir, f'part_1_optional_boxes_vis_consensus_{Consensus_thresh}_IOU_{IOU_thresh}')
    os.makedirs(vis_dir, exist_ok=True)

    df = pd.read_csv(df_path)
    clips = df['Clip_name'].unique()

    rater_judgements_dir = os.path.join(part_dir, 'rater_judgements')
    gt_dir = os.path.join(part_dir, 'reviewed_judgements_v3')
    gt_out_dir = os.path.join(part_dir, f'reviewed_judgements_optional_consensus_{Consensus_thresh}_IOU_{IOU_thresh}')
    os.makedirs(gt_out_dir, exist_ok=True)

    # main loop to process each video clip
    for clip in clips:

        print(f'processing {clip}')
        df_clip = df[df['Clip_name'] == clip]
        optional_total = 0

        # ream video frames
        video_path = df_clip['Video_path'].values[0]
        vidcap = cv2.VideoCapture(video_path)

        # if "6c1bac5c-aead-40b8-9c17-be97d86b68d4_segment9" in clip:
        #     breakpoint()

        if review_part2:
            clip_jsons = glob.glob(os.path.join(rater_judgements_dir, f'{clip}_*.json'))
            all_tracks, boxes_per_frame = collect_all_judgements(clip_jsons, 900)
            rater_total = len(all_tracks)

        else:
            # collect all rater judesgements
            A_tracks = glob.glob(os.path.join(rater_judgements_dir, clip, 'A_*.json'))
            A_tracks, A_boxes_per_frame = collect_all_judgements(A_tracks, 900)

            B_tracks = glob.glob(os.path.join(rater_judgements_dir, clip, 'B_*.json'))
            B_tracks, B_boxes_per_frame = collect_all_judgements(B_tracks, 900)

            C_tracks = glob.glob(os.path.join(rater_judgements_dir, clip, 'C_*.json'))
            C_tracks, C_boxes_per_frame = collect_all_judgements(C_tracks, 900)

            rater_total = len(A_tracks) + len(B_tracks) + len(C_tracks)

        # load the GT json
        clip_gt = os.path.join(gt_dir, clip+'_reviewed.json')
        with open(clip_gt, 'r') as f: gt_json = json.load(f)
        gt_tracks = ADAP_json_to_Tracks(clip_gt, total_frames=900)

        box_total = np.sum([len(gt_tracks[i]) for i in range(900)])
        box_ids = list(gt_json['annotation']['shapes'].keys())

        # loop over each frame to check optional boxes
        for i in range(len(gt_json['annotation']['frames'])):
            success, img = vidcap.read()
            frm = i + 1
            optional = 0

            # if str(frm) not in gt_json['annotation']['frames']: continue

            # loop through each unique boxes
            for id in box_ids:
                
                if id not in gt_json['annotation']['frames'][str(frm)]['shapesInstances']:
                    continue

                box = gt_json['annotation']['frames'][str(frm)]['shapesInstances'][id]
                x, y, w, h = box['x'], box['y'], box['width'], box['height']
                tlbr = [x, y, x+w, y+h]

                rater_boxes = []

                if review_part2:
                    rater_boxes = [b.tlbr for b in boxes_per_frame[i]]
                else:
                    rater_boxes.extend([b.tlbr for b in A_boxes_per_frame[i]])
                    rater_boxes.extend([b.tlbr for b in B_boxes_per_frame[i]])
                    rater_boxes.extend([b.tlbr for b in C_boxes_per_frame[i]])

                matches = np.sum(np.where(ious([tlbr], rater_boxes) > IOU_thresh, 1, 0))

                # if falls under optional threshold, mark it optional
                if matches/rater_total < Consensus_thresh:
                    gt_json['annotation']['frames'][str(frm)]['shapesInstances'][id]['metadata']['shapeAnswers'][0]['answer']['values'] = True
                    # draw this box
                    cv2.rectangle(img, tlbr[:2], tlbr[2:], (0, 0, 255), 2)
                    optional += 1

            # save frame after all boxes are checked
            if optional > 0:
                optional_total += optional
                frame_name = f'{clip}_{i:05d}_optional.jpg'
                cv2.imwrite(os.path.join(vis_dir, frame_name), img)
                # print(f'{clip}_{i:05d}_optional.jpg')

        # stats
        ratio = optional_total/box_total
        print(f'{optional_total} boxes marked optional, {ratio*100:.2f}%')

        # save the GT json
        clip_optional = os.path.join(gt_out_dir, clip+'_with_optional.json')
        with open(clip_optional, 'w') as f: json.dump(gt_json, f)