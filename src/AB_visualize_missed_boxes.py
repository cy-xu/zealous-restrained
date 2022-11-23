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
from tracker.yolox_matching import ious, linear_assignment

if __name__ == "__main__":

    IOU_thresh = 0.5 # higher IOU threshold here because the ground truth is more precise
    Diff_thresh = 0.3
    Optional_thresh = 0.25
    Consensus_thresh = 0.51
    fps = 30 # frames per second

    base_dir = '/home/cyxu/hdd/ego4d_eval/AB_testing_results/'
    part1_df = base_dir + 'part_1_progress_check/df_combined.csv'
    vis_dir = base_dir + 'part_1_missed_boxes_visualize/Group_C_EASY/'
    os.makedirs(vis_dir, exist_ok=True)

    df = pd.read_csv(part1_df)
    df = df[df['Outlier'] == 0]
    df = df[df['Difficulty'] == 'Easy']
    clips = df['Clip_name'].unique()

    rater_judgements_dir = base_dir + 'part_1_merged_judgements/rater_judgements/'
    gt_dir = base_dir + 'part_1_merged_judgements/reviewed_judgements_v2/'

    # main loop to process each video clip
    for clip in clips:

        print(f'processing {clip}')
        df_clip = df[df['Clip_name'] == clip]

        # ream video frames
        video_path = df_clip['Video_path'].values[0]
        vidcap = cv2.VideoCapture(video_path)

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

        rater_total = len(A_tracks) + len(B_tracks) + len(C_tracks)
        box_total = np.sum([len(gt_tracks[i]) for i in range(900)])
        box_ids = list(gt_json['annotation']['shapes'].keys())

        # loop over each frame to check optional boxes
        for i in range(900):
            success, img = vidcap.read()
            frm = i + 1
            A_missed, B_missed, C_missed = 0, 0, 0

            gt_tlbrs = []
            # loop through each unique boxes
            for id in box_ids:

                if id not in gt_json['annotation']['frames'][str(frm)]['shapesInstances']:
                    continue

                box = gt_json['annotation']['frames'][str(frm)]['shapesInstances'][id]

                # box is optional, ignore
                if box['metadata']['shapeAnswers'][0]['answer']['values']:
                    continue
                else:
                    x, y, w, h = box['x'], box['y'], box['width'], box['height']
                    tlbr = [x, y, x+w, y+h]
                    gt_tlbrs.append(tlbr)

            if len(gt_tlbrs) == 0:
                continue

            A_tlbrs = [[int(x) for x in b.tlbr] for b in A_boxes_per_frame[i]]
            B_tlbrs = [[int(x) for x in b.tlbr] for b in B_boxes_per_frame[i]]
            C_tlbrs = [[int(x) for x in b.tlbr] for b in C_boxes_per_frame[i]]

            # # check missing boxes in A
            # for rater_tlbr in A_tlbrs:
            #     _iou = ious(gt_tlbrs, [rater_tlbr])
            #     matches, u_gt, u_detection = linear_assignment(1-_iou, thresh=IOU_thresh)
            #     if len(u_detection) > 0:
            #         cv2.rectangle(img, rater_tlbr[:2], rater_tlbr[2:], (240, 0, 0), 1)
            #         A_missed += 1

            # # check missing boxes in B
            # for rater_tlbr in B_tlbrs:
            #     _iou = ious(gt_tlbrs, [rater_tlbr])
            #     matches, u_gt, u_detection = linear_assignment(1-_iou, thresh=IOU_thresh)
            #     if len(u_detection) > 0:
            #         cv2.rectangle(img, rater_tlbr[:2], rater_tlbr[2:], (0, 160, 255), 1)
            #         B_missed += 1

            # check missing boxes in C
            for rater_tlbr in C_tlbrs:
                _iou = ious(gt_tlbrs, [rater_tlbr])
                matches, u_gt, u_detection = linear_assignment(1-_iou, thresh=IOU_thresh)
                if len(u_detection) > 0:
                    cv2.rectangle(img, rater_tlbr[:2], rater_tlbr[2:], (0, 240, 0), 1)
                    C_missed += 1

            # save frame after all boxes are checked
            if np.sum([A_missed, B_missed, C_missed]) > 15:
                for gt_box in gt_tlbrs:
                    cv2.rectangle(img, gt_box[:2], gt_box[2:], (255, 255, 255), 2)

                cv2.putText(img, f'frame: {frm}, A missed: {A_missed}, B missed: {B_missed}, C missed: {C_missed}',
                            (0, int(15 * 2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

                frame_name = f'{clip}_{i:05d}_misssed.jpg'
                cv2.imwrite(os.path.join(vis_dir, frame_name), img)
                # print(f'{clip}_{i:05d}_optional.jpg')
