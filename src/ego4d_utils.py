# ego4d import
import os
import cv2
import json
import pandas as pd
import math
import av
import uuid
import numpy as np
import time
from nb_video_utils import _get_frames

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tracker.byte_tracker import BYTETracker, STrack
from tracker.byte_tracker_original import BYTETracker_original
from tracker.basetrack import TrackState
from tracker.visualize import plot_tracking_cues
from ADAP_json_utils import json_dump


def vid_df_des(df):
    return f"#{len(df)} {df.duration_sec.sum()/60/60:.1f}h"
def vid_des(videos):
    return f"#{len(videos)} {sum((x.duration_sec for x in videos))/60/60:.1f}h"
def deserialize_str_list(list_: str):
    list_ = list_[1:-2]
    items = list_.split("', '")
    return list(map(lambda z: z.strip("'"), items))
def to_1D(series):
    return pd.Series([x for _list in series for x in _list])



def pre_annotate_video(face_detector, args, av_clip_uid, av_video_path, frame_count, start_frame, end_frame, segment, base_dir, root_dir, fps=30, read_ego4d_gt=False, target_size=768, max_size=768):

    tic = time.time()

    vidcap = cv2.VideoCapture(av_video_path)
    success, img = vidcap.read()

    # stats_file = os.path.join(base_dir, 'dataset_stats.npy')
    # detection_file = os.path.join(base_dir, 'detections.npy')

    assert success, f"Failed to read video {av_video_path}"

    img_shape = img.shape[:2]
    # img_h, img_w = img_shape[1], img_shape[0]

    if args.original_bytetracker:
        tracker = BYTETracker_original(args, frame_rate=fps)
    else:
        tracker = BYTETracker(args, args.model_thresh_high, args.model_thresh_low, frame_rate=fps)

    if read_ego4d_gt:
        gt_tracks = read_clip_GT(av_clip_uid)
    else:
        gt_tracks = []

    # new object for each video clip
    sequence = {}
    sequence['path'] = av_video_path
    frame_name = os.path.basename(av_video_path)
    sequence['name'] = os.path.splitext(frame_name)[0]
    sequence['imgs'] = []
    sequence['blurred_imgs'] = []
    sequence['filenames'] = []
    sequence['frame_scores'] = {}

    # detection and tracker
    sequence['detection'] = []
    sequence['track_gt'] = []
    sequence['track_forward'] = []

    # new JSON format for ADAP tool
    # https://success.appen.com/hc/en-us/articles/360013100332-How-to-Interpreting-results-from-a-Video-Bounding-Box-Job
    sequence['json'] = {"ableToAnnotate": "true", "annotation": {}}
    sequence['json']['annotation']['shapes'] = {}
    sequence['json']['annotation']['frames'] = {}

    # new JSON format for ADAP tool
    sequence['gt_json'] = {"ableToAnnotate": "true", "annotation": {}}
    sequence['gt_json']['annotation']['shapes'] = {}
    sequence['gt_json']['annotation']['frames'] = {}


    # first pass, face detection and GT reading
    for i in range(end_frame):
        print(f'frame {i}')

        # read each frame
        if i > 0:
            success, img = vidcap.read()
        if not success:
            frame_count = i
            continue # end of video

        if i < start_frame: continue # skip the first N frames

        sequence['imgs'].append(img)

        _, face_dets = face_detector.blur_faces_in_image(img.copy(), scales=[target_size, max_size],threshold=args.model_thresh_low)

        if args.no_tracker:
            trakcs = []
            for jj in range(len(face_dets)):
                det = face_dets[jj]
                t = STrack(STrack.tlbr_to_tlwh(det[:4]), det[4])
                t.track_uuid = f'{i}_{jj}'
                trakcs.append(t)

            json_dump(sequence['json'], 0, i, trakcs)
            continue

        tracker.update(face_dets, img_shape, img_shape)

    # finish early if no trakcer is needed
    if args.no_tracker:
        # save the bounding box to json for adap tool
        json_path = os.path.join(base_dir, sequence['name']+'_NONE.json')
        with open(json_path, 'w') as f: json.dump(sequence['json'], f, indent=2)

        toc = time.time()
        print(f"Time taken: {toc-tic} seconds")

        vidcap.release()
        return 0

    if not args.original_bytetracker:
        # remove short tracks
        tracker.remove_short_tracks(min_len=args.min_track_lifespan)
        # tracker_forward.mark_tiny_tracks(min_size=args.tiny_ratio)
        tracker.save_tracks(base_dir)

    # third pass, read GT
    for i in range(end_frame):
        if i < start_frame: continue
        tracks = []
        if i in gt_tracks:
            b = gt_tracks[i]
            for k in range(len(b)):
                pid = int(b[k][0])
                x1 = int(b[k][1])
                y1 = int(b[k][2])
                x2 = int(b[k][3])
                y2 = int(b[k][4])
                # each person
                t = STrack([x1, y1, x2-x1, y2-y1], 1.0)
                t.state = TrackState.GT
                t.track_uuid = "GT_" + str(pid)
                tracks.append(t)
            sequence['track_gt'].append(tracks)
        else:
            sequence['track_gt'].append([])

    # create out dirs for images and videos
    out_dir = os.path.join(base_dir, sequence['name']+'_clip')
    os.makedirs(out_dir, exist_ok=True)
    out_dir_bbox = os.path.join(base_dir, sequence['name']+'_bbox')
    os.makedirs(out_dir_bbox, exist_ok=True)

    clean_vid = os.path.join(base_dir, sequence['name']+f'_segment{segment}'+'.mp4')
    bbox_vid = os.path.join(base_dir, sequence['name']+f'_segment{segment}_bbox'+'.mp4')

    # fourth pass, compute the final tracks and statistics
    for i in range(len(sequence['imgs'])):

        # draw the GT boxes
        img = sequence['imgs'][i]
        track_forward = tracker.tracks_per_frame[i]
        track_gt = sequence['track_gt'][i]

        frame_name = sequence['name']+'_'+f'{i:05d}'+'.jpg'
        # cv2.imwrite(os.path.join(out_dir, frame_name), img)

        if args.plot_boxes:
            img_bbox = plot_tracking_cues(img, track_forward, track_gt, frame_id=i)
            cv2.imwrite(os.path.join(out_dir_bbox, frame_name), img_bbox)

        # save to json for adap
        json_dump(sequence['json'], 0, i, track_forward)
        json_dump(sequence['gt_json'], 0, i, track_gt)

        # save the bounding box to json for adap tool
        json_path = os.path.join(base_dir, sequence['name']+'.json')
        with open(json_path, 'w') as f:
            json.dump(sequence['json'], f, indent=2)

        if read_ego4d_gt:
            gt_json_path = os.path.join(base_dir, sequence['name']+'_gt.json')
            with open(gt_json_path, 'w') as f:
                json.dump(sequence['gt_json'], f, indent=2)

    toc = time.time()
    print(f"Time taken: {toc-tic} seconds")

    vidcap.release()
    # convert images to h264 video, opencv direct h264 output is not working due to liscense
    os.chdir(out_dir)
    os.system(f"ffmpeg -framerate 30 -pattern_type glob -i '*.jpg' -c:v libx264 -pix_fmt yuv420p {clean_vid}")

    if args.plot_boxes:
        os.chdir(out_dir_bbox)
        os.system(f"ffmpeg -framerate 30 -pattern_type glob -i '*.jpg' -c:v libx264 -pix_fmt yuv420p {bbox_vid}")

    # change back to original directory
    os.chdir(root_dir)
    os.system(f"rm -r {out_dir}")
    os.system(f"rm -r {out_dir_bbox}")



def collect_AB_clips(base_dir):
    # walk through all the directories to collect all video clips (36 in total)
    clips = dict()
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.mp4'):
                clip_name = file.split('.')[0]
                if clip_name in clips:
                    continue
                clips[clip_name] = dict()
                clips[clip_name]['key'] = root.split('/')[-1]
                clips[clip_name]['video'] = os.path.join(root, clip_name+'.mp4')
                clips[clip_name]['appen'] = os.path.join(root, clip_name+'_APPEN_fix.json')
                clips[clip_name]['byte'] = os.path.join(root, clip_name+'_BYTE.json')
                clips[clip_name]['ego4d'] = os.path.join(root, clip_name+'_EGO4D.json')
                clips[clip_name]['retina_0.3'] = os.path.join(root, clip_name+'_0.3.json')
                clips[clip_name]['retina_0.01'] = os.path.join(root, clip_name+'_0.01.json')
                clips[clip_name]['retina_0.8'] = os.path.join(root, clip_name+'_0.8.json')

    return clips


def read_clip_GT(clip_uid):
    # read GT
    res = np.loadtxt('ego4d/headbox_wearer_speaker/' + clip_uid + '.txt') 
    if len(res.shape) == 1:
        res = np.expand_dims(res, axis=0)

    box = {};
    for n in range(res.shape[0]):
        if not (res[n][0] in box):
            box[res[n][0]] = []
        box[res[n][0]].append([res[n][1], res[n][2], res[n][3], res[n][4], res[n][5], res[n][6]]);

    # pid = int(b[k][0])
    # x1 = int(b[k][1])
    # y1 = int(b[k][2])
    # x2 = int(b[k][3])
    # y2 = int(b[k][4])

    return box


# in: video_path, frame_number, boxes: [{ object_type, bbox: {x, y, width, height} }]}, draw_labels
# out: path to image of bboxes rendered onto the video frame
def render_frame_with_bboxes(video_path, frame_number, boxes, draw_labels = True):
    colormap = { # Custom colors for FHO annotations
        'object_of_change': (0, 255, 255),
        'left_hand': (0, 0, 255),
        'right_hand': (0, 255, 0)
    }
    defaultColor = (255, 255, 0)
    rect_thickness = 5
    rectLineType = cv2.LINE_4
    fontColor = (0, 0, 0)
    fontFace = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 1
    fontThickness = 1
    with av.open(video_path) as input_video:
        frames = list(_get_frames([frame_number], input_video, include_audio=False, audio_buffer_frames=0))
        assert len(frames) == 1
        img = frames[0].to_ndarray(format="bgr24")
        for box in boxes:
            label, bbox = box['object_type'], box['bbox']
            rectColor = colormap.get(label, defaultColor) if label else defaultColor
            x, y, width, height = list(map(lambda x: int(x), [bbox['x'], bbox['y'], bbox['width'], bbox['height']]))
            cv2.rectangle(img, pt1=(x,y), pt2=(x+width, y+height), color=rectColor, thickness=rect_thickness, lineType=rectLineType)
            if label and draw_labels:
                textSize, baseline = cv2.getTextSize(label, fontFace, fontScale, fontThickness)
                textWidth, textHeight = textSize
                cv2.rectangle(img, pt1=(x - rect_thickness//2, y - rect_thickness//2), pt2=(x + textWidth + 10 + rect_thickness, y - textHeight - 10 - rect_thickness), color=rectColor, thickness=-1)
                cv2.putText(img, text=label, org=(x + 10, y - 10), fontFace=fontFace, fontScale=fontScale, color=fontColor, thickness=fontThickness, lineType=cv2.LINE_AA)
    path = f"/tmp/{frame_number}_{str(uuid.uuid1())}.jpg"
    cv2.imwrite(path, img)
    return path

# in: video_path, frames: [{ frame_number, frame_type, boxes: [{ object_type, bbox: {x, y, width, height} }] }]
# out: void; as a side-effect, renders frames from the video with matplotlib
def plot_frames_with_bboxes(video_path, frames, max_cols = 3):
    cols = min(max_cols, len(frames))
    rows = math.ceil(len(frames) / cols)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10*cols, 7 * rows))
    if len(frames) > 1:
        [axi.set_axis_off() for axi in axes.ravel()] # Hide axes
    for idx, frame_data in enumerate(frames):
        row = idx // max_cols
        col = idx % max_cols
        frame_path = render_frame_with_bboxes(video_path, frame_data['frame_number'], frame_data['boxes'])
        axes[row, col].title.set_text(frame_data['frame_type'])
        axes[row, col].imshow(mpimg.imread(frame_path, format='jpeg'))
    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.show()

# Aggregate av tracking bounding boxes in an indexable dictionary
def get_av_frame_dict(av_video_annotation):
    frame_aggregator = {}
    for person in av_video_annotation['persons']:
        for tracking_path in person['tracking_paths']:
            for track in tracking_path['track']:
                frame = frame_aggregator.get(track['video_frame'])
                if frame is None:
                    
                    # skip anchor frames (CY)
                    if track['video_frame'] == -1:
                        continue

                    frame = {
                        "frame_number": track['video_frame'],
                        "frame_label": f"Frame: {track['video_frame']}",
                        "frame_type": f"Frame: {track['video_frame']}",
                        "boxes": []
                    }
                frame['boxes'].append({
                    "object_type": tracking_path['track_id'],
                    "bbox": {
                        "x": track['x'],
                        "y": track['y'],
                        "width": track['width'],
                        "height": track['height']
                    }
                })
                frame_aggregator[track['video_frame']] = frame
    return frame_aggregator

# Get ordered list of frames 
def get_av_frames_with_bboxes(av_video_annotation):
    frame_dict = get_av_frame_dict(av_video_annotation)
    return sorted(list(frame_dict.values()), key=lambda x: x['frame_number'])





def extract_annotation(clip_uuid):
    """ Video API """
    videos_df = pd.read_csv(MANIFEST_PATH)

    # Load AV task annotation
    # with open(os.path.join(CLI_OUTPUT_DIR, VERSION, 'annotations', 'av_train.json'), "r") as f:
    with open(os.path.join(CLI_OUTPUT_DIR, VERSION, 'annotations', 'av_val.json'), "r") as f:
        av_annotations = json.load(f)
        av_ann_video_uids = [x["video_uid"] for x in av_annotations["videos"]]
    av_video_dict = {x["video_uid"]: x["clips"] for x in av_annotations["videos"]}
    print(f"AV: {len(av_ann_video_uids)} videos - top level: {av_annotations.keys()}")

    # read by clip uuid
    av_video = videos_df[videos_df.exported_clip_uid == clip_uuid].iloc[0]
    parent_video_uid = av_video.parent_video_uid

    # Locate AV Annotations
    av_video_annotations = av_video_dict.get(parent_video_uid)

    if av_video_annotations is None:
        print(f"AV annotations not found for clip {clip_uuid}")
        return None

    clip_idx = None
    for i in range(len(av_video_annotations)):
        anno = av_video_annotations[i]
        if anno['clip_uid'] == av_video_uid:
            clip_idx = i
        if not anno['valid']:
            clip_idx = None
    assert clip_idx is not None, f"Clip {av_video_uid} not found in {parent_video_uid}"

    """
    av_video_annotations[0]['social_segments_talking'], exmaple

    {'start_time': 291.03259, 'end_time': 295.08615, 'start_frame': 8730, 'end_frame': 8852, 'video_start_time': 731.6312878666668, 'video_end_time': 735.6848478666667, 'video_start_frame': 21948, 'video_end_frame': 22070, 'person': '4', 'target': '0', 'is_at_me': True},
   
    av_video_annotations[0]['social_segments_looking'], example

    {'start_time': 283.963, 'end_time': 284.87697, 'start_frame': 8518, 'end_frame': 8546, 'video_start_time': 724.5616978666667, 'video_end_time': 725.4756678666668, 'video_start_frame': 21736, 'video_end_frame': 21764, 'person': '2', 'target': None, 'is_at_me': False},

    av_video_annotations[0]['persons'][1]['tracking_paths'][1] - track_visual_anchor_2, example

    {'track_id': 'track_visual_anchor_2', 'track': [{'x': 1029.62, 'y': 319.85, 'width': 112.72, 'height': 144.03, 'frame': 0, 'video_frame': 13230, 'clip_frame': None}], 'suspect': False, 'unmapped_frames_count': 0, 'unmapped_frames': []}
    
    av_video_annotations[0]['persons'][1]['tracking_paths']
    {'x': 138.29, 'y': 137.29, 'width': 149.87, 'height': 194.39, 'frame': 8974, 'video_frame': 22192, 'clip_frame': None},

    """

    # Aggregate frames from av person tracking
    av_tracked_frame_dict = get_av_frame_dict(av_video_annotations[clip_idx])
    frame_annotations = get_av_frames_with_bboxes(av_video_annotations[clip_idx])

    return frame_annotations
    