import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import urllib.request

from tracker.yolox_matching import ious, linear_assignment
from tracker.basetrack import TrackState
from tracker.byte_tracker import STrack

def new_json_entry(track, optional_attribute=False):
    """
    here we force the boxes to be labeled by "human" despite
    they actually come from a model, it is because "machine"
    boxes will cause manual adjustments in ADAP tool to overwrite
    all other boxes with the same ID. 
    """
    new_dict = {}
    new_dict['annotated_by'] = "human"

    x, y, w, h = [round(i) for i in track.tlwh]
    new_dict['x'] = x
    new_dict['y'] = y
    new_dict['height'] = h
    new_dict['width'] = w

    if optional_attribute:
        new_dict['metadata'] = {
            'annotated_by': 'machine',
            'shapeAnswers': [{"questionId":"1e61f878","type":"Checkbox","name":"optional","answer":{"values":'true'}}]
            }

    return new_dict

def json_dump(json, ID_limit, frm_id, tracks, optional_attribute=False):
    frmID = str(frm_id + 1)
    real_labels = ['Tracked', 'New']

    # new frame entry
    json['annotation']["frames"][frmID] = {
        "frameRotation": 0,
        "rotatedBy": "machine",
        "shapesInstances":{}
        }

    for t in tracks:
        track_id = str(t.track_id)
        track_uuid = str(t.track_uuid)
        category = TrackState.states[t.state][0]

        # rewrite category names for ADAP tool
        if category in real_labels:
            category = "Certain"
        elif category == "Suspicious":
            category = "Uncertain"
            # do not produce suspicious boxes any more
            continue

        # track categories are saved here
        if track_uuid not in json['annotation']["shapes"]:
            json['annotation']["shapes"][track_uuid] = {
                "type": "box",
                "category": category,
            }

        # new box entries for each frame
        json['annotation']["frames"][frmID]["shapesInstances"][track_uuid] = new_json_entry(t, optional_attribute)


# convert ADAP tool's bbox format to top-left bottom-right
def ADAP_bbox_to_tlbr(json, bbox_dict, ignore_suspicion=True):
    tlbr = []
    cats = []
    ignore_categories = ['Uncertain', 'Suspicious']
    # ignore_categories = []

    for key, box in bbox_dict.items():
        category = json['annotation']['shapes'][key]['category']
        # skip suspicious boxes
        if ignore_suspicion and category in ignore_categories:
            continue
        x,y,w,h = box['x'], box['y'], box['width'], box['height']
        tlbr.append([x,y,x+w,y+h])
        cats.append(category)
    return tlbr, cats


def precision_recall_single_video(conf_matrix, gt, pred, thresholds, consider_optional=True):
    """
    Compute precision and recall for a single video.
    """

    for i in range(len(gt['annotation']['frames'])):
        # frames are 1-indexed
        i += 1

        # extract from ADAP json
        gt_bbox = gt['annotation']['frames'][str(i)]['shapesInstances']
        test_bbox = pred['annotation']['frames'][str(i)]['shapesInstances']

        gt_tlbr, test_tlbr, optionals = [], [], []

        box_ids = list(gt_bbox.keys())
        for j in range(len(box_ids)):
            id = box_ids[j]

            if consider_optional:
                optional_flag = gt_bbox[id]['metadata']['shapeAnswers'][0]['answer']['values']
                if optional_flag: optionals.append(j)
            
            # convert to top-left bottom-right
            x, y, w, h = gt_bbox[id]['x'], gt_bbox[id]['y'], gt_bbox[id]['width'], gt_bbox[id]['height']
            gt_tlbr.extend([[x, y, x+w, y+h]])

        # convert to top-left bottom-right format
        # gt_bbox, gt_cats = ADAP_bbox_to_tlbr(gt, gt_bbox)
        test_tlbr, _ = ADAP_bbox_to_tlbr(pred, test_bbox)

        # if remove otptional boxes, it will affect precision as well
        # so we should remove their count from the total in recall calculation

        # count optionals boxes that won't penalize recall
        # optionals = [i for i in range(len(gt_cats)) if 'optional' in gt_cats[i].lower()]

        # cython_bbox
        _ious = ious(gt_tlbr, test_tlbr)
        iou_cost_matrix = 1 - _ious

        for t in range(len(thresholds)):
            thresh = thresholds[t]
            matches, u_gt, u_detection = linear_assignment(iou_cost_matrix, thresh=thresh)

            # update matches and misses to ignore optionals
            if consider_optional and len(optionals) > 0:
                # remove optionals from the matches
                # matches = matches.tolist()
                # for pair in matches:
                #     if pair[0] in optionals: matches.remove(pair)
                u_gt = [i for i in u_gt if i not in optionals]
            
            conf_matrix[t, 0] += len(matches)  # TP
            conf_matrix[t, 2] += len(u_detection)  # FP
            conf_matrix[t, 3] += len(u_gt)  # FN

    return conf_matrix

def precision_recall_eval(gt, pred, thresholds, consider_optional=True):
    # confusion_matrix as a list: TP, TN, FP, FN
    # three categories: certain, uncertain, optional
    # optional can be considered for precision, but no penalty for recall
    steps = len(thresholds)

    conf_matrix = np.zeros((steps, 4))

    if isinstance(gt, list):
        assert len(gt) == len(pred)

        # for multiple videos, TP, FP, FN are accumulated to conf_matrix
        for i in range(len(gt)):
            gt_json = gt[i]
            pred_json = pred[i]
            conf_matrix = precision_recall_single_video(conf_matrix, gt_json, pred_json, thresholds, consider_optional)
    else:
        conf_matrix = precision_recall_single_video(conf_matrix, gt, pred, thresholds, consider_optional)

    # avoid division by zero
    eps = 1 if 0 in conf_matrix[:, 3] or 0 in conf_matrix[:, 2] else 0 

    precisions = [conf_matrix[i,0]/(conf_matrix[i,0]+conf_matrix[i,2]+eps) * 100 for i in range(steps)]

    recalls = [conf_matrix[i,0]/(conf_matrix[i,0]+conf_matrix[i,3]+eps) * 100 for i in range(steps)]

    return precisions, recalls


def ADAP_json_to_Tracks(json_path, total_frames=0, GT=False):
    # read json
    with open(json_path, 'r') as f:
        judgements = json.load(f)

    # extract tracks from ADAP json
    tracks_per_frame = []
    if total_frames == 0:
        total_frames = len(judgements['annotation']['frames'])

    for i in range(total_frames):
        # frames are 1-indexed
        i += 1

        if str(i) not in judgements['annotation']['frames']:
            tracks_per_frame.append([])
            continue

        # extract from ADAP json
        bboxes = judgements['annotation']['frames'][str(i)]['shapesInstances']
        tracks = []

        for key, value in bboxes.items():
            x1 = int(value['x'])
            y1 = int(value['y'])
            x2 = int(value['x'] + value['width'])
            y2 = int(value['y'] + value['height'])
            # each person
            t = STrack([x1, y1, x2-x1, y2-y1], 1.0)
            if GT:
                t.state = TrackState.GT
            else:
                t.state = TrackState.Tracked
            t.track_uuid = key
            tracks.append(t)

        tracks_per_frame.append(tracks)

    return tracks_per_frame


def batch_download_judgements(df, video_uuid, clip_dir, api_key):
    """
    This function reads the job report downloaded from ADAP dashboard
    to extract judgement for each row, save as new json files,
    it returns a list that contains the new json files' paths. 
    """
    output_jsons = []
    os.makedirs(clip_dir, exist_ok=True)

    # reject outliers
    df_use = df[df['Outlier'] == 0]
    df_use = df_use[df_use['Clip_name'] == video_uuid]

    # iterate through each row in the dataframe
    for index, row in df_use.iterrows():

        annotation_pr = row['annotation_pr'].split('"')
        json_url = [s for s in annotation_pr if "https://" in s][0]

        # judgements are in the format of https://requestor-proxy.appen.com/v1/redeem_token?token=abc
        # replace url to enable download via API and personal token
        json_url = json_url.replace("requestor-proxy", "api-beta")
        # attach personal API for direct download
        json_url += "&key=" + api_key

        with urllib.request.urlopen(json_url) as url:
            data = json.loads(url.read().decode())
            name = row['Group']+'_'+str(row['_worker_id'])+'_'+row['Clip_name']
            out_path = os.path.join(clip_dir, name + '.json')

            output_jsons.append(out_path)
            with open(out_path, 'w') as outfile:
                json.dump(data, outfile)

    return output_jsons