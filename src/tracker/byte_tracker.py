import numpy as np
import os
from .kalman_filter import KalmanFilter
from .basetrack import BaseTrack, TrackState
import tracker.yolox_matching as matching

# import cv2
# import torch
# import torchvision.models as models
# import torchvision.transforms as transforms
# import torch.nn as nn
# import face_recognition


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        # show predicted box for this many extra frames
        self.lost_delay = 5

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False, bypass_kalman=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh), bypass_kalman
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.lost_delay = 5

    def update(self, new_track, frame_id, bypass_kalman=False):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        if self.track_id == 0:
            self.track_id = self.next_id()

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh), bypass_kalman
            )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        
    def mark_Suspicious(self, kalman_filter, frame_id, same_id=0):
        self.kalman_filter = kalman_filter
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.state = TrackState.Suspicious
        self.is_activated = False
        # maintain same ID as the previous suspicious box
        if same_id > 0:
            self.track_id = same_id

        if self.track_id == 0:
            self.track_id = self.next_id()

        self.frame_id = frame_id
        self.start_frame = frame_id

    def mark_proposal(self):
        self.state = TrackState.Proposal
        self.lost_delay -= 1

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def custom_copy(self):
        new_t = STrack(tlwh=self.tlwh, score=self.score)
        new_t.track_id = self.track_id
        new_t.track_uuid = TrackState.states[self.state][0]+'_'+str(self.track_id)
        new_t.state = self.state
        new_t.is_activated = self.is_activated
        new_t.frame_id = self.frame_id
        return new_t

    def mark_tiny(self):
        self.state = TrackState.Suspicious
        self.track_uuid = self.track_uuid + '_tiny'

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, thres_high=0.3, thres_low=0.1, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.suspicious_tracks = []

        self.tracks_per_frame = {}  # type: Dict[int, List[STrack]]
        self.tracks_uuid = {}

        self.frame_id = 0
        self.args = args
        self.thres_high = thres_high
        self.thres_low = thres_low
        # self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # reset track id across different videos
        STrack.reset()

    def update(self, output_results, img_info, img_size, include_FP=True):

        # empty detection, but still keep a record
        if len(output_results) == 0:
            self.tracks_per_frame[self.frame_id] = []
            self.frame_id += 1
            return list()

        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        suspicious_tracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2

        # corrects for resized input to yolox
        img_h, img_w = img_info[0], img_info[1]
        # scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        # bboxes /= scale

        '''
        CY: to address the clustred tiny boxes that cause most confusion for human,
        we add a new additional step to first detect tiny, overlapped boxes.
        For each overlapping pair of boxes, we keep the one with the higher score.
        '''
        if self.args.tiny_ratio > 0.0 and self.args.cluster_iou > 0.0:
            
            cluster_IDs = set()
            tiny_area = self.args.tiny_ratio**2 * img_h * img_w

            # IOU between all boxes
            _ious = matching.ious(bboxes, bboxes)
            # diagonal values will be ones (self matching), make them 0
            # _ious = _ious - np.eye(_ious.shape[0])

            # find overlapping pairs 
            # matches, _, _ = matching.linear_assignment(1-_ious, 0.9)

            # linear_assignment is not suitable here as it providse best paris only
            # while there might be multiple overlapping boxes qualifies

            # for each box and one or more overlapping neighbors, keep the one with the highest score
            for i in range(len(_ious)):
                cluster_indices = np.where(_ious[i] > self.args.cluster_iou)[0]
                if len(cluster_indices) == 1:
                    # only self-matching, pass
                    continue

                cluster_scores = scores[cluster_indices]
                idx_to_keep = cluster_indices[np.argmax(cluster_scores)]

                # add all other idices to cluster_indices except idx_to_keep
                # check if the box is tiny, if so, it will be removed
                for idx in cluster_indices:
                    # keep the one with the highest score
                    if idx == idx_to_keep:
                        continue
                    # keep boxes that are more confident
                    if scores[idx] > self.args.model_thresh_high:
                        continue

                    _, _ , w, h = STrack.tlbr_to_tlwh(bboxes[idx])
                    # and only remove a box if it's also tiny
                    if w*h <= tiny_area:
                        cluster_IDs.add(idx)

            if self.args.debug_boxes and len(cluster_IDs) > 0:
                print(f'frame {self.frame_id}: {len(cluster_IDs)} tiny clusterred boxes to remove')
                print(f'scores {scores[list(cluster_IDs)]}')

            # remove tiny boxes from bboxes
            bboxes = np.delete(bboxes, list(cluster_IDs), axis=0)
            scores = np.delete(scores, list(cluster_IDs), axis=0)


        ############### skip tracking, displya all boxes for debug purpose 
        if self.args.debug_boxes:
            self.tracked_stracks = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                            (tlbr, s) in zip(bboxes, scores)]

            output_stracks = [t.custom_copy() for t in self.tracked_stracks]

            # index tracks by UUID
            for t in output_stracks:
                uuid = t.track_uuid
                if uuid not in self.tracks_uuid:
                    self.tracks_uuid[uuid] = {self.frame_id: t}
                else:
                    self.tracks_uuid[uuid][self.frame_id] = t

            # arrange tracks by frames
            self.tracks_per_frame[self.frame_id] = output_stracks
            self.frame_id += 1

            return
        ##################

        # Then starts ByteTrack matching process 
        remain_inds = scores > self.thres_high
        inds_low = scores > self.thres_low
        inds_high = scores < self.thres_high

        # high confidence detections
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]

        # low confidence detections
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections_first = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections_first = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed_tracks = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed_tracks.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''

        # the pool includes current tracks and lost tracks
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # match existing tracks with new detections
        dists = matching.iou_distance(strack_pool, detections_first)

        # multiply detection confidence (do we need it for faces?)
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)

        # matches: associated detections and tracklets
        # u_track: unmatched tracklets index
        # u_detection: unmatched detections index
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        # update existing tracks or re-activate lost tracks
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_first[idet]
            if track.state == TrackState.Tracked:
                track.update(detections_first[idet], self.frame_id, self.args.bypass_kalman)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, False, self.args.bypass_kalman)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        # match remaining track with low confidence detections
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=self.args.match_thresh_low)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, self.args.bypass_kalman)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, False, self.args.bypass_kalman)
                refind_stracks.append(track)

        # if a tracked obj does not have a low confidence match
        # mark it as Proposal for self.lost_delay frames until mark it lost
        for it in u_track:
            track = r_tracked_stracks[it]
            # add to lost tracks list
            states = [TrackState.Lost, TrackState.Proposal]
            if not track.state in states:
                # mark proposal, continue to predict
                if track.lost_delay > 0:
                    track.mark_proposal()
                else:
                    track.mark_lost()
                lost_stracks.append(track)

        # mark the unmatched low conf detections as FP
        sus_candidates = [detections_second[i] for i in u_detection_second]
        dists_sus = matching.iou_distance(sus_candidates, self.suspicious_tracks)
        matches_sus, u_candidates, _ = matching.linear_assignment(dists_sus, thresh=self.args.match_thresh_low)

        # append unmatched suspicious with new ID
        for idet in u_candidates:
            det = sus_candidates[idet]
            det.mark_Suspicious(self.kalman_filter, self.frame_id)
            suspicious_tracks.append(det)
    
        # append matched ones with the same ID
        for pair in matches_sus:
            idet, itrack = pair[0], pair[1]
            det = sus_candidates[idet]
            prev_id = self.suspicious_tracks[itrack].track_id
            det.mark_Suspicious(self.kalman_filter, self.frame_id, prev_id)
            suspicious_tracks.append(det)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections_first_un = [detections_first[i] for i in u_detection]
        # detecions should not update here, it will cover the FP ones

        # confirm previous frame's unconfirmed track with new detections
        dists = matching.iou_distance(unconfirmed_tracks, detections_first_un)
        # if not self.args.mot20:
        #     dists = matching.fuse_score(dists, detections)

        # fixed 0.7 threshold here for new tracks?
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed_tracks[itracked].update(detections_first_un[idet], self.frame_id, self.args.bypass_kalman)
            activated_stracks.append(unconfirmed_tracks[itracked])

        for it in u_unconfirmed:
            track = unconfirmed_tracks[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections_first_un[inew]
            if track.score < self.thres_low:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        if include_FP:
            self.tracked_stracks = joint_stracks(self.tracked_stracks, suspicious_tracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # we only save suspicious tracks for one frame for the purpose of consistent ID
        self.suspicious_tracks = suspicious_tracks

        output_stracks = [t.custom_copy() for t in self.tracked_stracks]

        # index tracks by UUID
        for t in output_stracks:
            uuid = t.track_uuid
            if uuid not in self.tracks_uuid:
                self.tracks_uuid[uuid] = {self.frame_id: t}
            else:
                self.tracks_uuid[uuid][self.frame_id] = t

        # arrange tracks by frames
        self.tracks_per_frame[self.frame_id] = output_stracks
        self.frame_id += 1

        # return nothing, call tracks usiing self.tracks_per_frame or self.tracks_uuid
        # return output_stracks

    def remove_short_tracks(self, min_len):
        """ remove tracks appeared in less than min_length frames""" 
        ids_to_pop = []
        for uuid, track_dict in self.tracks_uuid.items():
            if len(track_dict) < min_len:
                ids_to_pop.append(uuid)

        # dict cannot be modified during loop
        for uuid in ids_to_pop:
            self.tracks_uuid.pop(uuid)

        # remove these tracks from tracks_per_frame as well
        for frm, tracks in self.tracks_per_frame.items():
            for idx in reversed(range(len(tracks))):
                if tracks[idx].track_uuid in ids_to_pop:
                    tracks.pop(idx)


    def mark_tiny_tracks(self, min_size=0):
        """ mark tracks with tiny boxes as uncertain""" 
        if min_size == 0:
            return
            
        ids_to_mark = []
        for uuid, track_dict in self.tracks_uuid.items():
            # read all width and height
            all_wh = np.array([b.tlwh[2:] for b in track_dict.values()])
            if np.mean(all_wh[:, 0]) < min_size or np.mean(all_wh[:, 1]) < min_size:
                ids_to_mark.append(uuid)

        # dict cannot be modified during loop
        for uuid in ids_to_mark:
            for t in self.tracks_uuid[uuid].values():
                t.mark_tiny()

        # remove these tracks from tracks_per_frame as well
        for frm, tracks in self.tracks_per_frame.items():
            for t in tracks:
                if t.track_uuid in ids_to_mark:
                    t.mark_tiny()

    def save_tracks(self, output_dir):
        """ save self.tracks_per_frame and self.tracks_uuid as numpy files """
        # os.path.make_dir(output_dir, exist_ok=True)
        np.save(output_dir + '/tracks_per_frame.npy', self.tracks_per_frame)
        np.save(output_dir + '/tracks_uuid.npy', self.tracks_uuid)


    def load_tracks(self, input_dir):
        """ load self.tracks_per_frame and self.tracks_uuid from numpy files """
        self.tracks_per_frame = np.load(input_dir + '/tracks_per_frame.npy', allow_pickle=True).item()
        self.tracks_uuid = np.load(input_dir + '/tracks_uuid.npy', allow_pickle=True).item()


    # def cluster(self, images):
    #     """ cluster Tracked based on their visual features """
    #     self.visual_features = {} # key: track_uuid, value: list of visual features
        
    #     resnet18 = models.resnet18(pretrained=True)
    #     feature_net = nn.Sequential(*(list(resnet18.children())[0:4])) #second way
    #     transforms = torch.nn.Sequential(
    #         transforms.Resize((224, 224)),
    #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #     )

    #     # feature_net = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    #     # feature_net.eval()
    #     # feature_net = nn.Sequential(*list(feature_net.children())[:-1])

    #     # crop each box's image patch and extract features
    #     for uuid, frames in self.tracks_uuid.items():
    #         vis_features = {}
    #         feat_vectors = []
    #         largest_box = (0, [0, 1, 0, 1]) # idx, box

    #         for frm, t in frames.items():
    #             img = images[frm]
    #             box = [int(v) for v in t.tlbr]

    #             if area(t.tlbr) > area(largest_box[1]):
    #                 largest_box = (frm, box)

    #             patch = img[box[1]:box[3], box[0]:box[2], :]

    #             # normalize and extract features
    #             patch = cv2.resize(patch, (224, 224))
    #             patch = patch.astype(np.float32)/255.
    #             patch = torch.tensor(patch).unsqueeze(0).permute(0, 3, 1, 2)

    #             patch = transforms(patch)

    #             vis_feat = feature_net(patch)
    #             face_encoding = vis_feat.detach().numpy().flatten()
    #             feat_vectors.append(face_encoding)

    #         # NN model input needs to be certain size thus modulo
    #         # box = modulo_bounding_box(largest_box[1], modulo=4, min_size=16)
    #         # if box is None: continue

    #         box = largest_box[1]
    #         img = images[largest_box[0]]

    #         patch = img[box[1]:box[3], box[0]:box[2], :]
    #         # save a img patch for visualization
    #         vis_features["patch"] = patch

    #         # use dlib's facial landmark detector for feature extractor
    #         # left, top, right, bottom = box[0], box[1], box[2], box[3]
    #         # location = [(top, left, right, bottom)]
    #         # face_encoding = face_recognition.face_encodings(img, location, num_jitters=10, model="large")[0]

    #         # normalize and extract features from largest patch
    #         # patch = patch.astype(np.float32)/255.
    #         # patch = torch.tensor(patch).unsqueeze(0).permute(0, 3, 1, 2)
    #         # vis_feat = feature_net(patch)
    #         # face_encoding = vis_feat.detach().numpy().flatten()

    #         # use averge of sequence 
    #         face_encoding = np.mean(np.array(feat_vectors), axis=0)

    #         # simply flatten the thumbnail
    #         # face_encoding = patch.flatten()

    #         vis_features["vector"] = face_encoding

    #         # print(f'{uuid}: {vis_features["vector"]}')

    #         # visual feature includes both a image patch and a feature vector
    #         self.visual_features[uuid] = vis_features


def area(box):
    """ Compute area of a box """
    x1 = max(0, box[0])
    y1 = max(0, box[1])
    return (box[2] - x1) * (box[3] - y1)

def ratio(box):
    """ compute the width / height ratio of a box """
    x1 = max(0, box[0])
    y1 = max(0, box[1])
    return (box[2] - x1) / (box[3] - y1)

# a function that resize a bounding box to modulo of 4
def modulo_bounding_box(box, modulo=4, min_size=16) -> list:
    """
    Args:
        box: a bounding box in format [x1, y1, x2, y2]
        modulo: the modulo of the width and height
    Returns:
        a list of [x1, y1, x2, y1]
    """
    box = [max(0, int(x)) for x in box]
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    x2 = x1 + (w // modulo) * modulo
    y2 = y1 + (h // modulo) * modulo

    if x2-x1 < min_size or y2-y1 < min_size:
        return None

    return [x1, y1, x2, y2]



def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0) or tid == 0: # handle FP cases
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

def combine_stracks(main, extra, threshold=0.5):
    pdist = matching.iou_distance(main, extra)
    matches, u_main, u_extra = matching.linear_assignment(pdist, thresh=threshold)
    # add extra tracks (from backward detection) to the main track
    for i in u_extra:
        main.append(extra[i])
    return main