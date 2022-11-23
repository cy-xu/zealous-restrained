import numpy as np
from collections import OrderedDict


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3
    Suspicious = 4
    GT = 5
    Proposal = 6

    states ={}
    states[0] = ['New', (0, 255, 0)]
    states[1] = ['Tracked', (255, 255, 255)]
    states[2] = ['Lost', (128, 0, 0)]
    states[3] = ['Removed', (128, 0, 0)]
    states[4] = ['Suspicious', (0, 0, 255)]
    states[5] = ['GT', (0, 255, 0)]
    states[6] = ['Proposal', (255, 0, 0)]


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def reset():
        BaseTrack._count = 0
        BaseTrack.track_id = 0
        BaseTrack.history = OrderedDict()
        BaseTrack.features = []