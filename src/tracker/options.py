class TrackOptions():
    def __init__(self):
        # init default options
        self.__dict__["trt"] = False
        self.__dict__["model_thresh_high"] = 0.1
        self.__dict__["model_thresh_low"] = 0.01
        self.__dict__["match_thresh"] = 0.9
        self.__dict__["match_thresh_low"] = 0.9
        self.__dict__["min_box_area"] = 1
        self.__dict__["track_buffer"] = 30
        self.__dict__["min_track_lifespan"] = 10
        self.__dict__["tiny_ratio"] = 0.05
        self.__dict__["debug_boxes"] = False
        self.__dict__["cluster_iou"] = 0.1

        # Original settings for BYTE track
        self.__dict__["track_thresh"] = 0.6
        self.__dict__["mot20"] = False

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    def __len__(self):
        return len(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()
