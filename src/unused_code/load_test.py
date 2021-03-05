import os

class arguments:
    def __init__(self):
        self.model = "/home/ubuntu/catkin_ws/src/beacon_cam/src/yolov4/frozen_darknet_yolov4_model.xml"
        self.device = "MYRIAD"
        self.labels = ""#"/home/ubuntu/catkin_ws/src/beacon_cam/src/yolov4/labels_map.txt"
        self.prob_threshold = [0.9,0.8]
        self.iou_threshold = 0.4
        self.nireq = 1
        self.raw_output_message = False
        self.num_infer_requests = 1
        self.num_streams = ""
        self.number_threads = None
        self.no_show = False
        self.utilization_monitors = ''
        self.keep_aspect_ratio = False
        self.color_file = "cr.txt"
        self.color_range = {}
    def load_range(self):
        try:
            with open(self.color_file,'r') as cf:
                for line in cf:
                    line_split = line.split(" ")
                    self.color_range[line_split[0]] = [line_split[1], line_split[2], line_split[3], line_split[4]]
        except OSError as e:
            print(e)
            exit(1)
            
args = arguments()
args.load_range()

args.color_range['green']