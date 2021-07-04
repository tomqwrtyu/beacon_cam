#!/usr/bin/env python3

"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

#import logging
import threading
import os
import sys
import subprocess #shell command
from collections import deque
from argparse import ArgumentParser, SUPPRESS
from math import exp as exp
from time import perf_counter
from enum import Enum

import cv2
import pyrealsense2.pyrealsense2 as rs
import numpy as np
from openvino.inference_engine import IECore

import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker,MarkerArray
from beacon_cam.srv import *

#import keyboard 

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common'))
import helpers
import monitors
import lab
from performance_metrics import PerformanceMetrics


#logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
#log = logging.getLogger()

class arguments:
    def __init__(self):
        self.model = "/home/ubuntu/catkin_ws/src/beacon_cam/src/yolov4/frozen_darknet_yolov4_model.xml"
        self.device = "MYRIAD"
        self.labels = ""#"/home/ubuntu/catkin_ws/src/beacon_cam/src/yolov4/labels_map.txt"
        self.prob_threshold = [0.92,0.92]
        self.iou_threshold = 0.4
        self.nireq = 1
        self.raw_output_message = False
        self.num_infer_requests = 1
        self.num_streams = ""
        self.number_threads = None
        self.no_show = False
        self.utilization_monitors = ''
        self.keep_aspect_ratio = False
        self.color_file = "/home/ubuntu/catkin_ws/src/beacon_cam/src/cr.txt"
        self.color_range = {}
    def load_range(self):
        try:
            with open(self.color_file,'r') as cf:
                for line in cf:
                    line_split = line.split(" ")
                    self.color_range[line_split[0]] = [float(line_split[1]), float(line_split[2]), float(line_split[3]), float(line_split[4])]
        except OSError as e:
            rospy.loginfo(e)
            exit(1)

class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.side = side
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]

        self.isYoloV3 = False

        if param.get('mask'):
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

            self.isYoloV3 = True # Weak way to determine but the only one.


class Modes(Enum):
    USER_SPECIFIED = 0
    MIN_LATENCY = 1


class Mode():
    def __init__(self, value):
        self.current = value

    def get_other(self):
        return Modes.MIN_LATENCY if self.current == Modes.USER_SPECIFIED \
                                 else Modes.USER_SPECIFIED

    def switch(self):
        self.current = self.get_other()


def scale_bbox(x, y, height, width, class_id, confidence, im_h, im_w, is_proportional):
    if is_proportional:
        scale = np.array([min(im_w/im_h, 1), min(im_h/im_w, 1)])
        offset = 0.5*(np.ones(2) - scale)
        x, y = (np.array([x, y]) - offset) / scale
        width, height = np.array([width, height]) / scale
    xmin = int((x - width / 2) * im_w)
    ymin = int((y - height / 2) * im_h)
    xmax = int(xmin + width * im_w)
    ymax = int(ymin + height * im_h)
    # Method item() used here to convert NumPy types to native types for compatibility with functions, which don't
    # support Numpy types (e.g., cv2.rectangle doesn't support int64 in color parameter)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id.item(), confidence=confidence.item())


def parse_yolo_region(predictions, resized_image_shape, original_im_shape, params, threshold, is_proportional):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = predictions.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    size_normalizer = (resized_image_w, resized_image_h) if params.isYoloV3 else (params.side, params.side)
    bbox_size = params.coords + 1 + params.classes
    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for row, col, n in np.ndindex(params.side, params.side, params.num):
        # Getting raw values for each detection bounding box
        bbox = predictions[0, n*bbox_size:(n+1)*bbox_size, row, col]
        x, y, width, height, object_probability = bbox[:5]
        class_probabilities = bbox[5:]
        if object_probability < 0.5:
            continue
        # Process raw value
        x = (col + x) / params.side
        y = (row + y) / params.side
        # Value for exp is very big number in some cases so following construction is using here
        try:
            width = exp(width)
            height = exp(height)
        except OverflowError:
            continue
        # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
        width = width * params.anchors[2 * n] / size_normalizer[0]
        height = height * params.anchors[2 * n + 1] / size_normalizer[1]

        class_id = np.argmax(class_probabilities)
        confidence = class_probabilities[class_id]*object_probability
        if confidence < 0.5:
            continue
        objects.append(scale_bbox(x=x, y=y, height=height, width=width, class_id=class_id, confidence=confidence,
                                  im_h=orig_im_h, im_w=orig_im_w, is_proportional=is_proportional))
    return objects


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union


def resize(image, size, keep_aspect_ratio, interpolation=cv2.INTER_LINEAR):
    if not keep_aspect_ratio:
        return cv2.resize(image, size, interpolation=interpolation)

    iw, ih = image.shape[0:2][::-1]
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = cv2.resize(image, (nw, nh), interpolation=interpolation)
    new_image = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    dx = (w-nw)//2
    dy = (h-nh)//2
    new_image[dy:dy+nh, dx:dx+nw, :] = image
    return new_image


def preprocess_frame(frame, input_height, input_width, nchw_shape, keep_aspect_ratio):
    in_frame = resize(frame, (input_width, input_height), keep_aspect_ratio)
    if nchw_shape:
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    in_frame = np.expand_dims(in_frame, axis=0)
    return in_frame


def get_objects(output, net, new_frame_height_width, source_height_width, prob_threshold, is_proportional):
    objects = list()
    for layer_name, out_blob in output.items():
        out_blob = out_blob.buffer #reshape(net.layers[net.layers[layer_name].parents[0]].out_data[0].shape)
        layer_params = YoloParams(net.layers[layer_name].params, out_blob.shape[2])
        objects += parse_yolo_region(out_blob, new_frame_height_width, source_height_width, layer_params,
                                     prob_threshold, is_proportional)

    return objects


def filter_objects(objects, iou_threshold, prob_threshold):
    # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
    objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            # We perform IOU only on objects of same class 
            if objects[i]['class_id'] != objects[j]['class_id']: 
                continue

            if intersection_over_union(objects[i], objects[j]) > iou_threshold:
                objects[j]['confidence'] = 0

    return tuple(obj for obj in objects if obj['confidence'] >= prob_threshold[obj['class_id']])


def async_callback(status, callback_args):
    request, frame_id, frame_mode, frame, depth_frame, start_time, completed_request_results, empty_requests, \
    mode, event, callback_exceptions = callback_args

    try:
        if status != 0:
            raise RuntimeError('Infer Request has returned status code {}'.format(status))

        completed_request_results[frame_id] = (frame, depth_frame, request.output_blobs, start_time, frame_mode == mode.current)

        if mode.current == frame_mode:
            empty_requests.append(request)
    except Exception as e:
        callback_exceptions.append(e)

    event.set()


def await_requests_completion(requests):
    for request in requests:
        request.wait() 


def rs_isOpened(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    #depth_frame = frames.get_depth_frame()
    return 0 if not color_frame else 1


def color_dtm(lab_frame, y, x, diff, color_range): #if green then return 0,red then return 1,else then 2
    examine_axes_list = []
    x = x - diff
    y = y - diff
    #print(y,x,diff)
    green = color_range['green']
    red = color_range['red']
    green_count = 0
    red_count = 0
    for i in range(3):
        for j in range(3):
            if i == 0 or i == 2:
                if j == 1:
                    examine_axes_list.append([x+diff*i,y+diff*j])
            else:
                 examine_axes_list.append([x+diff*i,y+diff*j])
    for axis in examine_axes_list:
        a = lab_frame[x, y, 1]
        b = lab_frame[x, y, 2]
    
        if (a < green[0]+green[1] and a > green[0]-green[1]) \
             and ((b < green[2]+green[3] and b > green[2]-green[3])):
            green_count += 1
        elif (a < red[0]+red[1] and a > red[0]-red[1]) \
             and ((b < red[2]+red[3] and b > red[2]-red[3])):
            red_count += 1
    if green_count >= 5:
        return 0
    elif red_count >= 5:
        return 1
    else:
        return 2

class beacon_cam_server():
    def __init__(self):
        self.LastStorage = []
        self.pos3d_pub = None
        self.FRAME_ID = 'base_Camera'
        self.LifeTime = 1
        
    def start(self):
        rospy.init_node('beacon_camera')
        self.pos3d_pub = rospy.Publisher('cup_3d', marker_array,queue_size = 10)
        service = rospy.Service('cup_camera', cup_camera, self._request_handler)
        
    def publish_pos3d(self):
        marker_array = MarkerArray()
        for i, pos3d in  enumerate(self.LastStorage[1]):
            marker = Marker()
            marker.header.frame_id = self.FRAME_ID
            marker.header.stamp = rospy.Time.now()
            
            marker.id = i 
            marker.action = Marker.ADD
            marker.lifetime = rospy.Duration(self.LifeTime)
            marker.type = Marker.CYLINDER
            
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            if self.LastStorage[0][i] == '0':
                marker.color.g = 1.0
            elif self.LastStorage[0][i] == '1':
                marker.color.r = 1.0
                
            marker.scale.x = 7.0
            marker.scale.y = 7.0
            marker.scale.z = 12.0
            
            marker.points = pos3d
            marker_array.markers.append(marker)
        self.pos3d_pub.publish(marker_array)
            
    
    def _request_handler(self,request):
        response = cup_cameraResponse()
        if request.req:
            response.color = self.LastStorage[0]
            flat_list = [item for sublist in self.LastStorage[1] for item in sublist]
            response.cup_pos = flat_list
            return response
        else:
            return response

def get_transformed_points(points):
    rospy.wait_for_service('point_transform')
    try:
        transform = rospy.ServiceProxy('point_transform', point_transform)
        res = transform(points)
        return res
    except rospy.ServiceException:
         rospy.loginfo('Service call failed.')
         

        

def main():
    #args = build_argparser().parse_args()
    args = arguments()
    args.no_show = False
    if not os.path.isfile(args.color_file):
        lab.getData()
    args.load_range()
    
    #server
    ros_server = beacon_cam_server()
    ros_server.start()


    # ------------- 1. Plugin initialization for specified device and load extensions library if specified -------------
    rospy.loginfo("Creating Inference Engine...")
    ie = IECore()

    config_user_specified = {}
    config_min_latency = {}

    devices_nstreams = {}
    if args.num_streams:
        devices_nstreams = {device: args.num_streams for device in ['CPU', 'GPU'] if device in args.device} \
                           if args.num_streams.isdigit() \
                           else dict([device.split(':') for device in args.num_streams.split(',')])

    if 'CPU' in args.device:
        if args.cpu_extension:
            ie.add_extension(args.cpu_extension, 'CPU')
        if args.number_threads is not None:
            config_user_specified['CPU_THREADS_NUM'] = str(args.number_threads)
        if 'CPU' in devices_nstreams:
            config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                                                              if int(devices_nstreams['CPU']) > 0 \
                                                              else 'CPU_THROUGHPUT_AUTO'

        config_min_latency['CPU_THROUGHPUT_STREAMS'] = '1'

    if 'GPU' in args.device:
        if 'GPU' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                                                              if int(devices_nstreams['GPU']) > 0 \
                                                              else 'GPU_THROUGHPUT_AUTO'

        config_min_latency['GPU_THROUGHPUT_STREAMS'] = '1'

    # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) --------------------
    rospy.loginfo("Loading network")
    net = ie.read_network(args.model, os.path.splitext(args.model)[0] + ".bin")

    assert len(net.input_info) == 1, "Sample supports only YOLO V3 based single input topologies"

    # ---------------------------------------------- 3. Preparing inputs -----------------------------------------------
    rospy.loginfo("Preparing inputs")
    input_blob = next(iter(net.input_info))

    # Read and pre-process input images
    if net.input_info[input_blob].input_data.shape[1] == 3:
        input_height, input_width = net.input_info[input_blob].input_data.shape[2:]
        nchw_shape = True
    else:
        input_height, input_width = net.input_info[input_blob].input_data.shape[1:3]
        nchw_shape = False

    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None
    
    mode = Mode(Modes.USER_SPECIFIED)
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 10)
    #config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
    #config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    #depth_sensor.set_option(rs.option.enable_auto_exposure, False)
    if depth_sensor.supports(rs.option.depth_units):
        depth_sensor.set_option(rs.option.depth_units,0.001)
    depth_scale = depth_sensor.get_depth_scale()
    #print("Depth Scale is: " , depth_scale)
    align_to = rs.stream.color
    align = rs.align(align_to)

    wait_key_time = 1

    # ----------------------------------------- 4. Loading model to the plugin -----------------------------------------
    rospy.loginfo("Loading model to the plugin")
    exec_nets = {}

    exec_nets[Modes.USER_SPECIFIED] = ie.load_network(network=net, device_name=args.device,
                                                      config=config_user_specified,
                                                      num_requests=args.num_infer_requests)
    exec_nets[Modes.MIN_LATENCY] = ie.load_network(network=net, device_name=args.device.split(":")[-1].split(",")[0],
                                                   config=config_min_latency,
                                                   num_requests=1)

    empty_requests = deque(exec_nets[mode.current].requests)
    completed_request_results = {}
    next_frame_id = 0
    next_frame_id_to_show = 0
    mode_metrics = {mode.current: PerformanceMetrics()}
    prev_mode_active_request_count = 0
    event = threading.Event()
    callback_exceptions = []

    # ----------------------------------------------- 5. Doing inference -----------------------------------------------
    rospy.loginfo("Starting inference...")
    #print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    #print("To switch between min_latency/user_specified modes, press TAB key in the output window")

    presenter = monitors.Presenter(args.utilization_monitors, 55, 1280, 720)
        #(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 4), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 8)))
            
    
    while (rs_isOpened(pipeline) \
           or completed_request_results \
           or len(empty_requests) < len(exec_nets[mode.current].requests)) \
          and not callback_exceptions and not rospy.is_shutdown():
        if next_frame_id_to_show in completed_request_results:
            frame, depth_frame, output, start_time, is_same_mode = completed_request_results.pop(next_frame_id_to_show)
            lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

            next_frame_id_to_show += 1

            objects = get_objects(output, net, (input_height, input_width), frame.shape[:-1], args.prob_threshold,
                                  args.keep_aspect_ratio)
            objects = filter_objects(objects, args.iou_threshold, args.prob_threshold)

            if len(objects) and args.raw_output_message:
                rospy.loginfo(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")

            origin_im_size = frame.shape[:-1]
            presenter.drawGraphs(frame)
            
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            depth_image = np.asanyarray(depth_frame.get_data())
            
            label_zero_point = []
            label_one_pixel = []
            for obj in objects:
                # Validation bbox of detected object
                obj['xmax'] = min(obj['xmax'], origin_im_size[1])
                obj['ymax'] = min(obj['ymax'], origin_im_size[0])
                obj['xmin'] = max(obj['xmin'], 0)
                obj['ymin'] = max(obj['ymin'], 0)
                color = (min(obj['class_id'] * 12.5, 255),
                         min(obj['class_id'] * 7, 255),
                         min(obj['class_id'] * 5, 255))
                xavg = int((obj['xmin']+obj['xmax'])/2)
                yavg = int((obj['ymin']+obj['ymax'])/2)
                ydet = int((obj['ymin']*2+obj['ymax']*8)/10)
                diff = int(((obj['ymax']-obj['ymin'])+(obj['xmax']-obj['xmin']))/20)
                det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
                    str(obj['class_id'])
                if obj['class_id'] == 1:
                    continue
                    #label_one_pixel.append([[obj['xmin'],obj['ymin']],[obj['xmax'],obj['ymax']]])
                elif obj['class_id'] == 0:
                    real_depth = depth_frame.get_distance(xavg,ydet)
                    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [xavg, ydet], real_depth/depth_scale)
                    label_zero_point.append([depth_point,color_dtm(lab_frame, xavg, yavg, diff, args.color_range)])
                    # 0 = green,1 = red,2 = others

                if args.raw_output_message:
                    rospy.loginfo(
                        "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'],
                                                                                  obj['xmin'], obj['ymin'], obj['xmax'],
                                                                                  obj['ymax'],
                                                                                  color))
                cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
                cv2.putText(frame,
                            "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' % ',
                            (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            if any(label_zero_point):
                #detect_target = [x  for x in label_zero_point if not x[1] == 2]
                #maybe_target = [x  for x in label_zero_point if x[1] == 2]
                transformed_label_zero_point = [get_transformed_points(x[0]) for x in label_zero_point]
                ros_server.LastStorage = [[x[1] for x in label_zero_point],[x.tf_pos for x in transformed_label_zero_point]]
                ros_server.pos3d_pub()
                rospy.loginfo(ros_server.LastStorage)
            #Five cups colors if well detected in right region
            """
            if any(label_one_pixel):
                index_1 = 0
                index_2 = 0
                for i in range(len(label_one_pixel)):
                    if label_one_pixel[i][0][0] < label_one_pixel[index_1][0][0]:
                        index_1 = i
                    if label_one_pixel[i][1][0] > label_one_pixel[index_2][1][0]:
                        index_2 = i
                lenth_to_detect = int(round(label_one_pixel[index_2][1][1] - label_one_pixel[index_1][0][1]))
                mid_x_point = int((label_one_pixel[index_1][0][0]+label_one_pixel[index_2][1][0])/2)
                start = label_one_pixel[index_1][0][1]
                colors = [-1,-1,-1,-1,-1] # 0 for green 1 for red -1 for no cup
                for i in range(5):
                    target_color = [0,0,0]
                    for j in range(lenth_to_detect):
                        target_color[color_dtm(lab_frame, mid_x_point, j, 2,args.color_range)] += 1
                    if target_color[0] > int(lenth_to_detect/2):
                        colors[i] = 0
                    elif target_color[1] > int(lenth_to_detect/2):
                        colors[i] = 1
                    start  = start + lenth_to_detect
                colors_message = []
                for i in colors:
                    if i == 0:
                        colors_message.append("green")
                    elif i == 1:
                        colors_message.append("red")
                    else:
                        colors_message.append("no cup")
                rospy.loginfo(colors_message)
            """
            #end
            #helpers.put_highlighted_text(frame, "{} mode".format(mode.current.name), 
            #                             (10, int(origin_im_size[0] - 20)),
            #                             cv2.FONT_HERSHEY_COMPLEX, 0.75, (10, 10, 200), 2)
            
            if is_same_mode and prev_mode_active_request_count == 0:
                ros_server.LifeTime = mode_metrics[mode.current].update(start_time, frame)
            else:
                ros_server.LifeTime = mode_metrics[mode.get_other()].update(start_time, frame)
                prev_mode_active_request_count -= 1
                helpers.put_highlighted_text(frame, "Switching modes, please wait...",
                                             (10, int(origin_im_size[0] - 50)), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                                             (10, 200, 10), 2)
            if not args.no_show:
                cv2.namedWindow("Detection Results", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("Detection Results", cv2.WND_PROP_FULLSCREEN,  cv2.WINDOW_FULLSCREEN)
                cv2.imshow("Detection Results", frame)
                key = cv2.waitKey(wait_key_time)

                if key in {ord("q"), ord("Q"), 27}: # ESC key
                    break
                if key == 9: # Tab key
                    if prev_mode_active_request_count == 0:
                        prev_mode = mode.current
                        mode.switch()

                        prev_mode_active_request_count = len(exec_nets[prev_mode].requests) - len(empty_requests)
                        empty_requests.clear()
                        empty_requests.extend(exec_nets[mode.current].requests)

                        #mode_metrics[mode.current] = PerformanceMetrics()
                else:
                    presenter.handleKey(key)
                

        elif empty_requests and prev_mode_active_request_count == 0 and rs_isOpened(pipeline):
            start_time = perf_counter()
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)   
            depth_frame = aligned_frames.get_depth_frame() 
            color_frame = aligned_frames.get_color_frame()
            frame = np.asanyarray(color_frame.get_data())

            request = empty_requests.popleft()

            # resize input_frame to network size
            in_frame = preprocess_frame(frame, input_height, input_width, nchw_shape, args.keep_aspect_ratio)

            # Start inference
            request.set_completion_callback(py_callback=async_callback,
                                            py_data=(request,
                                                     next_frame_id,
                                                     mode.current,
                                                     frame,
                                                     depth_frame,
                                                     start_time,
                                                     completed_request_results,
                                                     empty_requests,
                                                     mode,
                                                     event,
                                                     callback_exceptions))
            request.async_infer(inputs={input_blob: in_frame})
            next_frame_id += 1

        else:
            event.wait()
            event.clear()

    if callback_exceptions:
        raise callback_exceptions[0]

    for mode, metrics in mode_metrics.items():
        print("\nMode: {}".format(mode.name))
        metrics.print_total()
    print(presenter.reportMeans())

    for exec_net in exec_nets.values():
        await_requests_completion(exec_net.requests)
        
    pipeline.stop()
    subprocess.call(["killall", "roslaunch"])


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
