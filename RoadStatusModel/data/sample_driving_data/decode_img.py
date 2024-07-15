import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import rosbag
import av
import cv2
import subprocess
import torch

def decode_video(bag_path, computer, camera_num, drop=30):
    """
        bag: "{$prefix}_camera_{$computer}_0.bag"
        camera_topic: "/{$computer}/camera_{$num}/camera_image/ffmpeg"
        cropped_topic: "/{$computer}/rf_perception/debug/image_crop"
        debug_topic: "/{$computer}/rf_debug/rf_perception/traffic_signal_state"
    """
    camera_topic = f"/{computer}/camera_{camera_num}/camera_image/ffmpeg"

    camera_number = camera_num
    print(camera_topic)
    bag = rosbag.Bag(bag_path)

    topics = bag.get_type_and_topic_info()[1].keys()
    if not camera_topic in topics:
        return

    dir_path = "/".join(bag_path.split("/")[:-1])
    save_dir = os.path.join(dir_path, bag_path[:-4], camera_topic.split("/")[2])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    _codec = av.codec.CodecContext.create('h264', 'r')
    # output = av.open(f"fast_camera_{camera_number}.h264", "w")
    fps = 100
    # output_stream = output.add_stream("h264", fps)

    pandar64_0_idx = 0
    camera_timestamps = []
    pts = 0
    for topic, msg, t in tqdm(bag.read_messages(topics=camera_topic)):
        if topic == camera_topic:
            packet = av.packet.Packet(msg.data)
            packet.pts = pts #int((msg.pts - min_pts) / 2)
            pts += 1
            try:
                # packet.stream = output_stream

                # output.mux(packet)

                frame = _codec.decode(packet)
                img = frame[0].to_ndarray(format='bgr24')
                img_size = (320, 180)
                img = cv2.resize(img, img_size)

                sensor_timestamp = "{0}.{1:09}".format(msg.header.stamp.secs, msg.header.stamp.nsecs)
                # 가정, lidar는 무조건 0.1s 주기

                camera_timestamps.append(sensor_timestamp)
                file_name = "{}.jpg".format(pandar64_0_idx//drop)
                if pandar64_0_idx % drop == 0:
                    cv2.imwrite(os.path.join(save_dir, file_name), img)

                    # cv2.imshow("img", img)
                pandar64_0_idx += 1
            except Exception as e:
                print(e)
                frame = None




    return camera_timestamps



def decode_camera_images(bag_path, car_platform):
    if bag_path[-4:] != '.bag':
        return
    print("Start decoding camera images!!")
    print(bag_path)
    dir_path = "/".join(bag_path.split("/")[:-1])
    if not os.path.exists(bag_path):
        print("No camera bag exists! Please check again!")
        return

    camera_num_map = {"v2": 6, "v3": 6, "v4": 8, "v5": 8}
    camera_pc_map = {"v2": "nebula", "v3": "nebula", "v4": "yondu", "v5": "yondu"}

    camera_pc = camera_pc_map[car_platform]

    for camera_topic_idx in range(0, camera_num_map[car_platform], 1):
        decode_video(bag_path, camera_pc, camera_topic_idx)







if __name__ == "__main__":
    # bag_dir_path = "/media/hyunkun/ReT7/CAL-TODO/2023-01-19-15-24-14_ioniq_v3_5"
    # analyzing_dir = "/media/hyunkun/ReT7/trafficlight_issue/analyzing"
    # analyzing_dir = "/media/hyunkun/ReT7/CAL-TODO/analyzing"
    v4_rain_dir = "/media/hyunkun/ReT7/KATRI_BLOCKAGE_TEST/v4_not_rain"
    bag_paths = [os.path.join(v4_rain_dir, bag_dir) for bag_dir in os.listdir(v4_rain_dir)]
    [decode_camera_images(bag_path, 'v5') for bag_path in bag_paths]
