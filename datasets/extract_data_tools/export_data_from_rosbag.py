"""
Export data from ROS bag to HDF5.
Command: python2 export_data_from_rosbag.py
        --data_path <path>
        --output_path <path>

Code from: https://github.com/SensorsINI/ECCV_network_grafting_algorithm
Author: Yuhuang Hu
"""

from __future__ import print_function, absolute_import

import argparse

import numpy as np

import h5py
from cv_bridge import CvBridge
import rosbag


def find_nearest(dataset, start_idx, search_value, search_gap=10000):
    num_events = dataset.shape[0]
    nearest_value_idx = 0

    for event_batch in range((num_events-start_idx) // search_gap):
        start_pos = start_idx+event_batch*search_gap
        end_pos = min(start_idx+(event_batch+1)*search_gap,
                      num_events)
        selected_events = dataset[start_pos:end_pos]

        nearest_idx = np.searchsorted(
            selected_events, search_value, side="left")

        if nearest_idx != search_gap:
            nearest_value_idx = start_idx+event_batch*search_gap+nearest_idx
            break

    return nearest_value_idx


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--color", action="store_true")

args = parser.parse_args()

bag = rosbag.Bag(args.data_path, "r")
bridge = CvBridge()

# HDF5 file
dataset = h5py.File(args.output_path, "w")
dvs_data = dataset.create_dataset(
    "davis/left/events",
    shape=(0, 4),
    maxshape=(None, 4),
    dtype="float64")

img_data = dataset.create_dataset(
    "davis/left/image_raw",
    shape=(0, 260, 346, 3),
    maxshape=(None, 260, 346, 3),
    dtype="uint8")

img_ind = dataset.create_dataset(
    "davis/left/image_raw_event_inds",
    shape=(0,),
    maxshape=(None,),
    dtype="int64")

if args.color is True:
    topics = ["/dvs/image_color", "/dvs/events"]
else:
    topics = ["/davis/left/image_raw", "/davis/left/events"]

current_frame_ts = None

event_ts_collector = np.zeros((0,), dtype="float64")
frame_ts_collector = np.zeros((0,), dtype="float64")

for topic, msg, t in bag.read_messages(topics=topics):
    print(topic)
    if topic == topics[1]:
        events = msg.events
        num_events = len(events)

        # save events
        dvs_data.resize(dvs_data.shape[0]+num_events, axis=0)

        event_data = np.array(
            [[x.x, x.y, float(x.ts.to_nsec())/1e9,
              x.polarity] for x in events],
            dtype="float64")

        dvs_data[-num_events:] = event_data

        event_ts_collector = np.append(
            event_ts_collector, [float(x.ts.to_nsec())/1e9 for x in events])
        print("Processed {} events".format(num_events))

    elif topic in topics[0]:
        im_rgb = bridge.imgmsg_to_cv2(msg, "rgb8")

        try:
            # save image
            img_data.resize(img_data.shape[0]+1, axis=0)
            img_data[-1] = im_rgb

            current_frame_ts = float(msg.header.stamp.to_nsec())/1e9
            frame_ts_collector = np.append(
                frame_ts_collector, [current_frame_ts], axis=0)
            print("Processed frame.")
        except TypeError:
            print("Some error")
            continue
bag.close()

# search for nearest event index
nearest_idx = 0
for frame_ts in frame_ts_collector:
    nearest_idx = find_nearest(event_ts_collector, nearest_idx, frame_ts)

    img_ind.resize(img_ind.shape[0]+1, axis=0)
    img_ind[-1] = nearest_idx

dataset.close()
