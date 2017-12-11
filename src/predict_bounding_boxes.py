#!/usr/local/bin/python3

"""
Predict bounding boxes for a JPG image or AVI video and
output a image / video with the bounding boxes overlayed.
"""

import os
import time
import argparse
import numpy as np
import tensorflow as tf
from moviepy.editor import VideoFileClip
import scipy.misc

from object_detector import TFObjectDetector
from util import load_tf_graph, load_image_into_numpy_array, create_category_index

def parse_args():
  """Parse input arguments"""
  parser = argparse.ArgumentParser(
                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--in_path', type=str,
                      help='path to JPEG or AVI file',
                      required=True)
  parser.add_argument("--out_path", 
                      help="path to output labeled image or video to", 
                      required=True)
  parser.add_argument('--model_path', type=str,
                      help='path to model graph (pb format)',
                      required=True)
  parser.add_argument('--labels_path', type=str,
                      help='path to class labels (pbtxt format)',
                      required=True)
  parser.add_argument('--num_classes', type=int, default=3,
                      help='number of object classes',
                      required=True)

  return parser.parse_args()

def label_video(object_detector, vid_path, out_path):
  """Write a new video with bounding boxes overlayed to disk"""
  clip = VideoFileClip(vid_path)
  labeled_clip = clip.fl_image(object_detector.label) 
  labeled_clip.write_videofile(out_path, audio=False, codec='mpeg4', fps=4)

def label_img(object_detector, img_path, out_path):
  """Write a new image with bounding boxes overlayed to disk"""
  image_np = load_image_into_numpy_array(img_path)
  labeled_img = object_detector.label(image_np)
  scipy.misc.imsave(out_path, labeled_img)

if __name__ == "__main__":
  args = parse_args()
  print ("Args: {}\n".format(args.__dict__))

  detection_graph = load_tf_graph(args.model_path)
  category_index = create_category_index(args.labels_path, args.num_classes)

  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      object_detector = TFObjectDetector(sess, detection_graph, category_index)
      _, ext = os.path.splitext(args.in_path)
      if ext == ".avi":
        label_video(object_detector, args.in_path, args.out_path)
      elif ext == ".jpg":
        label_img(object_detector, args.in_path, args.out_path)
      else:
        exit("Invalid input file type (expected avi or jpg)")
      print ("Output written to {}".format(args.out_path))