#!/usr/local/bin/python3

"""Convert a list of frames in JPEG format to an AVI video."""

import cv2
import argparse
import os

def parse_args():
  """Parse input arguments"""
  parser = argparse.ArgumentParser(
                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--path', '-p', type=str,
                      help='path to file containing frame file names in order (one per line)',
                      required=True)
  parser.add_argument('--base_dir', type=str,
                      help='base directory where frame files are located',
                      required=True)
  parser.add_argument("--out_path", 
                      help="output video file path",
                      required=True)

  return parser.parse_args()

def get_size(args, frame_paths):
  """Determine the width and height from the first frame"""
  frame_path = os.path.join(args.base_dir, frame_paths[0])
  frame = cv2.imread(frame_path)
  return frame.shape

def write_frames(args, frame_names, out):
  for frame_name in frame_names:
    frame_path = os.path.join(args.base_dir, frame_name)
    frame = cv2.imread(frame_path)

    out.write(frame) # Write out frame to video

if __name__ == "__main__":
  args = parse_args()
  print ("Args: {}\n".format(args.__dict__))

  # Get the list of frames
  with open(args.path) as f:
    frames = [line.strip() for line in f.readlines()]

  height, width, channels = get_size(args, frames)

  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Be sure to use lower case
  out = cv2.VideoWriter(args.out_path, fourcc, 20.0, (width, height))

  # Write the frames
  write_frames(args, frames, out)

  # Release everything if job is finished
  out.release()

  print("Output written to {}".format(args.out_path))
