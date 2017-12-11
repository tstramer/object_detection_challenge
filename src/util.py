import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from PIL import Image

def load_tf_graph(model_path):
  """Load the variables from a model into a tf graph"""
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  return detection_graph

def load_image_into_numpy_array(image_path):
  """Load an RGB image into a numpy array"""
  image = Image.open(image_path)
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def create_category_index(labels_path, max_classes):
  """Create a index from category id to name"""
  label_map = label_map_util.load_labelmap(labels_path)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=max_classes, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)
  return category_index