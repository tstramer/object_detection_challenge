import numpy as np
import tensorflow as tf

from object_detection.utils import visualization_utils as vis_util

class ObjectDetector:
  """ Base class representing an object detector."""

  def detect(self, image_np):
    """ 
    Given a numpy array representing an RGB image, return bounding
    boxes, classes, and confidence scores.
    """ 
    pass

  def label(self, image_np):
    """ 
    Given a numpy array representing an RGB image, return a image
    showing bounding boxes around detected objects
    """
    pass

class TFObjectDetector(ObjectDetector):

  def __init__(self, tf_session, detection_graph, category_index):
    self.tf_session = tf_session
    self.detection_graph = detection_graph
    self.category_index = category_index

  def detect(self, image_np):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
    classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = self.tf_session.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    return boxes, scores, classes

  def label(self, image_np):
    boxes, scores, classes = self.detect(image_np)

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        self.category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    return image_np