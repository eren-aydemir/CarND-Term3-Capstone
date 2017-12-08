from styx_msgs.msg import TrafficLight
from carla_classif import *

import numpy as np
import os
import tensorflow as tf
from collections import defaultdict

from PIL import Image
#import rospy

class TLClassifier(object):
    def __init__(self, imageType):
        self.imageType = imageType
        working_dir = os.path.dirname(os.path.realpath(__file__))
#        rospy.logerr("working_dir:{}".format(working_dir))
        if self.imageType == 'sim': 
            PATH_TO_MODEL = working_dir+'/sim_frcnn.pb'
        else:
            PATH_TO_MODEL = working_dir+'/real_frcnn.pb'

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, img):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})

        condition = scores > .5
        ex = np.extract(condition, classes)
        unq, cnt = np.unique(ex, return_counts=True)
        cntinds = cnt.argsort()
        most_probable = unq[cntinds[::-1]]
        tld_class =  int(most_probable.item(0)) if len(most_probable) > 0 else 4

        # Map TrafficLight message values with training classes 
        if tld_class == 1:
            return TrafficLight.GREEN
        elif tld_class == 2:
            return TrafficLight.RED
        elif tld_class == 3:
            #Treat yellow as red to increase react time
            return TrafficLight.RED #TrafficLight.YELLOW
        else:
            return TrafficLight.UNKNOWN

