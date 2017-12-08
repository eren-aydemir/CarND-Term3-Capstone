import numpy as np
import os
import tensorflow as tf

from PIL import Image

PATH_TO_MODEL = './data/frozen_inference_graph.pb'
NUM_CLASSES = 4

PATH_TO_TEST_IMAGES_DIR = './util/data/image'

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


class Classifier(object):
    def __init__(self):
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
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
        cl = {1:0, 2:0, 3:0, 4:0 }
        for i in range(int(num)):
            if scores.item(i) > .05:
                cl[classes.item(i)] += scores.item(i)
        return max(cl, key=cl.get)


if __name__ == '__main__':
    imgs = [ PATH_TO_TEST_IMAGES_DIR + '/' + f for f in os.listdir(PATH_TO_TEST_IMAGES_DIR) if f.endswith('jpg') ]
    imgs = imgs[:100]
    # print TEST_IMAGE_PATHS

    print ("initialize..")
    classif = Classifier()

    for image_path in imgs:
        image = Image.open(image_path)
        image_np = load_image_into_numpy_array(image)

        print ()
        print ("image:", image_path)
        print ("class:", classif.get_classification(image_np))
