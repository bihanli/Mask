from itertools import product
from math import sqrt
import tensorflow as tf

class Anchor(object):
    def __init__(self, img_size, feature_map_size, aspect_ratio, scale):
        """
        :param img_size:
        :param feature_map_size:
        :param aspect_ratio:
        :param scale:
        """
        self.num_anchors, self.priors = self._generate_anchors(img_size, feature_map_size, aspect_ratio, scale)

    def _generate_anchors(self, img_size, feature_map_size, aspect_ratio, scale):
        """
        :param img_size:
        :param feature_map_size:
        :param aspect_ratio:
        :param scale:
        :return:
        """
        prior_boxes = []
        num_anchors = 0
        for idx, f_size in enumerate(feature_map_size):
            # print("Create priors for f_size:%s", f_size)
            count_anchor = 0
            for j, i in product(range(f_size), range(f_size)):
                x = (i + 0.5) / f_size
                y = (j + 0.5) / f_size
                for ars in aspect_ratio:
                    a = sqrt(ars)
                    w = scale[idx] * a / img_size
                    h = scale[idx] / a / img_size

                    # original author make all priors squre
                    h = w

                    # directly use point form here => [ymin, xmin, ymax, xmax]
                    ymin = y - (h / 2.)
                    xmin = x - (w / 2.)
                    ymax = y + (h / 2.)
                    xmax = x + (w / 2.)
                    prior_boxes += [ymin * img_size, xmin * img_size, ymax * img_size, xmax * img_size]
                count_anchor += 1
            num_anchors += count_anchor
            # print(f_size, count_anchor)
        output = tf.reshape(tf.convert_to_tensor(prior_boxes), [-1, 4])
        return num_anchors, output
    def get_anchors(self):
        return self.priors