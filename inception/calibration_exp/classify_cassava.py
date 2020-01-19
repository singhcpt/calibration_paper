# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This file strips the model's confidence for its classification for each image
# Can be repurposed to just classify images 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255,
                                random_brightness=0,
                                random_crop=0,
                                random_scale=0,
                                random_theta=0,
                                random_shear=0,
                                random_translate_x=0,
                                random_translate_y=0,
                                random_standard_dev=0.0):
                                
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

  ## DISTORTIONS ##
  margin_scale = 1.0 + (random_crop / 100.0)
  resize_scale = 1.0 + (random_scale / 100.0)
  margin_scale_value = tf.constant(margin_scale)
  resize_scale_value = tf.random_uniform(shape=[],
                                         minval=1.0,
                                         maxval=resize_scale)
  scale_value = tf.multiply(margin_scale_value, resize_scale_value)
  precrop_width = tf.multiply(scale_value, input_width)
  precrop_height = tf.multiply(scale_value, input_height)
  precrop_shape = tf.stack([precrop_height, precrop_width])
  precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
  precropped_image = tf.image.resize_bilinear(normalized,
                                              precrop_shape_as_int)
  precropped_image_3d = tf.squeeze(precropped_image, axis=[0])
  cropped_image = tf.random_crop(precropped_image_3d,
                                 [input_height, input_width, 3])

  brightness_min = 1.0 - (random_brightness / 100.0)
  brightness_max = 1.0 + (random_brightness / 100.0)
  brightness_value = tf.random_uniform(shape=[],
                                       minval=brightness_min,
                                       maxval=brightness_max)
  brightened_image = tf.multiply(cropped_image, brightness_value)

  # Custom augmentations
  '''
  skewed_image = tf.keras.preprocessing.image.apply_affine_transform(
    tf.Session().run(brightened_image),
    theta=random.randrange(0, random_theta, 1),
    tx=0,
    ty=0,
    shear=random.randrange(0, random_shear, 1),
    zx=1,
    zy=1,
    row_axis=0,
    col_axis=1,
    channel_axis=2,
    fill_mode='nearest',
    cval=0.0,
    order=1
  )
  '''
  rotated_image = tf.contrib.image.rotate(brightened_image, math.radians(random.randrange(0,120)))
  translated_image = tf.contrib.image.translate(rotated_image, translations=[random.randrange(0, random_translate_x, 1), random.randrange(0, random_translate_y, 1)])
  noise = tf.random_normal(shape=tf.shape(translated_image), mean=0.0, stddev=randrange_float(0, random_standard_dev, 0.001), dtype=tf.float32)
  noise_image = translated_image + noise

  distort_result = tf.expand_dims(noise_image, 0, name='DistortResult')

  #sess = tf.Session()
  #result = sess.run(distort_result)

  #show_image(distort_result, normalized)

  return result

def randrange_float(start, stop, step):
    return random.randint(0, int((stop - start) / step)) * step + start

def show_image(distort, norm):
  show_sess = tf.Session()
  darr = show_sess.run(distort)
  darr_ = np.squeeze(darr)

  narr = show_sess.run(norm)
  narr_ = np.squeeze(narr)

  plt.imshow(narr_)
  plt.show()
  plt.imshow(darr_)
  plt.show()

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
  #file_name = "/Users/singhcpt/dev/calibration_exp/data/corn_gray_leaf_spot/Corn_G.L.S_2647.JPG"
  #model_file = \
  #  "/Users/singhcpt/dev/calibration_exp/data/retrained_graph.pb"
  # label_file = "/tmp/output_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "Placeholder"
  output_layer = "final_result"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)
  t = read_tensor_from_image_file(
      file_name,
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  output_operation = graph.get_operation_by_name(output_name)
  input_operation = graph.get_operation_by_name(input_name)

  with tf.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)
  for i in top_k:
    print(labels[i], results[i])