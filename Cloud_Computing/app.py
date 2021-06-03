#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import base64
import io
import os
import pathlib
import sys
import tempfile
import json

from werkzeug.utils import append_slash_redirect, secure_filename

MODEL_BASE = '/opt/models/research'
sys.path.append(MODEL_BASE)
sys.path.append(MODEL_BASE + '/object_detection')
sys.path.append(MODEL_BASE + '/slim')
PATH_TO_LABELS = MODEL_BASE + '/object_detection/data/label_map.pbtxt'

from flask import Flask
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from flask import send_file
from flask import jsonify
from flask_wtf.file import FileField
import numpy as np
from PIL import Image
from PIL import ImageDraw
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from werkzeug.datastructures import CombinedMultiDict
from wtforms import Form
from wtforms import ValidationError
from tensorflow.python.keras.models import load_model

# Patch the location of gfile
tf.gfile = tf.io.gfile


app = Flask(__name__)


content_types = {'jpg': 'image/jpeg',
                 'jpeg': 'image/jpeg',
                 'png': 'image/png'}
extensions = sorted(content_types.keys())


def is_image():
  def _is_image(form, field):
    if not field.data:
      raise ValidationError()
    elif field.data.filename.split('.')[-1].lower() not in extensions:
      raise ValidationError()

  return _is_image


class PhotoForm(Form):
  input_photo = FileField(
      'File extension should be: %s (case-insensitive)' % ', '.join(extensions),
      validators=[is_image()])


class ObjectDetector(object):

  def __init__(self):

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)
    model_dir = '/opt/acne_detection/model/saved_model'
    #model = tf.saved_model.load(str(model_dir))
    model = load_model(str(model_dir))
    model = model.signatures['serving_default']
    self.model = model

  def _load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

  def detect(self, image):
    image_np = self._load_image_into_numpy_array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis,...]
    output_dict = self.model(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                   for key,value in output_dict.items()}
    boxes = output_dict['detection_boxes']
    classes = output_dict['detection_classes'].astype(np.int64)
    scores = output_dict['detection_scores']
    return boxes, scores, classes, num_detections


def draw_bounding_box_on_image(image, box, color='red', thickness=4):
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  ymin, xmin, ymax, xmax = box
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)


def encode_image(image):
  image_buffer = io.BytesIO()
  image.save(image_buffer, format='PNG')
  imgstr = 'data:image/png;base64,{:s}'.format(
      base64.b64encode(image_buffer.getvalue()).decode().replace("'", ""))
  return imgstr


def detect_objects(image_path):
  image = Image.open(image_path).convert('RGB')
  boxes, scores, classes, num_detections = client.detect(image)
  image.thumbnail((480, 480), Image.ANTIALIAS)
  count = 0
  new_images = {}
  for i in range(num_detections):
    if scores[i] < 0.5: continue
    cls = classes[i]
    if cls not in new_images.keys():
      new_images[cls] = image.copy()
    draw_bounding_box_on_image(new_images[cls], boxes[i],
                               thickness=int(scores[i]*10)-4)
    count = count + 1

  result = {}
  result['original'] = encode_image(image.copy())

  for cls, new_image in new_images.items():
    category = client.category_index[cls]['name']
    result[category] = encode_image(new_image)
  converted_count = str(count)
  counting(converted_count)
  return result, converted_count

def counting(con_count):
  global str_count #bocor
  str_count = con_count

@app.route('/')
def upload():
  photo_form = PhotoForm(request.form)
  return render_template('index.html', photo_form=photo_form, result={}, count="")

#Web application API
@app.route('/detection', methods=['GET', 'POST'])
def post():
  form = PhotoForm(CombinedMultiDict((request.files, request.form)))
  if request.method == 'POST' and form.validate():
    with tempfile.NamedTemporaryFile() as temp:
      form.input_photo.data.save(temp)
      temp.flush()
      result, count = detect_objects(temp.name)

    photo_form = PhotoForm(request.form)
    return render_template('index.html',
                           photo_form=photo_form, result=result, count=count)
    
  else:
    return redirect(url_for('upload'))

@app.route('/recommendations', methods=['GET'])
def recom():
  count = int(str_count)
  if count<=5 and count>0:
    recoms = {
      "level" : "Low/Mild",
      "desc" : "Hardly visible from 2.5 meters away; a few scattered comedones and a few small papules; and very few pustules, comedones, and papules",
      "recom" : "Over-the-counter medicine helps! Benzoyl peroxide is a bactericidal agent that comes in the form of creams and gels in a variety of manners and is proven to be the most effective for treatment of mild to moderate mixed acne when used in combination with topical retinoids!\nA diet plan may also help in complementing this treatment regime as there have been numerous studies that show that there is a correlation between chocolate and milk consumption with acne/lesions breakout.\nTherapy through skin care clinics should also be considered. For Asian patients, there have been studies that show Chemical peels for acne and acne scars have been shown to be safe and effective. "
    }
    
  elif count > 5 and count <=10:
    recoms = {
      "level" : "Moderate",
      "desc" : "Easily recognizable; less than half of the affected area is involved; many comedones, papules, and pustules",
      "recom" : "Over-the-counter medicine helps! Benzoyl peroxide is a bactericidal agent that comes in the form of creams and gels in a variety of manners and is proven to be the most effective for treatment of mild to moderate mixed acne when used in combination with topical retinoids!\nYou may also consider getting oral antibiotics. Based on expert consensus on relative effectiveness, they recommend using doxycycline and minocycline (minocin). Taking a dosage of 50 to 100 mg once or twice per day. It should be noted however that once the severity of the acne has lowered, that you should stop taking oral antibiotics and instead switch over to over-the-counter medicine instead.\nTherapy through skin care clinics should also be considered. For Asian patients, there have been studies that show Chemical peels for acne and acne scars have been shown to be safe and effective. Photodynamic Therapy have been shown to be effective in moderate-to-severe acne."
    }
  elif count > 10 and count <=20:
    recoms = {
      "level" : "Severe",
      "desc" : "Entire area is involved; covered with comedones, numerous pustules and papules, a few nodules and cysts.",
      "recom" : "Consult with a dermatologist immediately. Oral antibiotics helps a lot but needs a doctor's prescription. Based on expert consensus on relative effectiveness, they recommend using doxycycline and minocycline (minocin). Taking a dosage of 50 to 100 mg once or twice per day. A minimum duration of six weeks is commonly required to see clinical improvement but it is not recommended to continue using these oral options for more than 3-4 months. However, these should be taken with other topical agents such as topical retinoids which has the full recommendation of experts!\nIt should be noted however that once the severity of the acne has lowered to mild levels, you should stop taking oral antibiotics and instead switch over to over-the-counter medicine such as topical retinoids that come in the form of cream and gels.\nTherapy through skin care clinics should also be considered. Photodynamic Therapy have been shown to be effective in moderate-to-severe acne."
    }
  elif count > 20:
    recoms = {
      "level" : "Very Severe",
      "desc" : "Highly inflammatory acne covering the affected area, nodules and cysts present",
      "recom" : "Consult with a dermatologist immediately, there are many studies that show that oral isotretinoin is effective for severe recalcitrant acne, however, it requires a doctor’s discretion or done in accordance with the region’s medical regulations.\nThere are many variables which come into play with these levels of acne, and it should be noted that therapic measures from expert dermatologists will help the most."
    }
  else :
    recoms = {
      "level" : "None",
      "desc"  : "Your Face is Clean",
      "recom" : "Stay Healthy"
    }
  return render_template("rekom.html", recoms=recoms)

#mobile API
#mobile version 2
@app.route('/mob/detectionv2', methods=['GET', 'POST'])
def mob_post():
  if request.method == 'POST':
    imgfile = request.files['img']
    filename = secure_filename(imgfile.filename)
    with tempfile.NamedTemporaryFile() as temp:
      imgfile.save(temp)  
      temp.flush()
      result, count = detect_objects(temp.name)
    count = int(count)
    if count < 1 :
      result['fore']=result['original']
      return json.dumps(result)
    else: return json.dumps(result)
    
  else :
    stats_res={'fore': 'Ada Kesalahan di Upload atau filenya'}
    return json.dumps(stats_res)

@app.route('/mob/treatment', methods=['GET'])
def mobcount():
  return str_count #string

@app.route('/test/connection')
def tes():
  return "You're Connected!!"

client = ObjectDetector()


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80, debug=False)
