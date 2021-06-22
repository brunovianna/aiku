# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import collections
from logging import error

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

import requests
from bs4 import BeautifulSoup
import urllib.request
import signal

import sys, serial
from io import BytesIO

try:
    # Python2
    import Tkinter as tk
except ImportError:
    # Python3
    import tkinter as tk
from PIL import Image, ImageTk

Object = collections.namedtuple('Object', ['label', 'score', 'bbox'])


def tiles_location_gen(img_size, tile_size, overlap):
  """Generates location of tiles after splitting the given image according the tile_size and overlap.

  Args:
    img_size (int, int): size of original image as width x height.
    tile_size (int, int): size of the returned tiles as width x height.
    overlap (int): The number of pixels to overlap the tiles.

  Yields:
    A list of points representing the coordinates of the tile in xmin, ymin,
    xmax, ymax.
  """

  tile_width, tile_height = tile_size
  img_width, img_height = img_size
  h_stride = tile_height - overlap
  w_stride = tile_width - overlap
  for h in range(0, img_height, h_stride):
    for w in range(0, img_width, w_stride):
      xmin = w
      ymin = h
      xmax = min(img_width, w + tile_width)
      ymax = min(img_height, h + tile_height)
      yield [xmin, ymin, xmax, ymax]


def non_max_suppression(objects, threshold):
  """Returns a list of indexes of objects passing the NMS.

  Args:
    objects: result candidates.
    threshold: the threshold of overlapping IoU to merge the boxes.

  Returns:
    A list of indexes containings the objects that pass the NMS.
  """
  if len(objects) == 1:
    return [0]

  boxes = np.array([o.bbox for o in objects])
  xmins = boxes[:, 0]
  ymins = boxes[:, 1]
  xmaxs = boxes[:, 2]
  ymaxs = boxes[:, 3]

  areas = (xmaxs - xmins) * (ymaxs - ymins)
  scores = [o.score for o in objects]
  idxs = np.argsort(scores)

  selected_idxs = []
  while idxs.size != 0:

    selected_idx = idxs[-1]
    selected_idxs.append(selected_idx)

    overlapped_xmins = np.maximum(xmins[selected_idx], xmins[idxs[:-1]])
    overlapped_ymins = np.maximum(ymins[selected_idx], ymins[idxs[:-1]])
    overlapped_xmaxs = np.minimum(xmaxs[selected_idx], xmaxs[idxs[:-1]])
    overlapped_ymaxs = np.minimum(ymaxs[selected_idx], ymaxs[idxs[:-1]])

    w = np.maximum(0, overlapped_xmaxs - overlapped_xmins)
    h = np.maximum(0, overlapped_ymaxs - overlapped_ymins)

    intersections = w * h
    unions = areas[idxs[:-1]] + areas[selected_idx] - intersections
    ious = intersections / unions

    idxs = np.delete(
        idxs, np.concatenate(([len(idxs) - 1], np.where(ious > threshold)[0])))

  return selected_idxs


def draw_object(draw, obj):
  """Draws detection candidate on the image.

  Args:
    draw: the PIL.ImageDraw object that draw on the image.
    obj: The detection candidate.
  """
  draw.rectangle(obj.bbox, outline='red')
  textbox_w, textbox_h = app.font.getsize(obj.label)
  draw.rectangle ((obj.bbox[0]+2,obj.bbox[3]-22,obj.bbox[0]+2+textbox_w,obj.bbox[3]-22+textbox_h),fill='yellow')
  draw.text((obj.bbox[0]+2, obj.bbox[3] - 22), obj.label, fill='#0000', font=app.font)
  ##draw.rectangle ((obj.bbox[0]+2,obj.bbox[3]-22,obj.bbox[0]+2+textbox_w,obj.bbox[3]-22+textbox_h),fill='yellow')
  ##draw.text((obj.bbox[0]+2, obj.bbox[3] - 22), str(obj.score), fill='#0000', font=app.font)


def reposition_bounding_box(bbox, tile_location):
  """Relocates bbox to the relative location to the original image.

  Args:
    bbox (int, int, int, int): bounding box relative to tile_location as xmin,
      ymin, xmax, ymax.
    tile_location (int, int, int, int): tile_location in the original image as
      xmin, ymin, xmax, ymax.

  Returns:
    A list of points representing the location of the bounding box relative to
    the original image as xmin, ymin, xmax, ymax.
  """
  bbox[0] = bbox[0] + tile_location[0]
  bbox[1] = bbox[1] + tile_location[1]
  bbox[2] = bbox[2] + tile_location[0]
  bbox[3] = bbox[3] + tile_location[1]
  return bbox


def detect_objects(img):


  tile_sizes = '1352x900,500x500,250x250'
  iou_threshold = .1
  tile_overlap = 15
  score_threshold = 0.5
  label = "/home/pi/apps/coral/test_data/coco_labels.txt"
  model = "/home/pi/apps/coral/test_data/ssd_mobilenet_v2_coco_quant_no_nms_edgetpu.tflite"

  interpreter = make_interpreter(model)
  interpreter.allocate_tensors()
  labels = read_label_file(label) if label else {}

  draw = ImageDraw.Draw(img)

  objects_by_label = dict()
  img_size = img.size
  tile_sizes = [
      map(int, tile_size.split('x')) for tile_size in tile_sizes.split(',')
  ]
  for tile_size in tile_sizes:
    for tile_location in tiles_location_gen(img_size, tile_size,
                                            tile_overlap):
      tile = img.crop(tile_location)
      try:
        _, scale = common.set_resized_input(
          interpreter, tile.size,
          lambda size, img=tile: img.resize(size, Image.NEAREST))
      except:
        print ("wrong size: "+str(tile.size))
      else:
        interpreter.invoke()
        objs = detect.get_objects(interpreter, score_threshold, scale)

        for obj in objs:
            bbox = [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax]
            bbox = reposition_bounding_box(bbox, tile_location)

            label = labels.get(obj.id, '')
            objects_by_label.setdefault(label,
                                        []).append(Object(label, obj.score, bbox))

  prompt = ""
  for label, objects in objects_by_label.items():
    idxs = non_max_suppression(objects, iou_threshold)
    for idx in idxs:
      draw_object(draw, objects[idx])
      print (objects[idx].label+" score: "+str(objects[idx].score))
      prompt = prompt+objects[idx].label+"+"
    

  if len(objects_by_label)==0:
    msg = "No object found"
    textbox_w, textbox_h = app.font.getsize(msg)
    try:
      draw.rectangle ((2,2,2+textbox_w,2+textbox_h),fill='#ffff00')
      draw.text((2, 2), msg, fill='#000000', font=app.font)
    except:
      print ("color exception ")

  return (img,prompt)


class App(tk.Tk):
    '''Tk window/label adjusts to size of image'''
    def __init__(self, x, y, delay):
        # the root will be self
        tk.Tk.__init__(self)

        self.attributes("-fullscreen", True)
        self.configure(bg='black')
        self.bind("<Escape>", lambda e: self.quit())


        # set x, y position only
        self.geometry('+{}+{}'.format(x, y))
        self.delay = delay

        self.w, self.h = self.winfo_screenwidth(), self.winfo_screenheight()

        self.font = ImageFont.truetype("Bitter-ExtraBold.ttf", 20)
        # allows repeat cycling through the pictures
        # store as (img_object, img_name) tuple
        self.picture_display = tk.Label(self)
        self.picture_display.pack()

        try:
          self.serialPort = serial.Serial(port = "/dev/ttyUSB0", baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
        except:
          print ("couldn't find oled display")



    def get_image(self):
        try:
          r = requests.get("https://en.wikipedia.org/wiki/Special:Random/File")
        except:
          print ("Network error:")
          self.after(self.delay, self.get_image)

        else:
          htmldata = r.text
          soup = BeautifulSoup(htmldata, 'html.parser')
          image_url = "https:"+soup.find_all('img')[0]['src']
          for item in soup.find_all('li'):

              if ('class' in item.attrs):
                  if ('mw-imagepage-linkstoimage-ns0' in item.attrs['class']):
                      print (item.a.string)


          tokens = image_url.split('.')
          image_type = tokens[len(tokens)-1]


          save_name = 'temp_image.'+image_type  #local name to be saved
          try:
            urllib.request.urlretrieve(image_url, save_name)
          except:
            print ("network error")
          else:
            img = Image.open(save_name)

            imgWidth, imgHeight = img.size
            #if imgWidth > self.w or imgHeight > self.h:
            ratio = min(self.w/imgWidth, self.h/imgHeight)
            imgWidth = int(imgWidth*ratio)
            imgHeight = int(imgHeight*ratio)
            img = img.resize((imgWidth,imgHeight), Image.ANTIALIAS)


            

            img, prompt = detect_objects (img)
            
            #for some reasons, it can't load some formats
            try:
              self.ph = ImageTk.PhotoImage(img)
            except:
              print ("couldn't load image")
            else:
              self.picture_display.config(image=self.ph)
              self.picture_display.update()
              print ("just showed "+image_url)

              response = BytesIO()
              if (prompt!=""):
                  prompts = prompt.split('+')
                  clean_prompt = ""
                  for oneword in prompts:
                      if  (clean_prompt.find(oneword))==-1:
                          clean_prompt = clean_prompt + oneword + "+"
                      
                  print ("prompts "+str(prompts))
                  print ("clean prompt: "+clean_prompt)

                  readable_prompt = ", ".join(clean_prompt.split('+'))
                  readable_prompt = readable_prompt[0:len(readable_prompt)-2]
                  readable_prompt = readable_prompt+"\n"
                  

                  response.write (b"--print--|thinking of...|")
                  response.write (readable_prompt.encode('utf-8')) 
                  self.serialPort.write(response.getvalue())

                  try:
                      r = requests.get("http://172.23.61.137:8070/"+clean_prompt)
                  except:
                      print ("Network error:")

                  print ("haikus: "+r.text)
                  #response.write ("GET request for haiku {}".format(self.path).encode('utf-8'))
                  response.write (b"--print--")
                  if (r.text != ""):
                      response.write (r.text.encode('utf-8')) 
                      delay = 7000
                  else:
                      response.write (b"|not inspired, sorry") 
                      delay = 2500
                  self.serialPort.write(response.getvalue())
              else:
                  response.write (b"--print--|Nothing found")
                  self.serialPort.write(response.getvalue())
                  delay = 2500

                

        self.after(self.delay, self.get_image)
        

    def run(self):
        self.mainloop()

def sigint_handler(sig, frame):
    app.quit()
    app.update()

# set milliseconds time between slides
delay = 7000

app = App(0, 0, delay)

signal.signal(signal.SIGINT, sigint_handler)

app.get_image()
app.run()
