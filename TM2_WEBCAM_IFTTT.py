#怪怪ㄉ地方代表有個資不給看

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import numpy as np

from PIL import Image
from tflite_runtime.interpreter import Interpreter

import cv2

import time
import requests

import threading

Condition_1 = '0 Mantis'
Condition_2 = '1 Locust'
Condition_3 = '2 Other'

SCORE = 0.5

DELAY_TIME = 3.0

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  # print('input_tensor=',input_tensor)
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

def send_Line(str1,str2,str3):
    LINE_event_name = 'LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL'
    LINE_key = 'KKKKKKKKKKKKKKKKKKKKKKKKKKKK'
    # Your IFTTT LINE_URL with event name, key and json parameters (values)
    LINE_URL='https://maker.ifttt.com/trigger/' + LINE_event_name + '/with/key/' + LINE_key
    r = requests.post(LINE_URL, params={"value1":str1,"value2":str2,"value3":str3})

def send_Sheets(str1,str2,str3):
    SHEETS_event_name = 'SSSSSSSSSSSSSSSSSSSSSSSSSSSSS'
    SHEETS_key = 'KKKKKKKKKKKKKKKKKKKKKKKKKKK'
    # Your IFTTT LINE_URL with event name, key and json parameters (values)
    SHEETS_URL='https://maker.ifttt.com/trigger/' + SHEETS_event_name + '/with/key/' + SHEETS_key
    r = requests.post(SHEETS_URL, params={"value1":str1,"value2":str2,"value3":str3})

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
    '--video', help='Video number', required=False, type=int, default=0)

  args = parser.parse_args()

  model_file = args.model + '/model.tflite'

  labels_file = args.model + '/labels.txt'
 
  labels = load_labels(labels_file)
  
  interpreter = Interpreter(model_file)
  interpreter.allocate_tensors()

  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]
   
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  cap = cv2.VideoCapture(args.video)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

  key_detect = 0
  times=1
  t_flag = 0
  PRE_TIME = time.time()
  
  while (key_detect==0):
    ret,image_src =cap.read()

    frame_width=image_src.shape[1]
    frame_height=image_src.shape[0]

    cut_d=int((frame_width-frame_height)/2)
    crop_img=image_src[0:frame_height,cut_d:(cut_d+frame_height)]

    image_width = crop_img.shape[1]
    image_height = crop_img.shape[0]

    image=cv2.resize(crop_img,(224,224),interpolation=cv2.INTER_AREA)

    if (times==1):
      results = classify_image(interpreter, image)
      label_id, prob = results[0]

      print(labels[label_id],prob)
 
    cv2.putText(crop_img,labels[label_id] + " " + str(round(prob,2)),
      (5,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 12, cv2.LINE_AA)
    cv2.putText(crop_img,labels[label_id] + " " + str(round(prob,2)),
      (5,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 4, cv2.LINE_AA)

    cv2.imshow('Detecting....',crop_img)

    if ((labels[label_id] == Condition_1) and (prob >= SCORE)
      and (t_flag == 0) and (time.time() - PRE_TIME > DELAY_TIME)):
      PRE_TIME = time.time()
      t_Line = threading.Thread(target = send_Line,args=(labels[label_id], '自訂字串1' , '自訂字串2'))
      t_Line.start()
      t_Sheets = threading.Thread(target = send_Sheets,args=(labels[label_id], '自訂字串3' , '自訂字串4'))
      t_Sheets.start()
      t_flag = 1

    elif ((labels[label_id] == Condition_2) and (prob >= SCORE)
      and (t_flag == 0) and (time.time() - PRE_TIME > DELAY_TIME)):
      PRE_TIME = time.time()
      t_Line = threading.Thread(target = send_Line,args=(labels[label_id], '自訂字串5' , '自訂字串6'))
      t_Line.start()
      t_Sheets = threading.Thread(target = send_Sheets,args=(labels[label_id], '自訂字串7' , '自訂字串8'))
      t_Sheets.start()
      t_flag = 1

    if (t_flag == 1):
      t_Line.join()
      t_Sheets.join()
      t_flag = 0
    

    times=times+1
    if (times>10):
      times=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
      key_detect = 1

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
