#  -------------------------------------------------------------
#   Copyright (c) Cavedu.  All rights reserved.
#  -------------------------------------------------------------

import argparse
import json
import os

import numpy as np
from PIL import Image

import tflite_runtime.interpreter as tflite

import cv2

import time
import requests

import threading

Condition_1 = 'Mantis'
Condition_2 = 'Locust'

SCORE = 0.9

DELAY_TIME = 3.0

def get_prediction(image, interpreter, signature):
    # process image to be compatible with the model
    input_data = process_image(image, image_shape)

    # set the input to run
    interpreter.set_tensor(model_index, input_data)
    interpreter.invoke()

    # grab our desired outputs from the interpreter!
    # un-batch since we ran an image with batch size of 1, and convert to normal python types with tolist()
    outputs = {key: interpreter.get_tensor(value.get("index")).tolist()[0] for key, value in model_outputs.items()}

    # postprocessing! convert any byte strings to normal strings with .decode()
    for key, val in outputs.items():
        if isinstance(val, bytes):
            outputs[key] = val.decode()

    return outputs

def process_image(image, input_shape):
    width, height = image.size
    # ensure image type is compatible with model and convert if not
    input_width, input_height = input_shape[1:3]
    if image.width != input_width or image.height != input_height:
        image = image.resize((input_width, input_height))

    # make 0-1 float instead of 0-255 int (that PIL Image loads by default)
    image = np.asarray(image) / 255.0
    # format input as model expects
    return image.reshape(input_shape).astype(np.float32)

def send_Line(str1,str2,str3):
    LINE_event_name = 'LLLLLLLLLLLLLLLLLL'
    LINE_key = 'KKKKKKKKKKKKKKKKKKKKKK'
    # Your IFTTT LINE_URL with event name, key and json parameters (values)
    LINE_URL='https://maker.ifttt.com/trigger/' + LINE_event_name + '/with/key/' + LINE_key
    r = requests.post(LINE_URL, params={"value1":str1,"value2":str2,"value3":str3})

def send_Sheets(str1,str2,str3):
    SHEETS_event_name = 'SSSSSSSSSSSSSSSSSSSS'
    SHEETS_key = 'KKKKKKKKKKKKKKKKKKKKKK'
    # Your IFTTT LINE_URL with event name, key and json parameters (values)
    SHEETS_URL='https://maker.ifttt.com/trigger/' + SHEETS_event_name + '/with/key/' + SHEETS_key
    r = requests.post(SHEETS_URL, params={"value1":str1,"value2":str2,"value3":str3})
    
def main():    
    global signature_inputs
    global input_details
    global model_inputs
    global signature_outputs
    global output_details
    global model_outputs
    global image_shape
    global model_index

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model',
        help='Model path of saved_model.tflite and signature.json file',
        required=True)
    parser.add_argument('--video',
        help='Set video number of Webcam.',
        required=False, type=int, default=0)
    args = parser.parse_args()
    
    with open( args.model + "/signature.json", "r") as f:
        signature = json.load(f)

    model_file = signature.get("filename")

    interpreter = tflite.Interpreter(args.model + '/' + model_file)
    interpreter.allocate_tensors()
    # print('interpreter=',interpreter.get_input_details())

    # Combine the information about the inputs and outputs from the signature.json file with the Interpreter runtime
    signature_inputs = signature.get("inputs")
    input_details = {detail.get("name"): detail for detail in interpreter.get_input_details()}
    model_inputs = {key: {**sig, **input_details.get(sig.get("name"))} for key, sig in signature_inputs.items()}
    signature_outputs = signature.get("outputs")
    output_details = {detail.get("name"): detail for detail in interpreter.get_output_details()}
    model_outputs = {key: {**sig, **output_details.get(sig.get("name"))} for key, sig in signature_outputs.items()}
    image_shape = model_inputs.get("Image").get("shape")
    model_index = model_inputs.get("Image").get("index")

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    key_detect = 0
    times = 1
    t_flag = 0
    PRE_TIME = time.time()
    
    while (key_detect == 0):
        ret,image_src = cap.read()

        frame_width = image_src.shape[1]
        frame_height = image_src.shape[0]

        cut_d = int((frame_width-frame_height)/2)
        crop_img = image_src[0:frame_height,cut_d:(cut_d+frame_height)]

        image = Image.fromarray(cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB))

        if (times==1):
            prediction = get_prediction(image, interpreter, signature)

            # print('Result = '+ prediction["Prediction"])
            # print(prediction)

            Label_name = signature['classes']['Label'][prediction['Confidences'].index(max(prediction['Confidences']))]
            print(Label_name)
            print('Confidences = ' + str(max(prediction['Confidences'])))
 
        cv2.putText(crop_img, Label_name + " " +
            str(round(max(prediction['Confidences']),3)),
            (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0,255,255), 6, cv2.LINE_AA)
        cv2.putText(crop_img, Label_name + " " +
            str(round(max(prediction['Confidences']),3)),
            (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0,0,0), 2, cv2.LINE_AA)
        
        cv2.imshow('Detecting....',crop_img)

        Conf = round(max(prediction['Confidences']),3)

        if ((Label_name == Condition_1) and (Conf >= SCORE)
            and (t_flag == 0) and (time.time() - PRE_TIME > DELAY_TIME)):
            PRE_TIME = time.time()
            t_Line = threading.Thread(target = send_Line,args=(Label_name, '自訂字串1' , '自訂字串2'))
            t_Line.start()
            t_Sheets = threading.Thread(target = send_Sheets,args=(Label_name, '自訂字串3' , '自訂字串4'))
            t_Sheets.start()
            t_flag = 1

        elif ((Label_name == Condition_2) and (Conf >= SCORE)
              and (t_flag == 0) and (time.time() - PRE_TIME > DELAY_TIME)):
            PRE_TIME = time.time()
            t_Line = threading.Thread(target = send_Line,args=(Label_name, '自訂字串5' , '自訂字串6'))
            t_Line.start()
            t_Sheets = threading.Thread(target = send_Sheets,args=(Label_name, '自訂字串7' , '自訂字串8'))
            t_Sheets.start()
            t_flag = 1

        if (t_flag == 1):
            t_Line.join()
            t_Sheets.join()
            t_flag = 0
            
                      
        times=times+1
        if (times >= 10):
            times=1

        read_key = cv2.waitKey(1)
        if ((read_key & 0xFF == ord('q')) or (read_key == 27) ):
            key_detect = 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
