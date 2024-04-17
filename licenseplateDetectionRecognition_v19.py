# Version 19
# Author: Zhiyi Li
# Date: April 13rd, 2024
# Ultralytics YOLO 🚀, GPL-3.0 license
# Add set for filtering 
import hydra
import torch
import sys

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

import easyocr
import cv2
import os
import datetime
from datetime import datetime
from datetime import time
import numpy as np
import yaml
from difflib import SequenceMatcher
import shutil
from postProcessing import *

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def time_in_range(start, end, x):
    """Return true if x is in the range [start, end]"""
    if start <= end:
        return start <= x <= end
    else:
        return start <= x or x <= end

reader = easyocr.Reader(['en'], gpu=True)

def perform_ocr_on_image(img, coordinates):
    x, y, w, h = map(int, coordinates)
    cropped_img = img[y:h, x:w]

    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray_img)

    text = ""
    for res in results:
        if len(results) == 1 or (len(res[1]) > 6 and res[2] > 0.2):
            text = res[1]

    return str(text)

def record(label, label_set):
    # Validate the license plate.   
    label = label.strip()
    print("label: ", label)
    print("label_set: ", label_set)
    
    log_file_name = "./samples/log.txt"
    check_file = os.path.exists(log_file_name)
     
    if not check_file: 
        with open(log_file_name, "w") as f:
            outLine = "Station_id" + "," + "timestamp" + "," + "licenseplate" + "\n"
            f.write(outLine)
        f.close()

    stationID = "Test"
    now_time = datetime.now()
     
    with open(log_file_name, "a") as f:
        outLine = stationID + "," + str(now_time) + "," +  str(label) + "\n"  
        f.write(outLine)

    f.close()
    return

# Read IP address and virtual lines from yml configuration file
def getIP():
    with open('samples/camera1.yml', 'r') as file: 
        config_service = yaml.safe_load(file)

    print (config_service)

    IP_address = config_service['camera']['IP_address']
    return IP_address

class DetectionPredictor(BasePredictor):
    
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):      
        
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        # save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        # Apply post process. 
        cur_time = datetime.now()
        current_time_str = cur_time.strftime('%H:%M:%S')
        print("cur_time: ", cur_time)
        print("cur_time_str: ", current_time_str)
        
        # start = time.time(14, 25, 00)
        # end = time.time(14, 30, 00) 
        # if time_in_range(start, end, cur_time):
        if current_time_str == "23:59:00":
            print("In range")
            print("cur_time: ", cur_time)
            postProcessing()
            print("postProcessing done")
            time.sleep(2)  
            # sys.exit(0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                
                
                text_ocr = perform_ocr_on_image(im0,xyxy)
                label = text_ocr
                  
                self.annotator.box_label(xyxy, label, color=colors(c, True))
                
                print ("label: ", label)
                
                # Only record when near the center of image
                # Location constraints, only record when near the center of image.
                                
                [height, width, _] = im0.shape
                center_x = int(width/2)
                center_y = int(height/2)

                x, y, w, h = map(int, xyxy)
                
                dx = center_x - x
                dy = center_y - y
                distance = np.sqrt(dx ** 2 + dy ** 2)
                # print("center_x: ", center_x)
                # print("center_y: ", center_y)
                # print("x: ", x)
                # print("y: ", y)

                print("distance: ", distance)
                
                threshold = 800  
                candidate = True
                if distance > threshold:
                    # The centroid is moving
                    # (do something here, e.g. draw a green bounding box)
                    candidate = False
                
                if candidate == True:
                    record(label, label_set)

                
                                                          
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)
  
        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt" #"best.pt"  
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    # Create a set and update every 5 mins 
    label_set = set()
    IP_address = getIP()
    print("IP_address: ", IP_address)
    start_time = datetime.now()
    already_process = False
    predict()
    
