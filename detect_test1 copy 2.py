import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from random import randint

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from los import density_calc

from datetime import datetime

#For SORT tracking
import skimage
from sort import *

from datetime import datetime


tracked_objects = {}

def update_objects(detected_objects, current_time):
    global tracked_objects
    for obj in detected_objects:
        obj_id = obj['id']
        bbox = {'x1': obj['x1'], 'y1': obj['y1'], 'x2': obj['x2'], 'y2': obj['y2']}
        roi = obj['roi_position']
        if obj_id not in tracked_objects:
            tracked_objects[obj_id] = {
                "last_position": bbox,
                "roi_position": roi,
                "timestamp": current_time,
                "duration": 0
            }
        else:
            last_obj = tracked_objects[obj_id]
            last_obj["last_position"] = bbox
            
            if last_obj["roi_position"] != roi:
                last_obj["roi_position"] = roi
                last_obj["timestamp"] = current_time.strftime("%H:%M:%S")
                last_obj["duration"] = 0
            

def determine_roi(x1, y1, x2, y2, im0) :
    y_middle = ((y2 - y1) / 2) + y1
    x_middle = ((x2 - x1) / 2) + x1

    roi1_x1, roi1_y1, roi1_x2, roi1_y2 = (int(0.124*im0.shape[1]),int(0.301*im0.shape[0]),int(0.310*im0.shape[1]),int(0.565*im0.shape[0]))
    roi2_x1, roi2_y1, roi2_x2, roi2_y2 = (int(0.395*im0.shape[1]),int(0.301*im0.shape[0]),int(0.620*im0.shape[1]),int(0.621*im0.shape[0]))
    roi3_x1, roi3_y1, roi3_x2, roi3_y2 = (int(0.620*im0.shape[1]),int(0.445*im0.shape[0]),int(0.806*im0.shape[1]),int(0.621*im0.shape[0]))
    roi4_x1, roi4_y1, roi4_x2, roi4_y2 = (int(0.077*im0.shape[1]),int(0.565*im0.shape[0]),int(0.404*im0.shape[1]),int(im0.shape[0]))
    roi5_x1, roi5_y1, roi5_x2, roi5_y2 = (int(0.521*im0.shape[1]),int(0.621*im0.shape[0]),int(0.758*im0.shape[1]),int(im0.shape[0]))
    roi6_x1, roi6_y1, roi6_x2, roi6_y2 = (int(0.758*im0.shape[1]),int(0.621*im0.shape[0]),int(im0.shape[1]),int(im0.shape[0]))


    # Check which ROI the detection falls into
    if roi1_x1 < x_middle < roi1_x2 and roi1_y1 < y_middle < roi1_y2:
        return "A"
        
    elif roi2_x1 < x_middle < roi2_x2 and roi2_y1 < y_middle < roi2_y2:
        return "B"
        
    elif ((roi3_x1 < x_middle < roi3_x2 and roi3_y1 < y_middle < roi3_y2)):
        return "C"
        
    elif ((roi4_x1 < x_middle < roi4_x2 and roi4_y1 < y_middle < roi4_y2)):
        return "D"
    
    elif ((roi5_x1 < x_middle < roi5_x2 and roi5_y1 < y_middle < roi5_y2)):
        return "E"
        
    elif ((roi6_x1 < x_middle < roi6_x2 and roi6_y1 < y_middle < roi6_y2)):
        return "F"
    else :
        return None


def draw_boxes(img, bbox, identities=None, categories=None, names=None, save_with_object_id=False, path=None, roi=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2

        # Check if the center of the box is within any of the ROIs
        in_roi = False
        if roi is not None:
            for roi_box in roi:
                roi_x1, roi_y1, roi_x2, roi_y2 = roi_box
                if roi_x1 <= x_center <= roi_x2 and roi_y1 <= y_center <= roi_y2:
                    in_roi = True
                    break

        # Only proceed to draw the box if it's within an ROI
        if in_roi:
            cat = int(categories[i]) if categories is not None else 0
            id = int(identities[i]) if identities is not None else 0
            label = f"{id}:{names[cat]}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 20), 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 144, 30), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, [255, 255, 255], 1)

            if save_with_object_id:
                txt_str = f"{id} {cat} {x1/img.shape[1]:.6f} {y1/img.shape[0]:.6f} {x2/img.shape[1]:.6f} {y2/img.shape[0]:.6f} {(x1 + (x2 - x1) / 2)/img.shape[1]:.6f} {(y1 + (y2 - y1) / 2)/img.shape[0]:.6f}\n"
                with open(path + '.txt', 'a') as f:
                    f.write(txt_str)
    return img

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace, colored_trk, save_bbox_dim, save_with_object_id= opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.colored_trk, opt.save_bbox_dim, opt.save_with_object_id
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    #.... Initialize SORT .... 
    #......................... 
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)
    #......................... 
    
    
    #........Rand Color for every trk.......
    rand_color_list = []
    amount_rand_color_prime = 5003 # prime number
    for i in range(0,amount_rand_color_prime):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    #......................................

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    now = datetime.now()

    prev_counter1 = 0
    prev_counter2 = 0
    prev_counter3 = 0
    prev_counter4 = 0
    prev_counter5 = 0
    prev_counter6 = 0

    dt_string1 = now.strftime("%H:%M:%S")
    dt_string2 = now.strftime("%H:%M:%S")
    dt_string3 = now.strftime("%H:%M:%S")
    dt_string4 = now.strftime("%H:%M:%S")
    dt_string5 = now.strftime("%H:%M:%S")
    dt_string6 = now.strftime("%H:%M:%S")


    current_time = now.strftime("%H:%M:%S")
    detected_objects = []

    



    for path, img, im0s, vid_cap in dataset:


        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            counter1, counter2, counter3, counter4, counter5, counter6 = 0, 0, 0, 0, 0, 0   
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh



            print(frame)
            roi1_x1, roi1_y1, roi1_x2, roi1_y2 = (int(0.124*im0.shape[1]),int(0.301*im0.shape[0]),int(0.310*im0.shape[1]),int(0.565*im0.shape[0]))
            roi2_x1, roi2_y1, roi2_x2, roi2_y2 = (int(0.395*im0.shape[1]),int(0.301*im0.shape[0]),int(0.620*im0.shape[1]),int(0.621*im0.shape[0]))
            roi3_x1, roi3_y1, roi3_x2, roi3_y2 = (int(0.620*im0.shape[1]),int(0.445*im0.shape[0]),int(0.806*im0.shape[1]),int(0.621*im0.shape[0]))
            roi4_x1, roi4_y1, roi4_x2, roi4_y2 = (int(0.077*im0.shape[1]),int(0.565*im0.shape[0]),int(0.404*im0.shape[1]),int(im0.shape[0]))
            roi5_x1, roi5_y1, roi5_x2, roi5_y2 = (int(0.521*im0.shape[1]),int(0.621*im0.shape[0]),int(0.758*im0.shape[1]),int(im0.shape[0]))
            roi6_x1, roi6_y1, roi6_x2, roi6_y2 = (int(0.758*im0.shape[1]),int(0.621*im0.shape[0]),int(im0.shape[1]),int(im0.shape[0]))

            roi = [(int(0.124*im0.shape[1]),int(0.301*im0.shape[0]),int(0.310*im0.shape[1]),int(0.565*im0.shape[0])),(int(0.395*im0.shape[1]),int(0.301*im0.shape[0]),int(0.620*im0.shape[1]),int(0.621*im0.shape[0])), (int(0.620*im0.shape[1]),int(0.445*im0.shape[0]),int(0.806*im0.shape[1]),int(0.621*im0.shape[0])), (int(0.077*im0.shape[1]),int(0.565*im0.shape[0]),int(0.404*im0.shape[1]),int(im0.shape[0])), (int(0.521*im0.shape[1]),int(0.621*im0.shape[0]),int(0.758*im0.shape[1]),int(im0.shape[0])), (int(0.758*im0.shape[1]),int(0.621*im0.shape[0]),int(im0.shape[1]),int(im0.shape[0]))]


            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))
                
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))
                
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks =sort_tracker.getTrackers()

                txt_str = ""

                #loop over tracks
                for track in tracks:
                    
                    if save_txt and not save_with_object_id:
                        # Normalize coordinates
                        txt_str += "%i %i %f %f" % (track.id, track.detclass, track.centroidarr[-1][0] / im0.shape[1], track.centroidarr[-1][1] / im0.shape[0])
                        if save_bbox_dim:
                            txt_str += " %f %f" % (np.abs(track.bbox_history[-1][0] - track.bbox_history[-1][2]) / im0.shape[0], np.abs(track.bbox_history[-1][1] - track.bbox_history[-1][3]) / im0.shape[1])
                        txt_str += "\n"
                
                if save_txt and not save_with_object_id:
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(txt_str)

                # draw boxes for visualization
                        
                if len(tracked_dets) > 0:
                    

                    for i, det in enumerate(pred):  # detections per image
                        if len(det):
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                            # Process each detection
                            for *xyxy, conf, cls in det:
                                x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                                obj_id = int(cls)  # example object ID, replace with actual ID if available
                                roi_position = determine_roi(x1, y1, x2, y2,im0)  # You need to implement this based on your ROIs
                                detected_objects.append({
                                    "id": obj_id,
                                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                    "roi_position": roi_position
                                })
                        else:
                            # No detections in this frame
                            pass

                    # Update global tracking information
                    update_objects(detected_objects, current_time)

                    # Draw bounding boxes based on tracked objects
                    for obj_id, obj_info in tracked_objects.items():
                        bbox = obj_info["last_position"]
                        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                        label = f"{obj_id}: {obj_info['roi_position']}"
                        cv2.rectangle(im0, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


                    # # Loop through each detection and draw the boxes
                    # for i, box in enumerate(bbox_xyxy):
                    #     x1, y1, x2, y2 = [int(i) for i in box]
                    #     y_middle = ((y2 - y1) / 2) + y1
                    #     x_middle = ((x2 - x1) / 2) + x1

                    #     # Check which ROI the detection falls into
                    #     if roi1_x1 < x_middle < roi1_x2 and roi1_y1 < y_middle < roi1_y2:
                    #         counter1 += 1
                            
                    #     elif roi2_x1 < x_middle < roi2_x2 and roi2_y1 < y_middle < roi2_y2:
                    #         counter2 += 1
                            
                    #     elif ((roi3_x1 < x_middle < roi3_x2 and roi3_y1 < y_middle < roi3_y2)):
                    #         counter3  += 1
                            
                    #     elif ((roi4_x1 < x_middle < roi4_x2 and roi4_y1 < y_middle < roi4_y2)):
                    #         counter4  += 1
                           
                    #     elif ((roi5_x1 < x_middle < roi5_x2 and roi5_y1 < y_middle < roi5_y2)):
                    #         counter5  += 1
                            
                    #     elif ((roi6_x1 < x_middle < roi6_x2 and roi6_y1 < y_middle < roi6_y2)):
                    #         counter6  += 1
                            

                    #     draw_boxes(im0, bbox_xyxy, identities, categories, names, save_with_object_id, txt_path, roi)
                        
            else: #SORT should be updated even with no detections
                tracked_dets = sort_tracker.update()
            #........................................................
                

            if prev_counter1 != counter1 :
                now = datetime.now()
                dt_string1 = now.strftime("%H:%M:%S")
                prev_counter1 = counter1

            if prev_counter2 != counter2 : 
                now = datetime.now()
                dt_string2 = now.strftime("%H:%M:%S")
                prev_counter2 = counter2

            if prev_counter3 != counter3 : 
                now = datetime.now()
                dt_string3 = now.strftime("%H:%M:%S")
                prev_counter3 = counter3

            if prev_counter4 != counter4 : 
                now = datetime.now()
                dt_string4 = now.strftime("%H:%M:%S")
                prev_counter4 = counter4

            if prev_counter5 != counter5 : 
                now = datetime.now()
                dt_string5 = now.strftime("%H:%M:%S")
                prev_counter5 = counter5

            if prev_counter6 != counter6 : 
                now = datetime.now()
                dt_string6 = now.strftime("%H:%M:%S")
                prev_counter6 = counter6

            print("Object")
            print(detected_objects)



            text1 = f"Head BBOX A : {counter1}"     
            level_of_service_1 = density_calc(counter1, 5)
            cv2.rectangle(im0, (roi1_x1,roi1_y1), (roi1_x2,roi1_y2), (0,255,0), 3)
            cv2.putText(im0, text1, (0,int(im0.shape[0]*0.04)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, "LoS :" + level_of_service_1, (500,int(im0.shape[0]*0.04)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, dt_string1, (300,int(im0.shape[0]*0.04)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)

            text2 = f"Head BBOX B : {counter2}"     
            level_of_service_2 = density_calc(counter2, 5)
            cv2.rectangle(im0, (roi2_x1,roi2_y1), (roi2_x2,roi2_y2), (0,255,0), 3)
            cv2.putText(im0, text2, (0,int(im0.shape[0]*0.04) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, "LoS :" + level_of_service_2, (500,int(im0.shape[0]*0.04) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, dt_string2, (300,int(im0.shape[0]*0.04) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)

            text3 = f"Head BBOX C : {counter3}"     
            level_of_service_3 = density_calc(counter3, 5)
            cv2.rectangle(im0, (roi3_x1,roi3_y1), (roi3_x2,roi3_y2), (0,255,0), 3)
            cv2.putText(im0, text3, (0,int(im0.shape[0]*0.04) +60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, "LoS :" + level_of_service_3, (500,int(im0.shape[0]*0.04)+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, dt_string3, (300,int(im0.shape[0]*0.04)+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)

            text4 = f"Head BBOX D : {counter4}"     
            level_of_service_4 = density_calc(counter4, 5)
            cv2.rectangle(im0, (roi4_x1,roi4_y1), (roi4_x2,roi4_y2), (0,255,0), 3)
            cv2.putText(im0, text4, (0,int(im0.shape[0]*0.04)+90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, "LoS :" + level_of_service_4, (500,int(im0.shape[0]*0.04)+90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, dt_string4, (300,int(im0.shape[0]*0.04)+90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)

            text5 = f"Head BBOX E : {counter5}"     
            level_of_service_5 = density_calc(counter5, 5)
            cv2.rectangle(im0, (roi5_x1,roi5_y1), (roi5_x2,roi5_y2), (0,255,0), 3)
            cv2.putText(im0, text5, (0,int(im0.shape[0]*0.04)+120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, "LoS :" + level_of_service_5, (500,int(im0.shape[0]*0.04)+120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, dt_string5, (300,int(im0.shape[0]*0.04)+120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)

            text6 = f"Head BBOX F : {counter6}"     
            level_of_service_6 = density_calc(counter6, 5)
            cv2.rectangle(im0, (roi6_x1,roi6_y1), (roi6_x2,roi6_y2), (0,255,0), 3)
            cv2.putText(im0, text6, (0,int(im0.shape[0]*0.04)+150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, "LoS :" + level_of_service_6, (500,int(im0.shape[0]*0.04)+150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, dt_string6, (300,int(im0.shape[0]*0.04)+150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--download', action='store_true', help='download model weights automatically')
    parser.add_argument('--no-download', dest='download', action='store_false',help='not download model weights if already exist')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='object_tracking', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--colored-trk', action='store_true', help='assign different color to every track')
    parser.add_argument('--save-bbox-dim', action='store_true', help='save bounding box dimensions with --save-txt tracks')
    parser.add_argument('--save-with-object-id', action='store_true', help='save results with object id to *.txt')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
