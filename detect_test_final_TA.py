import argparse
import time
import json
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


from sort import *

from datetime import datetime
import mysql.connector 


def determine_roi(x1, y1, x2, y2, im0) :
    y_middle = ((y2 - y1) / 2) + y1
    x_middle = ((x2 - x1) / 2) + x1

    # roi1_x1, roi1_y1, roi1_x2, roi1_y2 = int(0.215*im0.shape[1]),int(0.082*im0.shape[0]),int(0.729*im0.shape[1]),int(0.568*im0.shape[0])
    roi1_x1, roi1_y1, roi1_x2, roi1_y2 = int(0.2*im0.shape[1]),int(0.2*im0.shape[0]),int(0.8*im0.shape[1]),int(0.8*im0.shape[0])


    # Check which ROI the detection falls into
    if roi1_x1 < x_middle < roi1_x2 and roi1_y1 < y_middle < roi1_y2:
        return  "ROI_1" 
    # elif roi2_x1 < x_middle < roi2_x2 and roi2_y1 < y_middle < roi2_y2:
    #     return  "ROI_2"
    else :
        return None

def people_counting(img, bbox, counter, identities=None, categories=None, names=None, save_with_object_id=False, path=None, detected_object=None, frame_rate = None, current_frame = None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        # Check if the center of the box is within any of the ROIs
        roi_position = determine_roi(x1,y1,x2,y2,img)


        # Only proceed to draw the box if it's within an ROI
        if roi_position is not None:
            id = int(identities[i]) if identities is not None else 0
            id_updated = False
            current_video_time = current_frame / frame_rate if current_frame is not None and frame_rate is not None else 0

            # Check if the id is already in detected_object and update it
            for obj in detected_object:
                time_diff = 0
                if obj['id'] == id:
                    # Calculate the time difference in seconds based on video time
                    time_diff = current_video_time - obj['first_detected']
                    
                    if obj['roi_position'] == "ROI_1" :
                        counter += 1

        
                    break  # Exit the loop since we've updated the id

            if not id_updated:
            # If the id is not found, append the new object with the first_detected time
                time_diff = 0
                detected_object.append({
                    "id": id,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "roi_position": roi_position,
                    "first_detected": current_video_time,  # Store the initial detection time
                    "time_in_roi": time_diff  # Initialize the time in ROI as 0 since it's just detected
                })

            label = f"{id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            
            if time_diff >= 0  :
                    
                        box_color = (0, 255, 0)  # Color for the bounding box           
                        label_background_color = (0,0,255)
                
                        text_color = (0, 0, 0)  # Color for the text

                        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
                        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), label_background_color, -1)
                        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, text_color, 1)

    return img, counter


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
    half = False  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Set MySQL Database
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="kh670205"
    )

    mycursor = mydb.cursor()


    mycursor.execute("CREATE DATABASE IF NOT EXISTS testdb")
    mycursor.execute("USE testdb")

    mycursor.execute("""
        CREATE TABLE IF NOT EXISTS detectionData (
            detection_id INT,
            location_id INTEGER,
            timestamp VARCHAR(255),
            counter INTEGER,
            level_of_service VARCHAR(1),
            detection_duration INT
        )
    """)

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

    # Initialize Value
    detected_objects = []
    location_id = opt.room_ID
    prev_counter = 0 
    counter = 0
    det_id = 0
    los = "A"
    

    for path, img, im0s, vid_cap in dataset:
        duration = 0
        start_time = time.time()


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

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            

            # roi1_x1, roi1_y1, roi1_x2, roi1_y2 = int(0.215*im0.shape[1]),int(0.082*im0.shape[0]),int(0.729*im0.shape[1]),int( 0.568*im0.shape[0])
            roi1_x1, roi1_y1, roi1_x2, roi1_y2 = int(0.2*im0.shape[1]),int(0.2*im0.shape[0]),int(0.8*im0.shape[1]),int(0.8*im0.shape[0])


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
                        
                if len(tracked_dets) > 0:

                    bbox_xyxy = tracked_dets[:, :4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]

                    for i, box in enumerate(bbox_xyxy):
        
                        query = """
                            SELECT detection_id, counter 
                            FROM detectionData 
                            WHERE location_id = %s 
                            ORDER BY detection_id DESC 
                            LIMIT 1
                        """
                        # Execute the query
                        mycursor.execute(query, (location_id,))

                        # Fetch the result
                        result = mycursor.fetchone()

                        # Initialize the counters with the values from the database
                        if result:
                            det_id, counter = result
                            det_id += 1

                        img, counter = people_counting(im0, bbox_xyxy, 0, identities, categories, names, save_with_object_id, txt_path, detected_objects, 60, frame)
                        los = density_calc(counter, 4.95)
                        end_time = time.time()
                        duration = int((end_time - start_time) * 1000)
                        date = datetime.now()

                        if los == "F" :
                            date_json = date.strftime('%Y-%m-%d %H:%M:%S'),
                            dump = {
                                "detection_id" : det_id,
                                "location_id" : location_id,
                                "timestamp" : date_json,
                                "counter"   : counter,
                                "los"       : los,
                                "duration_ms" : duration
                            }
                            file_name = f"output_{det_id}.json"
                            file_path = 'outputs/' + file_name
                            
                            with open(file_path, 'w') as json_file:
                                json.dump(dump, json_file, indent = 4)


                        # Insert into Database
                        if prev_counter != counter :
            
                            sql = "INSERT INTO detectionData (detection_id, location_id, timestamp, counter, level_of_service, detection_duration) VALUES (%s, %s, %s, %s, %s, %s)"
                            
                            

                        
                            val = (det_id, location_id, date, counter, los, duration)
                            duration = 0

                            # Execute the query
                            mycursor.execute(sql, val)
                            mydb.commit()
                            prev_counter = counter
        


            else: #SORT should be updated even with no detections
                tracked_dets = sort_tracker.update()
            #........................................................
    
            text_size = 1
            offset_y = 37
            text_bold = 2
            
            cv2.rectangle(im0, (roi1_x1,roi1_y1), (roi1_x2,roi1_y2), (255,0,0), 3)

            text1 = f"Counter : {counter}"     
            text2 = f"Level of Service : {los}"
                    
            
            
            
            cv2.rectangle(im0, (roi1_x1,roi1_y1), (roi1_x2,roi1_y2), (0,255,0), 3)
            

            # cv2.rectangle(im0, (0,0), (int(im0.shape[0]*0.56),int(im0.shape[0]*0.05)), (255,255,255), -1)
            cv2.putText(im0, text1, (0,int(im0.shape[0]*0.06)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255) , text_bold, cv2.LINE_AA)    
            # cv2.putText(im0, text2, (0,int(im0.shape[0]*0.06+offset_y)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255) , text_bold, cv2.LINE_AA)    
            
            # cv2.putText(im0, "LoS : " + level_of_service, (0,int(im0.shape[0]*0.06)+offset_y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255) , text_bold, cv2.LINE_AA)    
            # cv2.putText(im0, dt_string1, (600,int(im0.shape[0]*0.04)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 255) , text_bold, cv2.LINE_AA)
        
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Break gracefully
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


            # Stream results
            # cv2.imshow(str(p), im0)
            #cv2.waitKey(1)  # 1 millisecond

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
        print(f"Results saved to {save_dir}{s}")
    

    print(f'Done. ({time.time() - t1:.3f}s)')


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

    parser.add_argument('--room-ID', type=int, default=1, help='detected room id')
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
       