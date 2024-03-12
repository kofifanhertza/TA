import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

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


def draw_boxes(img, bbox, x1, y1, x2, y2, identities=None, categories=None, names=None, save_with_object_id=False, path=None):
    for i, box in enumerate(bbox):
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id) + ":"+ names[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,20), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, [255, 255, 255], 1)
        # cv2.circle(img, data, 6, color,-1)   #centroid of box
        txt_str = ""
        if save_with_object_id:
            txt_str += "%i %i %f %f %f %f %f %f" % (
                id, cat, int(box[0])/img.shape[1], int(box[1])/img.shape[0] , int(box[2])/img.shape[1], int(box[3])/img.shape[0] ,int(box[0] + (box[2] * 0.5))/img.shape[1] ,
                int(box[1] + (
                    box[3]* 0.5))/img.shape[0])
            txt_str += "\n"
            with open(path + '.txt', 'a') as f:
                f.write(txt_str)
    return img

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
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


            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'

                        y_middle= (int(xyxy[3])-int(xyxy[1]))/2 + int(xyxy[1])
                        x_middle= (int(xyxy[2])-int(xyxy[0]))/2 + int(xyxy[0])

                        bbox_color = (255,0,0)
                        if ((roi1_x1 < x_middle < roi1_x2 and roi1_y1 < y_middle < roi1_y2)):
                            counter1  += 1
                            plot_one_box(xyxy, im0, label=label, color=bbox_color, line_thickness=2 )

                        elif ((roi2_x1 < x_middle < roi2_x2 and roi2_y1 < y_middle < roi2_y2)):
                            counter2  += 1
                            plot_one_box(xyxy, im0, label=label, color=bbox_color, line_thickness=2 )

                        elif ((roi3_x1 < x_middle < roi3_x2 and roi3_y1 < y_middle < roi3_y2)):
                            counter3  += 1
                            plot_one_box(xyxy, im0, label=label, color=bbox_color, line_thickness=2 )

                        elif ((roi4_x1 < x_middle < roi4_x2 and roi4_y1 < y_middle < roi4_y2)):
                            counter4  += 1
                            plot_one_box(xyxy, im0, label=label, color=bbox_color, line_thickness=2 )

                        elif ((roi5_x1 < x_middle < roi5_x2 and roi5_y1 < y_middle < roi5_y2)):
                            counter5  += 1
                            plot_one_box(xyxy, im0, label=label, color=bbox_color, line_thickness=2 )

                        elif ((roi6_x1 < x_middle < roi6_x2 and roi6_y1 < y_middle < roi6_y2)):
                            counter6  += 1
                            plot_one_box(xyxy, im0, label=label, color=bbox_color, line_thickness=2 )

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


            print(im0.shape)



            text1 = f"Head BBOX A : {counter1}"     
            level_of_service_1 = density_calc(counter1, 5)
            cv2.rectangle(im0, (roi1_x1,roi1_y1), (roi1_x2,roi1_y2), (0,255,0), 3)
            cv2.putText(im0, text1, (0,int(im0.shape[0]*0.04)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, "LoS :" + level_of_service_1, (400,int(im0.shape[0]*0.04)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, dt_string1, (200,int(im0.shape[0]*0.04)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)

            text2 = f"Head BBOX B : {counter2}"     
            level_of_service_2 = density_calc(counter2, 5)
            cv2.rectangle(im0, (roi2_x1,roi2_y1), (roi2_x2,roi2_y2), (0,255,0), 3)
            cv2.putText(im0, text2, (0,int(im0.shape[0]*0.04) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, "LoS :" + level_of_service_2, (400,int(im0.shape[0]*0.04) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, dt_string2, (200,int(im0.shape[0]*0.04) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)

            text3 = f"Head BBOX C : {counter3}"     
            level_of_service_3 = density_calc(counter3, 5)
            cv2.rectangle(im0, (roi3_x1,roi3_y1), (roi3_x2,roi3_y2), (0,255,0), 3)
            cv2.putText(im0, text3, (0,int(im0.shape[0]*0.04) +60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, "LoS :" + level_of_service_3, (400,int(im0.shape[0]*0.04)+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, dt_string3, (200,int(im0.shape[0]*0.04)+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)

            text4 = f"Head BBOX D : {counter4}"     
            level_of_service_4 = density_calc(counter4, 5)
            cv2.rectangle(im0, (roi4_x1,roi4_y1), (roi4_x2,roi4_y2), (0,255,0), 3)
            cv2.putText(im0, text4, (0,int(im0.shape[0]*0.04)+90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, "LoS :" + level_of_service_4, (400,int(im0.shape[0]*0.04)+90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, dt_string4, (200,int(im0.shape[0]*0.04)+90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)

            text5 = f"Head BBOX E : {counter5}"     
            level_of_service_5 = density_calc(counter5, 5)
            cv2.rectangle(im0, (roi5_x1,roi5_y1), (roi5_x2,roi5_y2), (0,255,0), 3)
            cv2.putText(im0, text5, (0,int(im0.shape[0]*0.04)+120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, "LoS :" + level_of_service_5, (400,int(im0.shape[0]*0.04)+120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, dt_string5, (200,int(im0.shape[0]*0.04)+120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)

            text6 = f"Head BBOX F : {counter6}"     
            level_of_service_6 = density_calc(counter6, 5)
            cv2.rectangle(im0, (roi6_x1,roi6_y1), (roi6_x2,roi6_y2), (0,255,0), 3)
            cv2.putText(im0, text6, (0,int(im0.shape[0]*0.04)+150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, "LoS :" + level_of_service_6, (400,int(im0.shape[0]*0.04)+150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            cv2.putText(im0, dt_string6, (200,int(im0.shape[0]*0.04)+150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)    
            
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
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
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
