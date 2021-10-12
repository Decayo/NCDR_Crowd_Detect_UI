import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadScreen_Capture
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, \
    check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from global_var import *
from ui_main import *
from mss import mss
import os
import threading
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
main_ui = None
detecting = False
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def detect(opt,area,stop_event):
    out, source, weights, show_vid, save_vid, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.show_vid, opt.save_vid, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    
    global detecting
    sx=int(srs.box_area[0])
    sy=int(srs.box_area[1])
    ex=int(srs.box_area[2])
    ey=int(srs.box_area[3])
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # if webcam:
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    # else:
    #     #dataset = LoadScreen_Capture()
    #     dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'
    #while true changed
    sct = mss()
    frame_idx = 0
    monitor_number = opt.monitor_num
    mon = sct.monitors[monitor_number]

    monitor_info = {'top' : sy, 'left' : sx, 'width' : ex-sx, 'height' : ey-sy,
    "mon": monitor_number
    }
    print(sy,sx)
    print("event?")
    #global src_state
    #get the fucking screen record
    #for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
    while not stop_event.wait(0.001):   
        
        if(detecting==True):
            break
        frame_idx = frame_idx + 1
        
        #sct_img = ImageGrab.grab(area)
        sct_img = sct.grab(monitor_info)
        sct_img = numpy.array(sct_img)
        sct_img = cv2.cvtColor(sct_img, cv2.COLOR_RGBA2RGB)
        dataset = LoadScreen_Capture(sct_img,img_size=imgsz)
        img, im0 , vid_cap = dataset.__next__()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        #print(img.shape)
        #print(img.mode)
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if(src_state==2):
                break
            #p, s, im0 = path, '', im0s
            s = ''
            #im0= im0s
            s += '%gx%g ' % img.shape[2:]  # print string
            #save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywh_bboxs = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    # to deep sort format
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)

                # pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)
                    # to MOT format
                    tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)

                    # Write MOT compliant results to file
                    if save_txt:
                        for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
                            bbox_top = tlwh_bbox[0]
                            bbox_left = tlwh_bbox[1]
                            bbox_w = tlwh_bbox[2]
                            bbox_h = tlwh_bbox[3]
                            identity = output[-1]
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_top,
                                                            bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            #print('%sDone. (%.3fs)' % (s, t2 - t1))
            print('%sDone. (%.3fs) ,%d' % (s, t2 - t1,src_state))
            # Stream results
            if show_vid:
                cv2.imshow("show", im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                     raise StopIteration

            # Save results (image with detections)
            if save_vid:
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

    # if save_txt or save_vid:
    #     print('Results saved to %s' % os.getcwd() + os.sep + out)
    #     if platform == 'darwin':  # MacOS
    #         os.system('open ' + save_path)

    # print('Done. (%.3fs)' % (time.time() - t0))

from tkinter_test import *
from win32 import win32api, win32gui, win32print
from win32.lib import win32con
import win32gui, win32ui, win32api, win32con
from win32.win32api import GetSystemMetrics
import threading
def draw_area_box(sx,sy,ex,ey):
    dc = win32gui.GetDC(0)
    dcObj = win32ui.CreateDCFromHandle(dc)
    hwnd = win32gui.WindowFromPoint((0,0))
    _sx=int(sx)
    _sy=int(sy)
    _ex=int(ex)
    _ey=int(ey)
    box_cor = (_sx,_sy,_ex,_ey)
    print(box_cor)
    red = win32api.RGB(255, 0, 0)
    while(src_state == 1 ):
        rect = win32gui.CreateRoundRectRgn(*box_cor, 4 , 4)
        
        win32gui.RedrawWindow(hwnd, box_cor, rect, win32con.RDW_INVALIDATE)
        time.sleep(2)
        for x in range(25):
            win32gui.SetPixel(dc, _sx+x, _sy, red)
            win32gui.SetPixel(dc, _sx+x, _ey, red)
            win32gui.SetPixel(dc, _ex-x, _sy, red)
            win32gui.SetPixel(dc, _ex-x, _ey, red)
            for y in range(25):
                win32gui.SetPixel(dc, _sx, _sy+y, red)
                win32gui.SetPixel(dc, _ex, _sy+y, red)
                win32gui.SetPixel(dc, _sx, _ey-y, red)
                win32gui.SetPixel(dc, _ex, _ey-y, red)

def start_drawing(pill2kill):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5/weights/crowdhuman_yolov5m.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', default=[0], type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument("--monitor-num", type=int, default=1)
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    global srs
    srs = ScreenShot()
    sx=int(srs.box_area[0])
    sy=int(srs.box_area[1])
    ex=int(srs.box_area[2])
    ey=int(srs.box_area[3])

    draw_area_thread = threading.Thread(target = draw_area_box,args=(sx,sy,ex,ey))
    draw_area_thread.start()
    #ui.pushButton_5.clicked.connect(lambda: break_detection_loop())
    with torch.no_grad():
        detect(args,srs.box_area,stop_event = pill2kill)
def break_detection_loop():
    print("thread stopping...")
    global detecting
    detecting = True
if __name__ == '__main__':

    
    
    start_drawing()
    
