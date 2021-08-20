# YOLOv5 ğŸš€ by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/bus.jpg --weights yolov5s.pt --img 640
    python onnx_modified.py --source data/images/bus.jpg --weights yolov5s.onnx --img 640
"""

import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import onnxruntime

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords, set_logging
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights,  # model.pt path(s)
        source,  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        name='exp',  # save results to project/name
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        ):

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))


    # Initialize
    set_logging()
    device = select_device(device)

    # Load model
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    check_requirements(('onnx', 'onnxruntime'))    
    session = onnxruntime.InferenceSession(weights, None)

    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        pt = False
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        pt = False
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    for path, img, im0s, vid_cap in dataset:
        img = img.astype('float32')
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        
        t1 = time_sync()
        # Inference
        print('runnning onnx prediction')
        pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()
        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    im0 = plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_width=line_thickness)

                cv2.imshow('image', im0)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # Print time (inference + NMS)
            print(f'{s} Done. ({t2 - t1:.3f}s)')




if __name__ == "__main__":
    check_requirements(exclude=('tensorboard', 'thop'))
    run(weights='yolov5s.onnx', source='data/images')
