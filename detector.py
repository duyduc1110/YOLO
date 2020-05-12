import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import pandas as pd
import cv2
from model import YOLO

image = False
video = True
PATH = './videos/soma.mp4'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESH = 0.5
NMS_THRESH = 0.45
BATCH_SIZE = 1
WIDTH = 416
HEIGHT = 416
NUM_BOX = 3
LABELS = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
COLORS = np.random.uniform(0, 255, size=(80, 3))


def load_config():
    modules = []
    with open('./cfg/yolov3.cfg', 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        lines = [line.rstrip().lstrip() for line in lines if line != '' if line[0] != '#']

        module = {}
        for line in lines:
            if line[0]=='[':
                if len(module) > 0:
                    modules.append(module)
                module = {}
                module['type'] = line[1:-1]
            else:
                key, value = line.split('=')
                module[key.strip()] = value.strip()
        modules.append(module)
    return modules


def load_model(modules):
    model = YOLO(modules)
    model.to(DEVICE)
    model.load_weights('./weights/yolov3.weights')
    return model


def letterbox_image(img, inp_dim):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    x = np.full((inp_dim[1], inp_dim[0], 3), 128)
    x[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    x = x[:, :, ::-1].transpose((2, 0, 1)).copy()
    x = torch.from_numpy(x).float().div(255.0).unsqueeze(0).to(DEVICE)

    return x


def xywh2wywy(inputs=None):
    x, y, w, h = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    x_min, x_max = x - w / 2, x + w / 2
    y_min, y_max = y - h / 2, y + h / 2

    return torch.cat((x_min.unsqueeze(1),
                      y_min.unsqueeze(1),
                      x_max.unsqueeze(1),
                      y_max.unsqueeze(1)
                      ), 1)


def iou(b1=None, b2=None):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1
    b2_x1, b2_y1, b2_x2, b2_y2 = b2
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    xx1 = max(b1_x1, b2_x1)
    yy1 = max(b1_y1, b2_y1)
    xx2 = min(b1_x2, b2_x2)
    yy2 = min(b1_y2, b2_y2)

    inter_area = (max(xx2 - xx1 + 1, 0)) * (max(yy2 - yy1 + 1, 0))
    union_area = b1_area + b2_area - inter_area

    return inter_area / union_area


def nms(inputs=None, NMS_TRESH=None):
    for i in range(inputs.shape[0] - 1):
        b1 = inputs[i]
        if b1[4] == 0:
            continue
        for j in range(i + 1, inputs.shape[0]):
            b2 = inputs[j]
            if b1[5] == b2[5] and b2[4] > 0:  # If same label
                iou_score = iou(b1[:4], b2[:4])
                if iou_score > NMS_TRESH:
                    inputs[j, 4] = 0
                # print('Box {}, {}\n{}\n{}\n{}'.format(i, j, b1[:4], b2[:4], iou(b1[:4], b2[:4])))

    return inputs[inputs[:, 4] > 0]


def draw_boxes(img, inputs):
    scale = min(HEIGHT / img.shape[0], WIDTH / img.shape[1])
    offset_x = (WIDTH - img.shape[1] * scale) / 2
    offset_y = (HEIGHT - img.shape[0] * scale) / 2

    for b in inputs:
        x_min = int((b[0] - offset_x) / scale)
        x_max = int((b[2] - offset_x) / scale)
        y_min = int((b[1] - offset_y) / scale)
        y_max = int((b[3] - offset_y) / scale)

        label = int(b[5])
        color = COLORS[label]
        text = str(LABELS[label])
        text_w, text_h = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 1)
        cv2.rectangle(img,
                      (x_min, y_min - text_h),
                      (x_min + text_w, y_min),
                      color,
                      -1)
        cv2.putText(img,
                    text,
                    (x_min, y_min),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 255, 255),
                    1)
    cv2.imshow("test", img)


def process_output(result=None, img=None):
    # Select label & score at bouding boxes
    max_pos = torch.argmax(result[:, 5:], 1)  # labels with highest score
    max_score = result[torch.arange(result.size(0)), max_pos + 5]  # score of that labels
    result = torch.cat((result[:, :5], max_pos.float().unsqueeze(1), max_score.unsqueeze(1)), 1)  # downside result

    # Turn xywh to xyxy and round it + Sort by po
    result[:, :4] = xywh2wywy(result[:, :4]).round()
    result = result[torch.argsort(result[:, 4], descending=True)]  # Sort by bouding box score

    # Do non max suppession
    result = nms(inputs=result, NMS_TRESH=NMS_THRESH)

    # Draw boxes
    draw_boxes(img, result)


def main(image, video, PATH):
    modules = load_config()
    model = load_model(modules)
    if image:
        img = cv2.imread(PATH)
        x = letterbox_image(img, (WIDTH, HEIGHT))
        with torch.no_grad():
            result = model(x)
            result = result[result[..., 4] >= THRESH]
            result = result.detach().cpu()
        if result.shape[0] == 0:
            draw_boxes(img, [])
        else:
            process_output(result, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif video:
        cap = cv2.VideoCapture(PATH)
        while True:
            _, frame = cap.read()
            frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
            x = letterbox_image(frame, (WIDTH, HEIGHT))
            with torch.no_grad():
                result = model(x)
                result = result[result[..., 4] >= THRESH]
                result = result.detach().cpu()
            if result.shape[0] == 0:
                draw_boxes(frame, [])
            else:
                process_output(result, frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        cap.release()


if __name__ == '__main__':
    main(image, video, PATH)