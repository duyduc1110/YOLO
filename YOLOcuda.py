import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2


def load_config():
    modules = []
    with open('yolov3.cfg', 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        lines = [line.rstrip().lstrip() for line in lines if line != '' if line[0] != '#']

        module = {}
        for line in lines:
            if line[0] == '[':
                if len(module) > 0:
                    modules.append(module)
                module = {}
                module['type'] = line[1:-1]
            else:
                key, value = line.split('=')
                module[key.strip()] = value.strip()
        modules.append(module)
    return modules

class ConvLayer(nn.Module):
    def __init__(self, idx, in_f, module):
        super(ConvLayer, self).__init__()
        self.create_module(idx, in_f, module)

    def create_module(self, idx, in_f, module):
        self.out_f = int(module['filters'])
        kernel_size = int(module['size'])
        stride = int(module['stride'])
        padding = (kernel_size - 1) // 2
        self.add_module('conv',
                        nn.Conv2d(in_f,
                                  self.out_f,
                                  kernel_size,
                                  stride, padding,
                                  bias=False if 'batch_normalize' in module else True))
        if 'batch_normalize' in module:
            self.add_module('norm', nn.BatchNorm2d(self.out_f))
        if module['activation'] == 'leaky':
            self.add_module('leaky', nn.LeakyReLU(inplace=True))

    def get_out_features(self):
        return self.out_f

    def forward(self, x):
        for sub_module in self.children():
            x = sub_module(x)
        return x


class ShortcutLayer(nn.Module):
    def __init__(self, pos):
        super(ShortcutLayer, self).__init__()
        self.pos = pos

    def forward(self, x, y):
        return x + y


class RouteLayer(nn.Module):
    def __init__(self, pos):
        super(RouteLayer, self).__init__()
        self.pos = [int(i) for i in pos.split(',')]

    def forward(self, x):
        return torch.cat(x, 1)


class UpsampleLayer(nn.Module):
    def __init__(self, scale):
        super(UpsampleLayer, self).__init__()
        self.up = nn.Upsample(scale_factor=scale)

    def forward(self, x):
        return self.up(x)


class YOLOLayer(nn.Module):
    def __init__(self, anchors=None):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors

    def get_boxes(self, inputs=None):
        grid_x = (torch.arange(self.input_w)
                  .repeat(self.input_h, 1)
                  .view(1, self.input_w, self.input_h, 1)
                  .repeat(BATCH_SIZE, 1, 1, NUM_BOX)
                  .to('cuda'))
        grid_y = (torch.arange(self.input_w)
                  .repeat(self.input_h, 1)
                  .t()
                  .view(1, self.input_w, self.input_h, 1)
                  .repeat(BATCH_SIZE, 1, 1, NUM_BOX)
                  .to('cuda'))
        anchor_w = torch.tensor([w / self.stride for w, _ in self.anchors]).to('cuda')
        anchor_h = torch.tensor([h / self.stride for _, h in self.anchors]).to('cuda')

        # Calculate bx, by, bw, bh
        inputs[..., 0] += grid_x
        inputs[..., 1] += grid_y
        inputs[..., 2] = torch.exp(inputs[..., 2]) * anchor_w
        inputs[..., 3] = torch.exp(inputs[..., 3]) * anchor_h

        # Truth ground boxes
        inputs[..., :4] *= self.stride

        inputs = inputs.view(BATCH_SIZE, -1, 85)

        return inputs

    def forward(self, inputs=None, targets=None):
        self.input_w = inputs.shape[-2]
        self.input_h = inputs.shape[-1]
        inputs = inputs.view(BATCH_SIZE, NUM_BOX, -1, self.input_w, self.input_h).permute(0, 3, 4, 1, 2).contiguous()
        self.stride = WIDTH / self.input_w

        # Sigmoid x, y, po and pc
        inputs[..., :2] = torch.sigmoid(inputs[..., :2])
        inputs[..., 4:] = torch.sigmoid(inputs[..., 4:])

        return self.get_boxes(inputs=inputs)


class YOLO(nn.Module):
    def __init__(self, modules):
        super(YOLO, self).__init__()
        self.layers = nn.ModuleList()
        self.create_modules(modules)

    def create_modules(self, modules):
        out_fs = [3]
        for idx, module in enumerate(modules[1:]):
            if module['type'] == 'convolutional':
                in_f = out_fs[-1]
                t = ConvLayer(idx, in_f, module)
                f = t.get_out_features()
                self.layers.append(t)
            elif module['type'] == 'shortcut':
                pos = int(module['from'])
                t = ShortcutLayer(pos)
                f = out_fs[-1]
                self.layers.append(t)
            elif module['type'] == 'route':
                pos = module['layers']
                t = RouteLayer(pos)
                f = sum([out_fs[i] for i in t.pos])
                self.layers.append(t)
            elif module['type'] == 'upsample':
                scale = module['stride']
                t = UpsampleLayer(scale)
                f = out_fs[-1]
                self.layers.append(t)
            elif module['type'] == 'yolo':
                mask = [int(i) for i in module['mask'].split(',')]
                anchors = [int(value) for value in module['anchors'].split(',')]
                anchors = [(anchors[2 * i], anchors[2 * i + 1]) for i in mask]
                t = YOLOLayer(anchors=anchors)
                f = out_fs[-1]
                self.layers.append(t)
            out_fs.append(f)

    def load_weights(self, weight_path=None):
        with open(weight_path, 'rb') as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)

        idx_w = 0
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, ConvLayer):
                conv_layer = layer.conv

                # Load weights to batch norm or conv bias
                if 'batch_normalize' in modules[idx + 1]:
                    bn_layer = layer.norm
                    length = bn_layer.bias.numel()
                    for i in ['bias', 'weight', 'running_mean', 'running_var']:
                        x = getattr(bn_layer, i)
                        weight_to_load = torch.from_numpy(weights[idx_w: idx_w + length])
                        weight_to_load = weight_to_load.view_as(x.data)
                        x.data.copy_(weight_to_load)
                        idx_w += length
                else:
                    length = conv_layer.bias.numel()
                    weight_to_load = torch.from_numpy(weights[idx_w: idx_w + length])
                    weight_to_load = weight_to_load.view_as(layer.conv.bias.data)
                    conv_layer.bias.data.copy_(weight_to_load)
                    idx_w += length

                # Load to conv weight
                length = conv_layer.weight.numel()
                weight_to_load = torch.from_numpy(weights[idx_w: idx_w + length])
                weight_to_load = weight_to_load.view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(weight_to_load)
                idx_w += length

                print('Loaded to Conv #{}, weight index is {}'.format(idx, idx_w))

    def create_bouding_boxes(self, inputs=None):
        boxes = []
        for i in range(inputs.shape[1]):
            obj_score = inputs[0, i, 4]

            if obj_score <= THRESH:
                continue

    def forward(self, x):
        outputs = []
        yolo_outputs = []
        for idx, layer in enumerate(model.layers):
            if isinstance(layer, ConvLayer):
                x = layer(x)
            elif isinstance(layer, ShortcutLayer):
                x = layer(x, outputs[layer.pos])
            elif isinstance(layer, RouteLayer):
                temp = [outputs[i] for i in layer.pos]
                x = layer(temp)
            elif isinstance(layer, UpsampleLayer):
                x = layer(x)
            elif isinstance(layer, YOLOLayer):
                yolo_output = layer(inputs=x)
                yolo_outputs.append(yolo_output)
                x = outputs[-1]
            outputs.append(x)
        yolo_outputs = torch.cat(yolo_outputs, 1)

        self.create_bouding_boxes(inputs=yolo_outputs)

        return yolo_outputs


def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) / new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) / new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:, :, ::-1] / 255., (int(new_w), int(new_h)))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[int((net_h - new_h) // 2):int((net_h + new_h) // 2), int((net_w - new_w) // 2):int((net_w + new_w) // 2),
    :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image


class BoundingBox:
    def __init__(self, x_min, y_min, x_max, y_max, score, label, obj_score):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.box = [self.x_min, self.y_min, self.x_max, self.y_max]

        self.score = score
        self.label = label
        self.obj_score = obj_score

    def _print(self):
        print(self.x_min, self.y_min, self.x_max, self.y_max, self.score, self.label, self.obj_score)


def xywh2wywy(inputs=None):
    x, y, w, h = inputs
    x_min, x_max = x - w / 2, x + w / 2
    y_min, y_max = y - h / 2, y + h / 2

    return torch.tensor([x_min, y_min, x_max, y_max], dtype=inputs.dtype)


def iou(box1=None, box2=None):
    x = [box1.x_min, box1.x_max, box2.x_min, box2.x_max]
    y = [box1.y_min, box1.y_max, box2.y_min, box2.y_max]

    if x[1] < x[2] or x[3] < x[0] or y[1] < y[2] or y[3] < y[0]:
        return 0
    else:
        inter_x = min(x[1] - x[2], x[3] - x[0])
        inter_y = min(y[1] - y[2], y[3] - y[0])
        inter = inter_x * inter_y
        overlap = (x[1] - x[0]) * (y[1] - y[0]) + (x[3] - x[2]) * (y[3] - y[2]) - inter
        iou = inter / overlap

        return iou


def nms(boxes=None):
    for i in range(len(boxes) - 1):
        if boxes[i].score == 0:
            continue
        for j in range(i + 1, len(boxes)):
            if boxes[i].label != boxes[j].label:
                break
            if iou(boxes[i], boxes[j]) > NMS_THRESH:
                boxes[j].score = 0

    return boxes


def draw_boxes(img, boxes):
    scale_x = img.shape[1] / WIDTH
    scale_y = img.shape[0] / HEIGHT

    for b in boxes:
        x_min = int(b.x_min * scale_x)
        x_max = int(b.x_max * scale_x)
        y_min = int(b.y_min * scale_y)
        y_max = int(b.y_max * scale_y)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv2.putText(img,
                    LABELS[b.label] + ' ' + str(b.obj_score),
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1e-3 * img.shape[0],
                    (0, 255, 0),
                    thickness=1)
    return img


if __name__ == '__main__':

    THRESH = 0.7
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
    NMS_THRESH = 0.45

    modules = load_config()

    model = YOLO(modules)
    model.load_weights('./weights/yolov3.weights')
    model.cuda()

    img = cv2.imread('dog.jpg')
    img_resized = cv2.resize(img[:, :, ::-1], (WIDTH, HEIGHT))

    img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).contiguous().to('cuda')
    img_tensor /= 255

    result = model(img_tensor)
    result = result.view(-1, 85)

    result = result.detach().cpu()

    boxes = []

    for bbox in result:
        if bbox[4] > THRESH:
            # Generate bouding box information
            x_min, y_min, x_max, y_max = xywh2wywy(bbox[:4])
            score = bbox[4]
            label = torch.argmax(bbox[5:])
            obj_score = bbox[label + 5]

            box = BoundingBox(x_min.tolist()
                              , y_min.tolist()
                              , x_max.tolist()
                              , y_max.tolist()
                              , score.tolist()
                              , label.tolist()
                              , obj_score.tolist())

            boxes.append(box)

    # sort boxes base on label class
    boxes.sort(key=lambda x: (x.label, -x.score))

    # nms boxes with NMS_THRESH value and filter boxes
    boxes = nms(boxes)
    boxes = [box for box in boxes if box.score > 0]

    img = draw_boxes(img, boxes)

    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
