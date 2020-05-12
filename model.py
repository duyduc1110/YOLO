import torch
import torch.nn as nn
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
                                  stride,
                                  padding,
                                  bias=False if 'batch_normalize' in module else True))
        if 'batch_normalize' in module:
            self.add_module('norm', nn.BatchNorm2d(self.out_f))
        if module['activation'] == 'leaky':
            self.add_module('leaky', nn.LeakyReLU(0.1, inplace=True))

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
                  .repeat(1, 1, 1, 3).to(DEVICE))
        grid_y = (torch.arange(self.input_w)
                  .repeat(self.input_h, 1)
                  .t()
                  .view(1, self.input_w, self.input_h, 1)
                  .repeat(1, 1, 1, 3).to(DEVICE))
        anchor_w = torch.tensor([w / self.stride for w, _ in self.anchors]).to(DEVICE)
        anchor_h = torch.tensor([h / self.stride for _, h in self.anchors]).to(DEVICE)

        # Calculate bx, by, bw, bh
        inputs[..., 0] += grid_x
        inputs[..., 1] += grid_y
        inputs[..., 2] = torch.exp(inputs[..., 2]) * anchor_w
        inputs[..., 3] = torch.exp(inputs[..., 3]) * anchor_h

        # Truth ground boxes
        inputs[..., :4] *= self.stride

        inputs = inputs.view(1, -1, 85)

        return inputs

    def forward(self, inputs=None, targets=None):
        self.input_w = inputs.shape[-2]
        self.input_h = inputs.shape[-1]
        inputs = inputs.view(1, 3, -1, self.input_w, self.input_h).permute(0, 3, 4, 1, 2).contiguous()
        self.stride = 416 / self.input_w

        # Sigmoid x, y, po and pc
        inputs[..., :2] = torch.sigmoid(inputs[..., :2])
        inputs[..., 4:] = torch.sigmoid(inputs[..., 4:])

        return self.get_boxes(inputs=inputs)


class YOLO(nn.Module):
    def __init__(self, modules):
        super(YOLO, self).__init__()
        self.layers = nn.ModuleList()
        self.create_modules(modules)
        self.modules = modules

    def create_modules(self, modules):
        out_fs = [3]
        for idx, module in enumerate(modules[1:]):
            if module['type'] == 'convolutional':
                in_f = out_fs[-1]
                t = ConvLayer(idx, in_f, module)
                f = t.out_f
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
                if 'batch_normalize' in self.modules[idx + 1]:
                    bn_layer = layer.norm
                    length = bn_layer.bias.numel()
                    for i in ['bias', 'weight', 'running_mean', 'running_var']:
                        x = getattr(bn_layer, i)
                        weight_to_load = torch.from_numpy(weights[idx_w: idx_w + length])
                        weight_to_load = weight_to_load.view_as(x.data)
                        if i in ['bias', 'weight']:
                            x.data.copy_(weight_to_load)
                        else:
                            x.copy_(weight_to_load)
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

    def forward(self, x):
        outputs = []
        yolo_outputs = []

        for idx, layer in enumerate(self.layers):
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

            # print(x.shape)
            outputs.append(x)

        yolo_outputs = torch.cat(yolo_outputs, 1)

        return yolo_outputs