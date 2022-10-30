import os
import json
import torch
import traceback
from torch import nn
import numpy as np
import torch.nn.functional as F
from .utils import Json, TransLog, LayerOut_id, REGISTERED_LIST, get_weight
from . import config as cfg



# global params
log = TransLog()
INLINE = False
PLULINE = False
DEBUG = cfg.DEBUG
PARAM_FLAG = True
JSON_PARAM = {}
MODULE_DICT = {}

# Tensor operator
raw__add__ = torch.Tensor.__add__
raw__sub__ = torch.Tensor.__sub__
raw__permute__ = torch.Tensor.permute
raw__expand_as__ = torch.Tensor.expand_as


PLUGINS_LIST = [
    "BasicBlock",
    "Hsigmoid",
    "Hswish",
]


def get_parameters():
    global PARAM_FLAG
    global JSON_PARAM

    if PARAM_FLAG:
        js_init = Json(os.path.join(cfg.JSON_FILE_DIR, cfg.MODELNAME, cfg.MODELNAME + ".json"))
        js_param = js_init.get_json_param()
        JSON_PARAM = js_param
        PARAM_FLAG = False
        return js_param

    js_param = JSON_PARAM

    return js_param


#  nn.Conv2d ---> F.conv2d
def _conv2d(raw, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    global INLINE
    x = raw(input, weight, bias, stride, padding, dilation, groups)
    INLINE = True
    name = log.add_layer(name="conv2d_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # extract weight to get wgt
    get_weight(weight, f"{name}.weight")
    weightKey = f"{name}"
    biasKey = f"{name}"

    # add json params
    if bias is not None:
        get_weight(bias, f"{name}.bias")
        biasFile = f"{name}"

    conv_params = dict(
        {
            "layerStyle": "conv",
            "layerName": name,
            "inputName": log.blobs(input, name),
            "weightKey": weightKey,
            "biasKey": biasKey,
            "parameter": {
                "input_c": input.shape[1],
                "output_c": x.shape[1],
                "kernel": [weight.shape[2], weight.shape[3]],
                "padding": padding,
                "stride": stride,
                "dilation": dilation,
                "groups": groups,
            },
        }
    )

    if DEBUG:
        print(conv_params)
    js_param = get_parameters()
    js_param["network"].append(conv_params)
    INLINE = False
    return x


# nn.ReLU ----> F.relu
def _relu(raw, input, inplace=False):
    global INLINE
    name = log.add_layer(name="relu_")
    inputName_ = log.blobs(input, name)  # 这样防止 x == input时，它们id一致
    x = raw(input, inplace)
    INLINE = True
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # no weight to extract
    # add json params
    relu_params = dict(
        {
            "layerStyle": "active",
            "layerName": name,
            "inputName": inputName_,
            "activeType": "relu",
        }
    )
    if DEBUG:
        print(relu_params)
    js_param = get_parameters()
    js_param["network"].append(relu_params)
    INLINE = False
    return x


# nn.leakyReLU ---> F.leakyReLU
def _leaky_relu(raw, input, negative_slope=0.01, inplace=False):
    global INLINE
    x = raw(input, negative_slope, inplace)
    INLINE = True
    name = log.add_layer(name="leaky_relu_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # no weight to extract
    # add json params
    leaky_relu_params = dict(
        {"layerStyle": "active", "layerName": name, "inputName": log.blobs(input, name), "active_type": "l_relu"}
    )

    if DEBUG:
        print(leaky_relu_params)
    js_param = get_parameters()
    js_param["network"].append(leaky_relu_params)
    INLINE = False
    return x


# nn,MaxPool2d ---> F.max_pool2d
def _max_pool2d(raw, *args, **kwargs):
    global INLINE
    # args = (input, kernel, stride, padding, dilation, ceil_mode, return_indices)
    x = raw(*args, **kwargs)
    INLINE = True
    name = log.add_layer(name="max_pool2d_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # no weight to extract
    # add json params
    max_pool2d_params = dict(
        {
            "layerStyle": "pool",
            "layerName": name,
            "inputName": log.blobs(args[0], name),
            "parameter": {
                "poolType": "kMAX",
                # "kernel": [kernel_size, kernel_size] if isinstance(kernel_size, int) else kernel_size,
                # "stride": [stride, stride] if isinstance(stride, int) else stride,
                # "padding": [padding, padding] if isinstance(padding, int) else padding,
                "kernel": [args[1], args[1]] if isinstance(args[1], int) else args[1],
                "stride": [args[2], args[2]] if isinstance(args[2], int) else args[2],
                "padding": [args[3], args[3]] if isinstance(args[3], int) else args[3],
            },
        }
    )
    if DEBUG:
        print(max_pool2d_params)
    js_param = get_parameters()
    js_param["network"].append(max_pool2d_params)
    INLINE = False
    return x


# nn.AvgPool2d ----> F.avg_pool2d
def _avg_pool2d(
    raw,
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    global INLINE
    x = raw(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
    INLINE = True
    name = log.add_layer(name="avg_pool2d_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # no weight to extract
    # add json params
    avg_pool2d_params = dict(
        {
            "layerStyle": "pool",
            "layerName": name,
            "inputName": log.blobs(input, name),
            "parameter": {
                "poolType": "kAVERAGE",
                "kernel": [kernel_size, kernel_size] if isinstance(kernel_size, int) else kernel_size,
                "stride": [stride, stride] if isinstance(stride, int) else stride,
                "padding": [padding, padding] if isinstance(padding, int) else padding,
            },
        }
    )
    if DEBUG:
        print(avg_pool2d_params)

    js_param = get_parameters()
    js_param["network"].append(avg_pool2d_params)
    INLINE = False
    return x


# nn.Linear ---> F.linear
def _linear(raw, input, weight, bias=None):
    global INLINE
    x = raw(input, weight, bias)
    INLINE = True
    name = log.add_layer(name="linear_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # extract weight
    get_weight(weight, f"{name}.weight")
    weightKey = f"{name}"
    biasKey = f"{name}"
    if bias is not None:
        get_weight(bias, f"{name}.bias")
        biasKey = f"{name}"
    # add json param
    linear_params = dict(
        {
            "layerStyle": "fc",
            "layerName": name,
            "inputName": log.blobs(input, name),
            "weightKey": weightKey,
            "parameter": {"input_c": input.shape[1], "output_c": x.shape[1]},
        }
    )
    if bias is not None:
        linear_params["biasKey"] = biasKey
    if DEBUG:
        print(linear_params)
    js_param = get_parameters()
    js_param["network"].append(linear_params)
    INLINE = False
    return x


# nn.AdaptiveAvgPool2d ---> F.adaptive_avg_pool2d
# tensorrt not support, just pytorch test
def _adaptive_avg_pool2d(raw, input, output_size):
    global INLINE
    x = raw(input, output_size)
    INLINE = True
    name = log.add_layer(name="adaptive_avg_pool2d_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    if isinstance(output_size, int):
        out_size_0 = output_size
        out_size_1 = output_size
    else:
        out_size_0 = output_size[0]
        out_size_1 = output_size[1]

    input_sz = np.array(input.shape[2:])  # input_size [H * W]
    output_sz = np.array([out_size_0, out_size_1])

    stride_sz = np.floor(input_sz / output_sz)
    kernel_sz = input_sz - (output_sz - 1) * stride_sz

    # no weight extract
    # add json params
    adaptive_avg_pool2d_params = dict(
        {
            "layerStyle": "pool",
            "layerName": name,
            "inputName": log.blobs(input, name),
            "parameter": {
                "poolType": "kAVG",
                "kernel": [int(kernel_sz[0]), int(kernel_sz[1])],
                "stride": [int(stride_sz[0]), int(stride_sz[1])],
                "padding": [0, 0],
            },
        }
    )
    if DEBUG:
        print(adaptive_avg_pool2d_params)
    js_param = get_parameters()
    js_param["network"].append(adaptive_avg_pool2d_params)
    INLINE = False
    return x


# nn.Softmax ---> F.softmax
def _softmax(raw, input, dim=None, _stacklevel=3, dtype=None):
    global INLINE
    x = raw(input, dim, _stacklevel, dtype)
    INLINE = True
    name = log.add_layer(name="softmax_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # no weight to extract
    # add json params
    softmax_params = dict(
        {
            "layerStyle": "softmax",
            "layerName": name,
            "inputName": log.blobs(input, name),
        }
    )
    if DEBUG:
        print(softmax_params)
    js_param = get_parameters()
    js_param["network"].append(softmax_params)
    INLINE = False
    return x


# ConvTranspose2d ---> F.conv_transpose2d
def _conv_transpose2d(raw, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    global INLINE
    x = raw(input, weight, bias, stride, padding, output_padding, groups, dilation)
    INLINE = True
    name = log.add_layer(name="Deconv2d_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # extract weight
    get_weight(weight, f"{name}.weight")
    weightKey = f"{name}"
    if bias is not None:
        get_weight(bias, f"{name}.bias")
        biasFile = f"{name}.bias"

    # add json params
    conv_transpose2d_params = dict(
        {
            "layerStyle": "deconv",
            "layerName": name,
            "inputName": log.blobs(input, name),
            "weightKey": weightKey,
            "parameter": {
                "input_c": input.shape[1],
                "output_c": x.shape[1],
                "kernel": [weight.shape[2], weight.shape[3]],
                "padding": padding,
                "stride": stride,
            },
        }
    )
    if bias is not None:
        conv_transpose2d_params["biasFile"] = biasFile
    if DEBUG:
        print(conv_transpose2d_params)

    js_param = get_parameters()
    js_param["network"].append(conv_transpose2d_params)
    return x


# ['ConstantPad1d', 'ConstantPad2d', 'ConstantPad3d']   ---> F.pad
def _pad(raw, input, pad, mode="constant", value=0):
    global INLINE
    x = raw(input, pad, mode, value)
    INLINE = True
    name = log.add_layer(name="pad_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # not weight extract
    # add json params
    pad_params = dict(
        {
            "layerStyle": "padding",
            "layerName": name,
            "inputName": log.blobs(input, name),
            "parameter": {
                "input_c": input.shape[1],
                "prePadding": [0, 0],
                "postPadding": [1, 1],
            },
        }
    )
    if DEBUG:
        print(pad_params)
    js_param = get_parameters()
    js_param["network"].append(pad_params)
    INLINE = False
    return x


# F.interpolate
def _interpolate(
    raw, input, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None
):
    global INLINE
    x = raw(input, size, scale_factor, mode, align_corners, recompute_scale_factor)
    INLINE = True
    name = log.add_layer(name="interpolate_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # no weight extract
    # add json param
    resizeMode = {"nearest": 0, "bilinear": 1}
    interpolate_params = dict(
        {
            "layerStyle": "resize",
            "layerName": name,
            "inputName": log.blobs(input, name),
            "resizeMode": resizeMode[mode],
            "alignCorners": align_corners,
            "resizeDim": size,
        }
    )
    if DEBUG:
        print(interpolate_params)
    js_param = get_parameters()
    js_param["network"].append(interpolate_params)
    INLINE = False
    return x


# nn.BathcNorm --> F.batch_norm
def _batch_norm(
    raw, input, weight, bias, running_mean, running_var, training, momentum, eps, torch_backends_cudnn_enabled
):
    global INLINE
    x = raw(input, weight, bias, running_mean, running_var, training, momentum, eps, torch_backends_cudnn_enabled)
    INLINE = True
    name = log.add_layer(name="BN_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # extract weight to get wgt
    get_weight(weight, f"{name}.weight")
    get_weight(bias, f"{name}.bias")
    get_weight(running_mean, f"{name}.running_mean")
    get_weight(running_var, f"{name}.running_var")

    # add json params
    weightKey = f"{name}"
    bn_params = dict(
        {
            "layerStyle": "bn",
            "layerName": name,
            "inputName": log.blobs(input, name),
            "weightKey": weightKey,
        }
    )
    if DEBUG:
        print(bn_params)
    js_param = get_parameters()
    js_param["network"].append(bn_params)
    INLINE = False
    return x


# nn.Sigmoid ---> torch.sigmoid
def _sigmoid(raw, input):
    global INLINE
    x = raw(input)
    INLINE = True
    name = log.add_layer(name="sigmoid_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # no weiht to extract
    # add json params
    sigmoid_params = dict(
        {
            "layerstyle": "active",
            "layerName": name,
            "inputName": log.blobs(input, name),
            "active_type": "sigmoid",
        }
    )
    if DEBUG:
        print(sigmoid_params)
    js_param = get_parameters()
    js_param["network"].append(sigmoid_params)
    INLINE = False
    return x


# torch.flatten
def _flatten(raw, input, start_dim=1, end_dim=-1):
    global INLINE
    x = raw(input, start_dim, end_dim)
    INLINE = True
    name = LayerOut_id[int(id(input))]
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    INLINE = False
    return x


# torch.cat
def _cat(raw, inputs, dim=0):
    global INLINE

    x = raw(inputs, dim)
    INLINE = True
    inputName = []
    for input in inputs:
        inputName.append(log.blobs(input, name="cat"))

    name = log.add_layer(name="cat_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # no weight extract
    # add json params
    cat_params = dict(
        {
            "layerStyle": "concat",
            "layerName": name,
            "inputName": inputName,
            "axis": dim,
        }
    )
    if DEBUG:
        print(cat_params)

    js_param = get_parameters()
    js_param["network"].append(cat_params)
    INLINE = False
    return x


# F.instance_norm
def _instance_norm(
    raw,
    input,
    weight,
    bias,
    running_mean,
    running_var,
    use_input_stats,
    momentum,
    eps,
    torch_backends_cudnn_enabled,
):
    global INLINE
    x = raw(
        input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, torch_backends_cudnn_enabled
    )
    INLINE = True
    name = log.add_layer(name="IN_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # extract weight
    get_weight(weight, f"{name}.weight")
    get_weight(bias, f"{name}. bias")

    # add json params
    weightKey = f"{name}"
    biasKey = f"{name}"
    instance_norm_params = dict(
        {
            "layerStyle": "in",
            "layerName": name,
            "inputName": log.blobs(input, name),
            "weightKey": weightKey,
            "biasKey": biasKey,
        }
    )

    if DEBUG:
        print(instance_norm_params)

    js_param = get_parameters()
    js_param["network"].append(instance_norm_params)
    INLINE = False
    return x


# torch.topk
def _topk(raw, input, k, dim=None, largest=True, sorted=True):
    global INLINE
    x = raw(input, k, dim, largest, sorted)
    INLINE = True
    name = log.add_layer(name="topk_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # no weight extract
    # add json params
    topk_params = dict(
        {
            "layerStyle": "topk",
            "layerName": name,
            "inputName": log.blobs(input, name),
            "TopKOperation": "kMAX" if largest else "kMIN",
            "k": k,
            "reduceAxes": 1,
            "outputIndex": 0,
        }
    )
    if DEBUG:
        print(topk_params)
    js_param = get_parameters()
    js_param["network"].append(topk_params)
    INLINE = False
    return x


# torch.argmax
def _argmax(raw, input, dim, keepdim=False):
    global INLINE
    x = raw(input, dim, keepdim)
    INLINE = True
    name = log.add_layer(name="argmax_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # no weight extract
    # add json params
    argmax_params = dict(
        {
            "layerStyle": "argMax",
            "layerName": name,
            "inputName": log.blobs(input, name),
            "outputName": "argMaxTestout",
            "parameter": {
                "reShape": [1, 8, 16],
                "chooseInde": dim,
            },
        }
    )
    if DEBUG:
        print(argmax_params)
    js_param = get_parameters()
    js_param["network"].append(argmax_params)
    INLINE = False
    return x


# torch.div
def _div(raw, input, other):
    global INLINE
    x = raw(input, other)
    INLINE = True
    name = log.add_layer(name="div_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # add json params
    div_params = dict(
        {
            "layerStyle": "eltwise",
            "layerName": name,
            "eltType": "kDIV",
            "inputName": [log.blobs(input, name), log.blobs(other, name)],
        }
    )
    if DEBUG:
        print(div_params)
    js_param = get_parameters()
    js_param["network"].append(div_params)
    INLINE = False
    return x


# torch.split
def _split(raw, tensor, split_size_or_sections, dim=0):
    global INLINE
    x = raw(tensor, split_size_or_sections, dim)
    INLINE = True
    name = log.add_layer(name="split_")
    layerName = []
    start = 0
    slicePoint = [
        start,
    ]

    for i in range(len(x)):
        layerName.append(name + "_idx{}".format(i + 1))
        log.add_blobs([x[i]], name=layerName[-1])
        LayerOut_id[int(id(x[i]))] = layerName[-1]
        start += len(x[i])
        slicePoint.append(start)

    split_params = dict(
        {
            "layerStyle": "slice",
            "layerName": layerName,
            "inputName": log.blobs(tensor, name),
            "axis": dim,
            "slicePoint": slicePoint[:-1],
        }
    )
    if DEBUG:
        print(split_params)

    js_param = get_parameters()
    js_param["network"].append(split_params)
    INLINE = False
    return x


# torch.reshape
def _reshape(raw, input, shape):
    global INLINE
    x = raw(input, shape)
    INLINE = True
    name = log.add_layer(name="reshape_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    # add json params
    reshape_params = dict(
        {
            "layerStyle": "shuffle",
            "layerName": name,
            "inputName": log.blobs(input, name),
            "isReshape": True,
            "reshapeFirst": True,
            "reshape": shape,
        }
    )
    if DEBUG:
        print(reshape_params)
    js_param = get_parameters()
    js_param["network"].append(reshape_params)
    INLINE = False
    return x


# _add
def _add(input, *args):

    if isinstance(args[0], float) or isinstance(args[0], int):
        x = raw__add__(input, *args)
        return x

    global INLINE
    x = raw__add__(input, *args)
    INLINE = True
    name = log.add_layer(name="add_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name
    add_params = dict(
        {
            "layerStyle": "eltwise",
            "layerName": name,
            "eltType": "kSUM",
            "inputName": {
                "inputName_1": log.blobs(input, name),
                "inputName_2": log.blobs(args[0], name),
            },
        }
    )
    if DEBUG:
        print("__add__")
        print(add_params)
    js_param = get_parameters()
    js_param["network"].append(add_params)
    INLINE = False
    return x


# _sub
def _sub(input, *args):
    global INLINE
    x = raw__sub__(input, *args)
    INLINE = True
    name = log.add_layer(name="sub_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    sub_params = dict(
        {
            "layerStyle": "eltwise",
            "layerName": name,
            "eltType": "kSUB",
            "inputName": [log.blobs(input, name), log.blobs(args[0], name)],
        }
    )

    if DEBUG:
        print("__sub__")
        print(sub_params)

    js_param = get_parameters()
    js_param["network"].append(sub_params)
    INLINE = False
    return x


# expand_as
def _expand_as(input, *args):
    global INLINE
    x = raw__expand_as__(input, *args)
    INLINE = True
    name = log.add_layer(name="expand_as_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    expand_as_params = dict(
        {
            "layerStyle": "expand",
            "layerName": name,
            "inputName": log.blobs(input, name),
            "expand_as": log.blobs(args[0], name),
        }
    )

    if DEBUG:
        print("__expand_as__")
        print(expand_as_params)

    js_param = get_parameters()
    js_param["network"].append(expand_as_params)
    INLINE = False
    return x


def _Hwsife(input, *agrs):
    pass


# _permute
def _permute(input, *args):
    global INLINE
    x = raw__permute__(input, *args)
    INLINE = True
    name = log.add_layer(name="permute_")
    log.add_blobs([x], name=name)
    LayerOut_id[int(id(x))] = name

    permute_params = dict(
        {
            "layerStype": "shuffle",
            "layerName": name,
            "inputName": log.blobs(input, name),
            "isReshape": False,
            "reshapeFirst": False,
            "reshape": None,
            "isPermute": True,
            "permute": args,
        }
    )
    if DEBUG:
        print("__permute__")
        print(permute_params)

    js_param = get_parameters()
    js_param["network"].append(permute_params)
    INLINE = False
    return x


class RegOp(object):
    """
    Registration Operator
    """

    def __init__(self, raw, replace, **kwargs):
        self.obj = replace
        self.raw = raw

    def __call__(self, *args, **kwargs):
        global PLULINE
        if INLINE:
            return self.raw(*args, **kwargs)

        else:

            for stack in traceback.walk_stack(None):
                flag = True
                state_stack = stack[0]
                # 第一层判断
                if "self" in state_stack.f_locals:
                    module_name = type(state_stack.f_locals["self"]).__name__
                    if module_name in PLUGINS_LIST:
                        
                        module_id = 0
                        if isinstance(state_stack.f_locals, dict):
                            if "x" in state_stack.f_locals.keys():
                                module_id = id(state_stack.f_locals["x"])
                            if "input" in state_stack.f_locals.keys():
                                module_id = id(state_stack.f_locals["input"])

                        if module_name not in MODULE_DICT.keys():
                            MODULE_DICT[module_name] = module_id
                            # TODO 
                            print("module_name: ", module_name)
                            # print("module_id: ", module_id)
                            break

                        if module_name in MODULE_DICT.keys():
                            if MODULE_DICT[module_name] == module_id:
                                break
                            else:
                                MODULE_DICT[module_name] = module_id
                                # TODO
                                print("module_name: ", module_name)
                                print("module_id: ", module_id)
                                PLULINE = True
                                break

                # 向上查找
                while flag:
                    state_stack = state_stack.f_back
                    if state_stack.f_code.co_name == "_call_impl":
                        flag = False

                # 往上一层判断
                state_stack = state_stack.f_back
                if "self" in state_stack.f_locals:
                    module_name = type(state_stack.f_locals["self"]).__name__
                    if module_name in PLUGINS_LIST:
                        
                        module_id = 0
                        if isinstance(state_stack.f_locals, dict):
                            if "x" in state_stack.f_locals.keys():
                                module_id = id(state_stack.f_locals["x"])
                            if "input" in state_stack.f_locals.keys():
                                module_id = id(state_stack.f_locals["input"])


                        if module_name not in MODULE_DICT.keys():
                            MODULE_DICT[module_name] = module_id
                            # TODO 
                            print("module_name: ", module_name)
                            print("module_id: ", module_id)
                            break

                        if module_name in MODULE_DICT.keys():
                            if MODULE_DICT[module_name] == module_id:
                                break
                            else:
                                MODULE_DICT[module_name] = module_id
                                # TODO
                                print("module_name: ", module_name)
                                print("module_id: ", module_id)
                                PLULINE = True
                                break
                break

            out = self.obj(self.raw, *args, **kwargs)
            return out


def create_network():

    # 创建json param
    get_parameters()
    weight_path = os.path.join(cfg.WEIGHTS_DIR, cfg.MODELNAME, cfg.MODELNAME + ".weights")
    if os.path.exists(weight_path):
        os.remove(weight_path)
    # 第一行添加多个空格，方便采用 seek，更改参数的个数
    with open(weight_path, "w") as file:
        file.write("0         \n")
    return


def reg_functional_op():
    """
    Registration list about all torch.nn.functional support op
    """
    F.conv2d = RegOp(F.conv2d, _conv2d)
    F.relu = RegOp(F.relu, _relu)
    F.leaky_relu = RegOp(F.leaky_relu, _leaky_relu)
    F.max_pool2d = RegOp(F.max_pool2d, _max_pool2d)
    F.avg_pool2d = RegOp(F.avg_pool2d, _avg_pool2d)
    F.linear = RegOp(F.linear, _linear)
    F.adaptive_avg_pool2d = RegOp(F.adaptive_avg_pool2d, _adaptive_avg_pool2d)
    F.softmax = RegOp(F.softmax, _softmax)
    F.conv_transpose2d = RegOp(F.conv_transpose2d, _conv_transpose2d)
    F.pad = RegOp(F.pad, _pad)
    F.interpolate = RegOp(F.interpolate, _interpolate)


def reg_torch_op():
    """
    Registration list about all torch support op
    """
    torch.batch_norm = RegOp(torch.batch_norm, _batch_norm)
    torch.sigmoid = RegOp(torch.sigmoid, _sigmoid)
    torch.flatten = RegOp(torch.flatten, _flatten)
    torch.cat = RegOp(torch.cat, _cat)
    torch.instance_norm = RegOp(torch.instance_norm, _instance_norm)
    torch.topk = RegOp(torch.topk, _topk)
    torch.argmax = RegOp(torch.argmax, _argmax)
    torch.matmul = RegOp(torch.div, _div)
    torch.split = RegOp(torch.split, _split)
    torch.reshape = RegOp(torch.reshape, _reshape)


def reg_torch_nn_op():
    """
    Registration list about all torch.nn support op
    """
    # Hsigmoid = RegOp(Hsigmoid, _Hsigmoid)


def reg_tensor_op():
    """
    Registration list about all tensor support op
    """
    for tensor_ in [torch.Tensor]:
        # c = a + b
        tensor_.__add__ = _add

        # c = a - b
        tensor_.__sub__ = _sub

        # # view (instead bu torch.reshape), permute for [TRT] shuffle layer
        # tensor_.permute = RegTensorOp(tensor_.permute, _permute)
        #
        # # expand_as for [TRT] expand layer
        # tensor_.expand_as = RegTensorOp(tensor_.expand_as, _expand_as)


def reg_plugin_op():
    """
    Registration list about all plugin support op
    """
    pass


class Build:
    """
    build the configuration file.
    """

    def __init__(self, model=None, input_var=None):
        self.model = model
        self.input = input_var
        create_network()
        reg_functional_op()  # torch.nn.functional
        reg_torch_op()  # torch
        reg_torch_nn_op()  # torch.nn
        reg_tensor_op()  # torch.Tensor
        reg_plugin_op()  # plugin

    def build(self):

        print("starting ...")
        INLINE = False
        self.model.eval()

        log.init([self.input])
        with torch.no_grad():
            output = self.model(self.input)
        INLINE = True

        js_param = get_parameters()
        # mark output layer
        if len(output) >= 2:
            for i, out in enumerate(output):
                for j, layer_param in enumerate(js_param["network"]):
                    if layer_param["layerName"] == LayerOut_id[int(id(output))]:
                        js_param["network"][j]["outputName"] = f"{cfg.OUTPUTBLOBNAME}_{i + 1}"

        elif len(output) == 1:
            for j, layer_param in enumerate(js_param["network"]):
                if layer_param["layerName"] == LayerOut_id[int(id(output))]:
                    js_param["network"][j]["outputName"] = cfg.OUTPUTBLOBNAME
                    break

        # save json file
        with open(os.path.join(cfg.JSON_FILE_DIR, cfg.MODELNAME, cfg.MODELNAME + ".json"), "w") as file:
            json.dump(js_param, file, indent=4, ensure_ascii=False)

        print("successed! ...")
        return
