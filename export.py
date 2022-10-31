import torch
import numpy as np
import onnx
import onnx.helper as helper


all_tensors = []
objmap = {}
nodes = []
initializers = []

def hook_forward(fn):

    fnnames = fn.split(".")
    fn_module = eval(".".join(fnnames[:-1]))
    fn_name = fnnames[-1]
    oldfn = getattr(fn_module, fn_name)

    def make_hook(bind_fn):

        ilayer = 0
        def myforward(self, x):
            global all_tensors
            nonlocal ilayer
            y = oldfn(self, x)

            bind_fn(self, ilayer, x, y)
            all_tensors.extend([x, y])
            ilayer += 1
            return y
        
        setattr(fn_module, fn_name, myforward)
    return make_hook


@hook_forward("torch.nn.Conv2d.forward")
def symbolic_conv2d(self, ilayer, x, y):
    print(f"{type(self)} -> Input {get_obj_idd(x)}, Output {get_obj_idd(y)}")

    inputs = [
        get_obj_idd(x),
        append_initializer(self.weight.data, f"conv{ilayer}.weight"),
        append_initializer(self.bias.data, f"conv{ilayer}.bias")
    ]

    nodes.append(
        helper.make_node(
            "Conv", inputs, [get_obj_idd(y)], f"conv{ilayer}",
            kernel_shape=self.kernel_size, group=self.groups, pads=[0, 0] + list(self.padding), dilations=self.dilation, strides=self.stride
        )
    )


@hook_forward("torch.nn.ReLU.forward")
def symbolic_relu(self, ilayer, x, y):
    print(f"{type(self)} -> Input {get_obj_idd(x)}, Output {get_obj_idd(y)}")

    nodes.append(
        helper.make_node(
            "Relu", [get_obj_idd(x)], [get_obj_idd(y)], f"relu{ilayer}"
        )
    )

@hook_forward("torch.Tensor.__add__")
def symbolic_add(a, ilayer, b, y):
    print(f"Add -> Input {get_obj_idd(a)} + {get_obj_idd(b)}, Output {get_obj_idd(y)}")

    nodes.append(
        helper.make_node(
            "Add", [get_obj_idd(a), get_obj_idd(b)], [get_obj_idd(y)], f"add{ilayer}" 
        )
    )


@hook_forward("torch.nn.MaxPool2d.forward")
def symbolic_maxpool2d(self, ilayer, x, y):
    print(f"{type(self)} -> Input {get_obj_idd(x)}, Output {get_obj_idd(y)}")

    nodes.append(
        helper.make_node(
            "MaxPool2d", [get_obj_idd(x)], [get_obj_idd(y)], f"maxpool2d{ilayer}",
            kernel_shape=self.kernel_size, dilations=self.dilation, strides=self.stride
        )
    )


@hook_forward("torch.nn.Linear.forward")
def symbolic_linear(self, ilayer, x, y):
    print(f"{type(self)} -> Input {get_obj_idd(x)}, Output {get_obj_idd(y)}")

    inputs = [
        get_obj_idd(x),
        append_initializer(self.weight.data, f"linear{ilayer}.weight"),
        append_initializer(self.bias.data, f"linear{ilayer}.bias")
    ]

    nodes.append(
        helper.make_node(
            "Linear", inputs, [get_obj_idd(y)], f"linear{ilayer}")
    )

@hook_forward("torch.nn.AdaptiveAvgPool2d.forward")
def symbolic_adaptiveAvgpool2d(self, ilayer, x, y):
    print(f"{type(self)} -> Input {get_obj_idd(x)}, Output {get_obj_idd(y)}")

    nodes.append(
        helper.make_node(
            "AdaptiveAvgPool2d", [get_obj_idd(x)], [get_obj_idd(y)], f"AdaptiveAvgPool2d{ilayer}",
            output_size=self.output_size,
        )
    )

@hook_forward("torch.flatten")
def symbolic_flatten(x, ilayer, _, y):
    print(f"flatten -> Input {get_obj_idd(x)} , Output {get_obj_idd(y)}")

    nodes.append(
        helper.make_node(
            "flatten", [get_obj_idd(x)], [get_obj_idd(y)], f"flatten{ilayer}",
            start_dim=1,
        )
    )


def append_initializer(value, name):
    initializers.append(
        helper.make_tensor(
            name=name,
            data_type=helper.TensorProto.DataType.FLOAT,
            dims=list(value.shape),
            vals=value.data.cpu().numpy().astype(np.float32).tobytes(),
            raw=True
        )
    )
    return name


def get_obj_idd(obj):
    global objmap

    idd = id(obj)
    if idd not in objmap:
        objmap[idd] = str(len(objmap))
    return objmap[idd]


class export:
    def __init__(self, input, output):

        self.inputs = [
            helper.make_value_info(
                name="0",
                type_proto=helper.make_tensor_type_proto(
                    elem_type=helper.TensorProto.DataType.FLOAT,
                    shape=["batch", input.size(1), input.size(2), input.size(3)]
                )
            )
        ]

        self.outputs = [
            helper.make_value_info(
                name=str(len(objmap) - 1),
                type_proto=helper.make_tensor_type_proto(
                    elem_type=helper.TensorProto.DataType.FLOAT,
                    shape=["batch", output.size(1)]
                )
            )
        ]

        self.nodes = nodes

        self.initializer = initializers

        self.graph = helper.make_graph(
            name="mymodel",
            inputs=self.inputs,
            outputs=self.outputs,
            nodes=self.nodes,
            initializer=self.initializer
        )

        self.opset = [ helper.make_operatorsetid("ai.onnx", 11) ]


    def run(self):

        model = helper.make_model(self.graph, opset_imports=self.opset, producer_name="pytorch", producer_version="1.10")

        onnx.save_model(model, "save.onnx")

