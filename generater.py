import torch
import argparse
import models
from src import Build
from torchsummary import summary
from src import config


model_names = models.__all__

parser = argparse.ArgumentParser(description="PyTorch Auto generate json for TensorRT!")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="alexnet",
    choices=model_names,
    help="model architecture",
)
parser.add_argument(
    "-c", "--checkpoints", dest="checkpoints", default=None, type=str, help="load checkpoints to evaluate"
)
parser.add_argument("--pretrained", dest="pretrained", action="store_false", help="use pre-trained model")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.pretrained:
        print(
            "=> using pre-trained model '{}', checkpoints '{}'".format(
                args.arch, args.checkpoints if args.checkpoints is not None else "torchvision"
            )
        )
        # model = models.__dict__[args.arch](pretrained=True)
        model = model_names[args.arch](pretrained=True, checkpoints=args.checkpoints)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = model_names[args.arch]()

    # parameter check
    config.MODELNAME = args.arch

    models_list = [
        "alexnet",
        "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg19",
        "vgg19_bn",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "wide_resnet50_2",
        "wide_resnet101_2",
        "squeezenet1_0",
        "squeezenet1_1",
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201",
        "mobilenet_v2",
    ]

    if args.arch in models_list:
        input_var = torch.ones([1, 3, 224, 224])
        args.INPUT_C = 3
        args.INPUT_H = 224
        args.INPUT_W = 224
    else:
        input_var = torch.ones([1, 3, 640, 640])
        args.INPUT_C = 3
        args.INPUT_H = 640
        args.INPUT_W = 640

    model.eval()
    import ipdb; ipdb.set_trace()
    model = model.to("cuda:0")
    # input_var = torch.ones([1, 3, 224, 224])

    # summary(model, (input_var.shape[1], input_var.shape[2], input_var.shape[3]))

    builder = Build(model.to("cuda:0"), input_var.to("cuda:0"))
    builder.build()
