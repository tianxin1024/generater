from .alexnet.alexnet import alexnet
# from .vgg.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
# from .resnet.resnet import (
#     resnet18,
#     resnet34,
#     resnet50,
#     resnet101,
#     resnet152,
#     resnext50_32x4d,
#     resnext101_32x8d,
#     wide_resnet50_2,
#     wide_resnet101_2,
# )
# from .squeezenet.squeezenet import squeezenet1_0, squeezenet1_1
# from .densenet.densenet import densenet121, densenet161, densenet169, densenet201
# from .mobilenet.mobilenetv2 import mobilenet_v2
# from .mobilenet.mobilenetv3 import mobilenetv3_large, mobilenetv3_small


__all__ = dict(
    {
        "alexnet": alexnet,
        # "vgg11": vgg11,
        # "vgg11_bn": vgg11_bn,
        # "vgg13": vgg13,
        # "vgg13_bn": vgg13_bn,
        # "vgg16": vgg16,
        # "vgg16_bn": vgg16_bn,
        # "vgg19": vgg19,
        # "vgg19_bn": vgg19_bn,
        # "resnet18": resnet18,
        # "resnet34": resnet34,
        # "resnet50": resnet50,
        # "resnet101": resnet101,
        # "resnet152": resnet152,
        # "resnext50_32x4d": resnext50_32x4d,
        # "resnext101_32x8d": resnext101_32x8d,
        # "wide_resnet50_2": wide_resnet50_2,
        # "wide_resnet101_2": wide_resnet101_2,
        # "squeezenet1_0": squeezenet1_0,
        # "squeezenet1_1": squeezenet1_1,
        # "mobilenet_v2": mobilenet_v2,
        # "densenet121": densenet121,
        # "densenet161": densenet161,
        # "densenet169": densenet169,
        # "densenet201": densenet201,
        # "mobilenetv3_large": mobilenetv3_large,
        # "mobilenetv3_small": mobilenetv3_small,
    }
)
