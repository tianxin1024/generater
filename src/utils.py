import os
import json
import struct
from . import config as cfg

LayerOut_id = dict({})


REGISTERED_LIST = [
    # ================= TensorRT Support ===========================#
    "Conv2d",
    "ConvTranspose2d",
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "ZeroPad2d",
    "Linear",
    "LeakyReLU",
    "Sigmoid",
    "Softmax",
    "MaxPool2d",
    "AvgPool2d",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    # ================= TensorRT Not Support ===========================#
    "AdaptiveAvgPool2d",
]


class Json:
    """
    Json class to dump json_file
    """

    def __init__(self, json_file):
        self.json_file = json_file
        self.data = {}
        self.data["input_c"] = cfg.INPUT_C
        self.data["input_h"] = cfg.INPUT_H
        self.data["input_w"] = cfg.INPUT_W
        self.data["enginePath"] = os.path.join(cfg.JSON_FILE_DIR, cfg.MODELNAME, cfg.MODELNAME + ".engine")
        self.data["onnxPath"] = os.path.join(cfg.JSON_FILE_DIR, cfg.MODELNAME, cfg.MODELNAME + ".onnx")
        self.data["weightPath"] = os.path.join(cfg.WEIGHTS_DIR, cfg.MODELNAME, cfg.MODELNAME + ".weights")
        self.data["weightsDir"] = os.path.join(cfg.WEIGHTS_DIR, cfg.MODELNAME)
        if not os.path.exists(self.data["weightsDir"]):
            os.makedirs(self.data["weightsDir"])
        if cfg.INT8:
            self.data["int8"] = cfg.INT8
        else:
            self.data["fp16"] = cfg.FP16

        self.data["cali_txt"] = cfg.CALI_TXT
        self.data["cali_table"] = cfg.CALI_TABLE
        self.data["Mean"] = cfg.MEAN
        self.data["Std"] = cfg.STD
        self.data["inputBlobName"] = cfg.INPUTBLOBNAME
        self.data["outputBlobName"] = cfg.OUTPUTBLOBNAME
        self.data["maxBatchsize"] = cfg.MAXBATCHSIZE
        self.data["outputSize"] = cfg.OUTPUTSIZE
        self.data["network"] = []

    def get_json_param(self):
        return self.data

    def dump(self):
        with open(self.json_file, "w") as file:
            json.dump(self.data, file, indent=4, ensure_ascii=False)


class BlobLog:
    """
    BlobLog class for setting and getting data with id(data).
    """

    def __init__(self):
        self.data = {}

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)


class TransLog(object):
    """
    Core components for connecting up and down of network
    """

    def __init__(self):
        self.layers = {}
        self.detail_layers = {}
        self.detail_blobs = {}
        self._blobs = BlobLog()
        self._blobs_data = []

    def init(self, inputs):
        """
        init input data
        """
        LayerOut_id[int(id(inputs))] = "data"
        self.add_blobs(inputs, name="data")

    def add_layer(self, name="layer"):
        if name in self.layers:
            return self.layers[name]
        if name not in self.detail_layers.keys():
            self.detail_layers[name] = 0
        self.detail_layers[name] += 1
        name = "{}{}".format(name, self.detail_layers[name])
        self.layers[name] = name

        return self.layers[name]

    def add_blobs(self, blobs, name="blob"):
        rst = []
        for blob in blobs:
            self._blobs_data.append(blob)
            blob_id = int(id(blob))
            rst.append(name)
            self._blobs[blob_id] = rst[-1]

        return rst

    def blobs(self, var, name):
        var = id(var)
        try:
            return self._blobs[var]
        except:
            # print("Warning: cannot found blob {}".format(name))
            return None


def get_weight(weight, name):

    with open(os.path.join(cfg.WEIGHTS_DIR, cfg.MODELNAME, cfg.MODELNAME + ".weights"), "r+") as rfile:
        count = rfile.readline().split()[0]
        count = str(int(count) + 1)
        rfile.seek(0)
        rfile.write(count)
    rfile.close()

    with open(os.path.join(cfg.WEIGHTS_DIR, cfg.MODELNAME, cfg.MODELNAME + ".weights"), "a") as file:

        value = weight.reshape(-1).cpu().numpy()

        file.write("{} {}".format(name, len(value)))
        for item in value:
            file.write(" ")
            file.write(struct.pack(">f", float(item)).hex())
        file.write("\n")
    file.close()


# def get_weight(weight, name):
#     ts = weight.cpu().detach().numpy().copy()
#     shape = ts.shape
#     size = shape
#     allsize = 1
#     for idx in range(len(size)):
#         allsize *= size[idx]
#
#     ts = ts.reshape(allsize)
#     with open(os.path.join(cfg.WEIGHTS_DIR, name), "wb") as file:
#         temp = struct.pack("i", allsize)
#         file.write(temp)
#         for i in range(allsize):
#             temp = struct.pack("f", ts[i])
#             file.write(temp)
