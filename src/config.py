DEBUG = 0  # debug model if DEBUG=1 else mute debug = 0

MODELNAME = ""
JSON_FILE_DIR = "/home/tianxin/mygithub/generater/weights/"
WEIGHTS_DIR = "/home/tianxin/mygithub/generater/weights/"
INPUT_C = 3
INPUT_H = 224
INPUT_W = 224
FP16 = False
INT8 = False
CALI_TXT = "cali_txt"
CALI_TABLE = "cali_table"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INPUTBLOBNAME = "data"
OUTPUTBLOBNAME = "prob"
MAXBATCHSIZE = 10
OUTPUTSIZE = 1000
