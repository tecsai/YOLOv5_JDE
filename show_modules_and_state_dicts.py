import sys
import argparse
import os
import struct
import torch
from utils.torch_utils import select_device


def parse_args():
    parser = argparse.ArgumentParser(description='Convert .pt file to .wts')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    args = parser.parse_args()
    return args.weights


pt_file = parse_args()

# Initialize
device = select_device('cpu')
# Load model
model = torch.load(pt_file, map_location=device)['model'].float()  # load to FP32

# update anchor_grid info
anchor_grid = model.model[-1].anchors * model.model[-1].stride[...,None,None]
# model.model[-1].anchor_grid = anchor_grid
delattr(model.model[-1], 'anchor_grid')  # model.model[-1] is detect layer
model.model[-1].register_buffer("anchor_grid",anchor_grid) #The parameters are saved in the OrderDict through the "register_buffer" method, and then saved to the weight.

model.to(device).eval()


""" modules """
f =  open("model_modules.txt","w")
for k, v in model.named_modules():
    f.write("{}\n".format(k))
    f.write("{}\n".format(v))
f.close()

""" state_dicts """
f =  open("model_state_dicts.txt","w")
for k, v in model.state_dict().items():
    f.write("{}\n".format(k))
    if k == "model.24.anchor_grid":
        f.write("{}\n".format(v))
    if k == "model.24.anchors":
        f.write("{}\n".format(v))
f.close()
