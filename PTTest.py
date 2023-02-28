import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import glob
import cv2
import torch
import torch.nn as nn
import yaml


# input_tensor = torch.rand(1, 3, 80, 80)
# # print(input_tensor.size(2))

# embedding_conv_0 = nn.Conv2d(input_tensor.size(1), 256, 3, stride=1, padding=1, bias=False)
# # bn = nn.BatchNorm2d(256)
# # avg_pool_0 = nn.AvgPool2d((80, 80), stride=2)

# embedding_tensor = embedding_conv_0(input_tensor)
# # embedding_tensor_bn = bn(embedding_tensor)
# # embedding_tensor_bn_avgpool = avg_pool_0(embedding_tensor_bn)

# # fc_in = embedding_tensor_bn_avgpool.view(embedding_tensor_bn_avgpool.size(0), -1)

# print(embedding_tensor.shape)



# m = nn.BatchNorm1d(100, affine=False)
# input = torch.randn(1, 100)
# output = m(input)

# m = nn.Softmax(dim=1)
# input = torch.randn(2, 572, 80, 80)
# output = m(input)
# print(output.shape)

# input = torch.randn(1, 2, 4, 4)
# print(input)
# input = torch.argmax(input, dim=1)
# print(input.shape)
# print(input)

# input = input.unsqueeze(1)
# print(input.shape)
# print(input)

# input = torch.randn(1, 80, 80, 128)
# classifier = nn.Linear(128, 20)
# output = classifier(input)
# print(output.shape)

# Example of target with class indices
# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5)
# target = torch.empty(3, dtype=torch.long)
# print("target.shape: {}".format(target.shape))
# target = target.random_(5)
# print("target.shape: {}".format(target.shape))
# output = loss(input, target)

# # Example of target with class probabilities
# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5)
# target = torch.randn(3, 5).softmax(dim=1)
# print("target.shape: {}".format(target.shape))
# # output = loss(input, target)

# target = np.asarray([[1,5,9],
#                      [5,8,1]])
# target = target[:, 1]
# print(target.shape)
# target = torch.nn.functional.one_hot(torch.from_numpy(target))
# print("target.shape: {}".format(target.shape))

# r = torch.randn(1, 11, 2)
# l = torch.randn(5, 1, 2)

# res = r + l
# print(res.shape)

# uniform = np.random.uniform(-1, 1, 3)
# print(type(uniform))
# print(uniform.shape)
# print(uniform)

# a = 1
# b = 5
# N = 2
# ss = a+(b-a)*(np.random.random_integers(N)-1)/(N-1)
# print(ss)

# x = {}
# x["name"] = "derek"
# x["grade"] = 100
# np.save("test.cache", x)


# x = np.load("test.cache.npy", allow_pickle=True).item()
# print(x.keys())
# print(x.values())
# print(type(x['name']))
# print(type(x['grade']))

# atensor = torch.randint(0, 10, size=(2, 4))
# max_atensor = atensor.max(1)
# print(atensor)
# print(max_atensor)

# targets = torch.rand(3, 2)
# print(targets)

# target_t  = targets%1. < 0.5
# print(target_t)

# i, j = target_t.T
# print(i)
# print(j)

# offset = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]).long()
# print(offset.dtype)

# dict_0 = {}
# dict_0["name"] = "derek"
# dict_0["age"] = 35

# file=open("./sai.yaml",'w',encoding='utf-8')
# yaml.dump(dict_0, file)
# file.close()

# with open("./sai.yaml") as f:
#     sai = yaml.safe_load(f)  # model dict
# print(sai["age"])
# print(sai.get("name"))

# x = '/2T/001_AI/3001_YOLOv5_JDE/001_AL/YOLOv5_JDE/images/000000.jpg'
# print(x.rsplit('/001_AL/', 1))

# sa, sb = os.sep + 'images' + os.sep, os.sep + 'gt' + os.sep
# print(sb.join(x.rsplit(sa, 1)))
# print(sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt')

bi = np.floor(np.arange(33) / 4).astype(np.int)
print(bi)
nb = bi[-1] + 1  # number of batches
print(nb)
shapes = [[[1, 1]]] * nb
print(shapes)