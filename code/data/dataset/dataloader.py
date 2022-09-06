# ======================================================================
# CS-GAN
# Modified from MC-GAN (Samaneh Azadi)
# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# By Zhenjie Jin
# ======================================================================

import os
import random

import warnings
import torchvision.transforms as transforms


def normalize_stack(input, val=0.5):
    # normalize an tensor with arbitrary number of channels:
    # each channel with mean=std=val
    len_ = input.size(1)
    mean = (val, ) * len_
    std = (val, ) * len_
    t_normal_stack = transforms.Compose([transforms.Normalize(mean, std)])
    return t_normal_stack(input)


def CreateDataLoader(opt):
    data_loader = None
    if opt.stack:
        data_loader = StackDataLoader()
    elif opt.partial:
        data_loader = PartialDataLoader()
    else:
        data_loader = DataLoader()
    data_loader.initialize(opt)
    return data_loader


class FlatData(object):
    def __init__(self,
                 data_loader,
                 data_loader_base,
                 fineSize,
                 max_dataset_size,
                 rgb,
                 dict_test={},
                 base_font=False,
                 blank=0.7
                 ):
        self.data_loader = data_loader
        self.data_loader_base = data_loader_base
        self.fineSize = fineSize
        self.max_dataset_size = max_dataset_size
        self.rgb = rgb
        self.blanks = blanks
        self.data_loader_base_iter  = iter(self.data_loader_base_iter)
        self.base_font = base_font
        self.A_base, self.A_base_paths = next(self.data_loader_base_iter)
        self.random_dict = dict_test
        self.dict = False
        if len(self.random_dict.keys()):
            self.dict = True

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):