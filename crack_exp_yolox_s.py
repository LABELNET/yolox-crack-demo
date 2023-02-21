#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # 1. 修改为数据集中初始化的 coco 格式路径
        self.data_dir = "datasets/crack"
        # 2. coco数据集 默认值，不变
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        # 3. 标签种类数 1
        self.num_classes = 1
        # 4. 训练世代数
        self.max_epoch = 30
        self.data_num_workers = 4
        self.eval_interval = 1
