import time
import os
from PIL import Image, ImageOps
import pickle
from argparse import ArgumentParser
import torch
import torch.nn as nn
import yaml
from box import Box
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from mask_utils import parse_images_and_bboxes, pad_image, get_transform


class MasksDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        is_train = False
        if root == 'train':
            is_train = True
        self.transforms = get_transform(is_train)
        # load all image files, sorting them to
        # ensure that they are aligned
        self.data = parse_images_and_bboxes(root)

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.data[idx][0])
        img = pad_image(Image.open(img_path).convert("RGB"))

        image_id = torch.tensor([idx])
        box = self.data[idx][2]
        box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        labels = self.data[idx][3]
        labels = torch.as_tensor([labels, ], dtype=torch.int64) + 1
        area = torch.tensor([(box[3] - box[1]) * (box[2] - box[0]), ])
        iscrowd = torch.zeros((1,), dtype=torch.int8)
        box = torch.as_tensor([box, ], dtype=torch.float32)
        target = {}
        target["boxes"] = box
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.data)


def define_model(cfg):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=cfg.pretrain, pretrained_backbone=cfg.pretrain,
                                                                 box_detections_per_img=1)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.num_classes)

    return model






