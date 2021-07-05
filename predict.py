import time
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import pickle
from argparse import ArgumentParser
import torch
import torch.nn as nn
import yaml
from box import Box
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from mask_utils import pad_image, save_checkpoint
from torch.utils.data import DataLoader
from maskModel import MasksDataset, define_model
from engine import train_one_epoch, evaluate, evaluate_results
import utils
import argparse
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

PATH = './model.pth'

download_file_from_google_drive("1jzf_XUfU5jvXDe7J5_mwdAr6TiwZI5-N", PATH)

with open('config.yaml', 'r') as f:
    cfg_dict = yaml.safe_load(f)

cfg = Box(cfg_dict)
# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Reading input folder
files = [] #os.listdir(args.input_folder)
proper_mask_pred = []
bbox_pred = []
#####
# TODO - your prediction code here
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = define_model(cfg.model)
model.load_state_dict(torch.load(PATH))
model.eval()
model.to(device)
dataset = MasksDataset(args.input_folder)
dataloader = DataLoader(dataset, shuffle=False, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                             pin_memory=True, collate_fn=utils.collate_fn)
cpu_device = torch.device("cpu")

all_outputs = []

with torch.no_grad():
    for images, targets in dataloader:

        images = list(image.to(device) for image in images)

        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        for prediction, target in zip(outputs, targets):

            filename = dataset.data[target["image_id"]][0]
            files.append(filename)

            try:
                box_pred = prediction['boxes'][0]
                box_pred = [box_pred[0], box_pred[1], box_pred[2] - box_pred[0], box_pred[3] - box_pred[1]]
                bbox_pred.append(box_pred)
                proper_mask_pred.append(bool(prediction['labels'].item()-1))

            except:
                print(filename)
                box_pred = np.random.randint(0, high=224, size=(4))
                bbox_pred.append(list(box_pred))
                proper_mask_pred.append(False)




# bbox_pred = np.random.randint(0, high=224, size=(4, len(files)))
# proper_mask_pred = np.random.randint(2, size=len(files)).astype(np.bool)
bbox_pred = np.array(bbox_pred).T
prediction_df = pd.DataFrame(zip(files, *bbox_pred, proper_mask_pred),
                             columns=['filename', 'x', 'y', 'w', 'h', 'proper_mask'])
prediction_df.to_csv("prediction.csv", index=False, header=True)
####


