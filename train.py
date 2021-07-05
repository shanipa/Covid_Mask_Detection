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
from mask_utils import pad_image, save_checkpoint
from torch.utils.data import DataLoader
from maskModel import MasksDataset, define_model
from engine import train_one_epoch, evaluate, evaluate_results
import utils
from clearml import Task

def get_optimizer_and_scheduler(cfg, model_params):

    if cfg.optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model_params, lr=cfg.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
        return optimizer, lr_scheduler


def train(args, cfg_dict):

    cfg = Box(cfg_dict)

    # setting up the logger
    description = args.description
    if not description:
        description = 'Debugging'

    # Create logging directory
    task_name = f"{time.strftime('%Y-%m-%d %H:%M', time.localtime())}_{description}"

    log_dir_path = f"./experiments/training_{task_name}"
    if cfg.log_results:
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)

    # Using ClearML logger
    if cfg.log_results:
        task = Task.init(project_name='Lab_HW2',
                         task_name=task_name,
                         output_uri=log_dir_path,
                         reuse_last_task_id=args.reuse,
                         auto_resource_monitoring=False,
                         auto_connect_frameworks=False)
        logger = task.get_logger()

        cfg_dict = task.connect_configuration(cfg_dict, description=description)
        with open(f'{log_dir_path}/params.pickle', 'wb') as f:
            pickle.dump(cfg_dict, f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # defining the model
    model = define_model(cfg.model)
    model.to(device)


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer, lr_scheduler = get_optimizer_and_scheduler(cfg.optimizer, params)

    # loading the datasets

    train_dataset = MasksDataset("train")
    test_dataset = MasksDataset("test")

    # defining the data loaders
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                                  pin_memory=True, collate_fn=utils.collate_fn)

    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                                  pin_memory=True, collate_fn=utils.collate_fn)

    res_dict = {"train_loss": [], "test_loss": [], "train_iou": [], "test_iou": [], "train_acc": [], "test_acc": []}

    best_score = 0

    for epoch in range(cfg.epochs):
        loss_sum = 0
        loss_sum = train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=cfg.print_freq)
        res_dict["train_loss"].append(loss_sum)
        # update the learning rate
        lr_scheduler.step()


        # evaluate on the train dataset
        a, curr_acc, curr_iou = evaluate_results(model, train_dataloader, device=device)
        combined_score = (curr_acc + curr_iou)/2

        if cfg.log_results:
            print("Train loss:", loss_sum)
            print("Train acc:", curr_acc, "\tTrain iou:", curr_iou, "\tTrain score:", combined_score)
            logger.report_scalar(title="Loss",
                                 series="Train",
                                 iteration=epoch,
                                 value=loss_sum)

            logger.report_scalar(title="Accuracy",
                                 series="Train",
                                 iteration=epoch,
                                 value=curr_acc)

            logger.report_scalar(title="IOU",
                                 series="Train",
                                 iteration=epoch,
                                 value=curr_iou)

            logger.report_scalar(title="Score",
                                 series="Train",
                                 iteration=epoch,
                                 value=combined_score)




        # evaluate on the test dataset
        a, curr_acc, curr_iou = evaluate_results(model, test_dataloader, device=device)
        combined_score = (curr_acc + curr_iou)/2

        if cfg.log_results:
            print("Test acc:", curr_acc, "\tTest iou:", curr_iou, "\tTest score:", combined_score)

            logger.report_scalar(title="Accuracy",
                                 series="Test",
                                 iteration=epoch,
                                 value=curr_acc)

            logger.report_scalar(title="IOU",
                                 series="Test",
                                 iteration=epoch,
                                 value=curr_iou)

            logger.report_scalar(title="Score",
                                 series="Test",
                                 iteration=epoch,
                                 value=combined_score)

        if combined_score > best_score:
            best_score = combined_score
            print("yayyy")
            if cfg.log_results:
                save_checkpoint(log_dir_path, epoch, model, best_score)


    return model



if __name__=='__main__':

    # general run
    with open('config.yaml', 'r') as f:
        cfg_dict = yaml.safe_load(f)

    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, default='config.yaml', help="path to config file")
    parser.add_argument("-desc", "--description", type=str, default='',
                        help="short description of current experiment")
    parser.add_argument("-reuse", "--reuse", action='store_true', help="task id from clearml logger")
    args = parser.parse_args()

    trained_model = train(args, cfg_dict)
