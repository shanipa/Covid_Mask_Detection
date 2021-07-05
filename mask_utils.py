import os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import transforms as T



def calc_iou(bbox_a, bbox_b):
    """.to(device)
    Calculate intersection over union (IoU) between two bounding boxes with a (x, y, w, h) format.
    :param bbox_a: Bounding box A. 4-tuple/list.
    :param bbox_b: Bounding box B. 4-tuple/list.
    :return: Intersection over union (IoU) between bbox_a and bbox_b, between 0 and 1.
    """
    x1, y1, w1, h1 = bbox_a
    x2, y2, w2, h2 = bbox_b
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0.0 or h_intersection <= 0.0:  # No overlap
        return 0.0
    intersection = w_intersection * h_intersection
    union = w1 * h1 + w2 * h2 - intersection  # Union = Total Area - Intersection
    return intersection / union


def show_images_and_bboxes(data, image_dir):
    """
    Plot images with bounding boxes. Predicts random bounding boxes and computes IoU.
    :param data: Iterable with (filename, image_id, bbox, proper_mask) structure.
    :param image_dir: Path to directory with images.
    :return: None
    """
    for filename, image_id, bbox, proper_mask in data:
        # Load image
        im = cv2.imread(os.path.join(image_dir, filename))
        # BGR to RGB
        im = im[:, :, ::-1]
        # Ground truth bbox
        x1, y1, w1, h1 = bbox
        # Predicted bbox
        x2, y2, w2, h2 = random_bbox_predict(bbox)
        # Calculate IoU
        iou = calc_iou(bbox, (x2, y2, w2, h2))
        # Plot image and bboxes
        fig, ax = plt.subplots()
        ax.imshow(im)
        rect = patches.Rectangle((x1, y1), w1, h1,
                                 linewidth=2, edgecolor='g', facecolor='none', label='ground-truth')
        ax.add_patch(rect)
        rect = patches.Rectangle((x2, y2), w2, h2,
                                 linewidth=2, edgecolor='b', facecolor='none', label='predicted')
        ax.add_patch(rect)
        fig.suptitle(f"proper_mask={proper_mask}, IoU={iou:.2f}")
        ax.axis('off')
        fig.legend()
        plt.show()


def random_bbox_predict(bbox):
    """
    Randomly predicts a bounding box given a ground truth bounding box.
    For example purposes only.
    :param bbox: Iterable with numbers.
    :return: Random bounding box, relative to the input bbox.
    """
    return [x + np.random.randint(-15, 15) for x in bbox]


def pad_image(im, desired_size=224):
    old_size = im.size
    #     old_size = cv_im.shape[:2]

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    #     padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    padding = (0, 0, delta_h, delta_w)
    new_im = ImageOps.expand(im, padding)

    #     top, bottom = 0, delta_h
    #     left, right = 0, delta_w

    #     color = [0, 0, 0]
    #     new_im = cv2.copyMakeBorder(cv_im, top, bottom, left, right, cv2.BORDER_CONSTANT,
    #         value=color)
    return new_im


def plot(res_dict):
    plt.plot(res_dict["train_acc"], c="red", label="train accuracy")
    plt.plot(res_dict["test_acc"], c="blue", label="test accuracy")
    plt.ylim(0, 1.1)
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()

    plt.savefig(f'acc.png')
    open(f'train_acc.txt', 'w').write(str(res_dict["train_acc"]))
    open(f'test_acc.txt', 'w').write(str(res_dict["test_acc"]))

    plt.clf()
    plt.plot(res_dict["train_loss"], c="red", label="train loss")
    plt.plot(res_dict["test_loss"], c="blue", label="test loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.grid()
    plt.legend()
    plt.savefig(f'loss.png')
    plt.clf()
    open(f'train_loss.txt', 'w').write(str(res_dict["train_loss"]))
    open(f'test_loss.txt', 'w').write(str(res_dict["test_loss"]))

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        #transforms.append(T.RandomHorizontalFlip(0.5))
        pass
        # TODO: understand better the filp

    return T.Compose(transforms)

def save_checkpoint(path, epoch, model, best_score, grid=None):

    check_point_path = os.path.join(path, f"{epoch}_{best_score}")

    if not os.path.exists(check_point_path):
        os.mkdir(check_point_path)
    #
    # # save grid to model check point
    # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    #
    # im = Image.fromarray(ndarr)
    # im.save(os.path.join(check_point_path, 'examples.jpeg'))

    # saving the model
    model_path = os.path.join(check_point_path, 'model.pth')
    torch.save(model.state_dict(), model_path)

