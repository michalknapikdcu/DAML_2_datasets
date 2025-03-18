#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import json
import typing

image_path = '../../DAML_2_datasets/CNNs/VOC2012_selection/images/'
metadata_json = '../../DAML_2_datasets/CNNs/VOC2012_selection/metadata.json'

class BoundingBoxDataset(Dataset):

    def __init__(self, image_annotation_json, image_directory, transform = None):
        """
        Initializator: loads file with details about images, image directory,
        and transform pipeline.

        Arguments:
            image_annotation_json: assigns to each class a set of images together
                with bounding boxes (see example metadata.json in data directory)
            image_directory: here are the .jpg images
            transform: the optional transformation pipeline
        """
        self.image_directory = image_directory
        with open(image_annotation_json) as json_annotations:
            self.image_annotations = json.load(json_annotations)
        self.transform = transform

    def __len__(self):
        """Returns the number of images: sum the number of images per class."""
        return sum([len(imgs) for _,imgs in self.image_annotations.items()])

    def __getitem__(self, idx):
        """
        Checks self.image_annotations and loads and returns the image from
        corresponding index. The image is augmented with labels, i.e. class 
        and BoundingBoxes (in the current case we only have one such box).
        """

a = BoundingBoxDataset('./testdir/metadata.json', './testdir/images')