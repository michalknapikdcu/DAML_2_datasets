#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import tv_tensors
import json
import typing

image_path = 'images/'
metadata_json = 'metadata.json'

class BoundingBoxDataset(Dataset):

    def __init__(self, image_annotation_json, image_directory, transform = None):
        """
        Initializator: loads file with details about images, image directory,
        and transform pipeline.

        Arguments:
            image_annotation_json: assigns to each class a set of images together
                with bounding boxes (see example metadata.json in data directory)
            image_directory: here are the .jpg images
            transform: the optional transformation pipeline (use only transforms
                that can handle BoundingBoxes)
        """

        # load image annotations 
        self.image_directory = image_directory
        with open(image_annotation_json) as json_annotations:
            self.image_annotations = json.load(json_annotations)
        self.transform = transform

        # transform class names to class indices
        self.classes = list({details['class'] for details in self.image_annotations})
        for details in self.image_annotations:
            descriptive_class = details['class']
            details['class'] = self.classes.index(descriptive_class)

        print('-- dataset prepared ')
        print(f'-- loaded {len(self.image_annotations)} annotations', end=' ')
        print(f'with the following classes: {self.classes}')

    def __len__(self):
        """Returns the number of images: sum the number of images per class."""
        return len(self.image_annotations)

    def __getitem__(self, idx):
        """
        Checks self.image_annotations and loads and returns the image from
        corresponding index. The image is augmented with labels, i.e. class 
        and BoundingBoxes (in the current case we only have one such box).
        """

        # read image from .jpg found in annotation no idx
        img_details = self.image_annotations[idx]
        curr_image = read_image(self.image_directory + '/' + img_details['file'])

        # fetch class id from annotations
        curr_class = img_details['class']
        _,img_height,img_width = curr_image.shape

        # convert the bounding box from annotation to BoundingBoxes
        curr_annot_bbox = [img_details['xmin'], img_details['ymin'], \
                           img_details['xmax'], img_details['ymax']]
        
        # if self.transform:
        #     curr_image = self.transform(curr_image)
        # return curr_image
       
        print(curr_image.shape)
        print(img_details['file'])
        curr_bbox = tv_tensors.BoundingBoxes(curr_annot_bbox, format="XYXY", \
                                             canvas_size=(img_height, img_width))
        return curr_annot_bbox, curr_bbox
    
a = BoundingBoxDataset('metadata.json', 'images')
print(a[0])

## TODO - te transformacje https://pytorch.org/vision/main/transforms.html