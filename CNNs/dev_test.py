#!/usr/bin/python3

import torch
import torchvision.transforms.functional as F
from torchvision import transforms, utils, tv_tensors
from torchvision.transforms import v2 as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import json

image_path = 'images/'
metadata_json = 'metadata.json'

class BoundingBoxDataset(Dataset):

    def __init__(self, image_annotation_json, image_directory, target_size, transform_list=None):
        """
        Initializator: loads file with details about images, image directory,
        and transform pipeline. Also builds transform pipelines: one for pre-transforming all
        images (i.e. resize, change type to float32, normalize), and a custom one.
        It also calls auxiliary functions for computing pixel means and standard deviations.

        Arguments:
            image_annotation_json: assigns to each class a set of images together
                with bounding boxes (see example metadata.json in data directory)
            image_directory: here are the .jpg images
            target_size: all images and bounding boxes will be resized to this size;
                the first step of transform pipeline required for both the models and DataLoader
            transform: a list of transforms that will be put into the optional transformation 
                pipeline (use only transforms that can handle BoundingBoxes); 
        """

        # load image annotations 
        print('-- loading annotations')
        self.image_directory = image_directory
        with open(image_annotation_json) as json_annotations:
            self.image_annotations = json.load(json_annotations)

        self.target_size = target_size

        # this set of transforms is common and used to prepare the images for further
        # processing: resizing the images and bounding boxes to the provided size and 
        # normalizing the images

        # firstly compute means and std deviations needed to normalize the images 
        self.img_channel_means, self.img_channel_std = None, None
        self._compute_means_and_stddev()

        # now, compile the common image transformation pipeline
        self.pretransform_pipeline = transforms.Compose([transforms.Resize(self.target_size),\
                                                        transforms.ToDtype(torch.float32, scale=True),\
                                                        transforms.Normalize(self.img_channel_means,\
                                                                             self.img_channel_std)])
        # compile the custom pipeline 
        self.transform = None
        if transform_list is not None:
            self.transform = transforms.Compose(transform_list)

        # transform class names to class indices
        self.classes = list({details['class'] for details in self.image_annotations})
        for details in self.image_annotations:
            descriptive_class = details['class']
            details['class'] = self.classes.index(descriptive_class)

        self.suppress_transforms = False

        print('-- dataset prepared ')
        print(f'-- loaded {len(self.image_annotations)} annotations', end=' ')
        print(f'with the following classes: {self.classes}')

    def _compute_means_and_stddev(self):
        """
        An auxiliary function for computing means and std deviations over channels for
        the entire dataset. (For bigger datasets it would make more sense to estimate these
        values e.g. over minibatches).
        """

        print('-- computing means and std deviations over the dataset')

        resize_and_convert_pipeline = transforms.Compose([transforms.Resize(self.target_size),\
                                                          transforms.ToDtype(torch.float32, scale=True)])
        # means are computed by summing all pixel values per channel, and dividing by
        # the number of pixels
        # std are computed similarly, using formula std(X) = sqrt(E(X-EX)^2) = sqrt(E(X^2)-(EX)^2)
        means = torch.zeros((3,1,1), dtype=torch.float32) 
        stds = torch.zeros((3,1,1), dtype=torch.float32) 
        all_pixels_cnt = self.target_size[0]*self.target_size[1]*len(self.image_annotations)

        for idx in range(0, len(self.image_annotations)):
            _,curr_img = self.read_details_and_image_from_annotations(idx)
            transformed_img = resize_and_convert_pipeline(curr_img)

            # add pixel values per channel
            means += transformed_img.sum([1,2],keepdim=True)

            # add sums of squares of pixel values per channel
            stds += (transformed_img * transformed_img).sum([1,2],keepdim=True)

        means = means/all_pixels_cnt
        stds = torch.sqrt(stds/all_pixels_cnt - means*means)

        self.img_channel_means = means.reshape([3])
        self.img_channel_std = stds.reshape([3])

    def __len__(self):
        """Returns the number of images: sum the number of images per class."""
        return len(self.image_annotations)

    def transform_suppression(self, suppress):
        """Controls if transforms should be suppressed in index access (used in visualisations)"""
        self.transform_suppression = suppress

    def read_details_and_image_from_annotations(self, idx):
        """
        A helper function that returns image details and the image read from file, 
        from self.annotations.
        
        Arguments:
            idx: the index (not bound-checked) of the image in self.image_annotations
        """
        img_details = self.image_annotations[idx]
        return img_details,read_image(self.image_directory + '/' + img_details['file'])

    def __getitem__(self, idx):
        """
        Checks self.image_annotations and loads and returns the image from the
        corresponding index. The image is augmented with labels, i.e. class 
        and BoundingBoxes (in the current case we only have one such box).

        Arguments:
            idx: the index of image from the dataset
        """

        # read image from .jpg found in annotation under a given idx
        img_details,curr_image = self.read_details_and_image_from_annotations(idx)

        # fetch class id from annotations
        curr_class = img_details['class']
        _,img_height,img_width = curr_image.shape

        # convert the bounding box from annotation to BoundingBoxes
        curr_annot_bbox = [img_details['xmin'], img_details['ymin'], \
                           img_details['xmax'], img_details['ymax']]

        curr_bbox = tv_tensors.BoundingBoxes(curr_annot_bbox, format="XYXY", \
                                             canvas_size=(img_height, img_width))

        # apply transformations
        if not self.suppress_transforms:

            # mandatory transformations:  
            # resize, convert to float and normalize both the image and bounding boxes
            curr_image = self.pretransform_pipeline(curr_image)
            curr_bbox = self.pretransform_pipeline(curr_bbox)

            # TODO now
            if self.transform:
                curr_image = self.transform(curr_image)
                curr_bbox = self.transform(curr_bbox)
        
        # Datasets are usually expected to return pairs of (datapoint, target),
        # which means in our case: (image, (class, bounding box)) 

        return (curr_image, (curr_class, curr_bbox))

a = BoundingBoxDataset('metadata.json', 'images', (333, 500))
a.suppress_transforms = False
train_dataloader = DataLoader(a, batch_size=64, shuffle=True)

# SHOWER 

img,(cl,bbox) = a[40]

pilim = F.to_pil_image(img)
boxes = draw_bounding_boxes(img,bbox,colors=['red'])
F.to_pil_image(boxes).show()
#show(boxes)

#print(a[0])

#for a in a:
#    print(a)


## TODO - te transformacje https://pytorch.org/vision/main/transforms.html