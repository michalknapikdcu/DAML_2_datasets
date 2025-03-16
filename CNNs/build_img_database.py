#!/usr/bin/env python3

import sys
import os
import random
import xml.etree.ElementTree as eltree 

# This function check if class, annotation, and jpeg files are reachable.
def sanitize_data_sources(voc_class_dir, voc_annotation_dir, voc_jpeg_dir, voc_classes):

    data_dirs = [(voc_class_dir, '.txt class'), (voc_annotation_dir, 'annotations'), \
                 (voc_jpeg_dir, 'images')]

    # check that directories exist 
    for dir, purpose in data_dirs:
        if not os.path.exists(dir):
            print(f'error: cannot find {purpose} directory under {dir}')
            sys.exit(1)

    # check that class .txt files are there
    for voc_class in voc_classes:
        voc_class_file = voc_class_dir + voc_class + '_trainval.txt'

        if not os.path.exists(voc_class_file):
            print(f'error: cannot find {voc_class} file under {voc_class_file}')
            sys.exit(1)

# This function, for a given class named class_name:
# - uses (sanitized) voc_* parameters to reach files from VOC 2012 dataset
# - visits .txt class file of class class_name and collects image file names s.t.
#   each image contains a single object with a bounding box
# - it then returns a dict with details about the class and files
# - (if no_imgs is not None, then it tries to return a draw of no_imgs from the dict)
def collect_dir_for_class(voc_class_dir, voc_annotation_dir, voc_jpeg_dir,\
                            class_name, no_imgs = None):

    file_name_to_bounding_box_dict = {}

    # read class spec. file and collect the names of image files labeled with '1' 
    # (these contain easily recognizable objects of the class)
    path_to_file = voc_class_dir + class_name + '_trainval.txt'
    with open(path_to_file, 'r') as f:
        class_file_names = {sline[0] for line in f.readlines() \
                      if (sline := line.strip().split()) and len(sline) == 2 and sline[1] == '1'}

    # now, collect from these class file names only those whose annotations
    # show that they describe a single object only 
    for img_file in class_file_names:
        img_file_path = voc_jpeg_dir + img_file + '.jpg'
        annotation_file_path = voc_annotation_dir + img_file + '.xml' 

        # parse the .xml and check requirements
        with open(annotation_file_path, 'r') as xml_f:
            xmlstruct = eltree.parse(xml_f).getroot()

            # requirement: choose only files annotated with a single object
            objects = xmlstruct.findall('./object')
            if len(objects) != 1:
                continue 

            # requirement (sanity check): the annotation is consistent with class name
            img_object = objects[0]
            object_name = img_object.find('name').text
            if object_name != class_name:
                continue

            # requirement: there is only one bounding box
            bounding_box = img_object.findall('./bndbox') 
            if len(objects) != 1:
                continue 
            bounding_box = bounding_box[0]

            # read the bounding box details
            bbox_fields = ['xmin', 'ymin', 'xmax', 'ymax']
            bounding_box_pos = {field:int(bounding_box.find(field).text) for field in bbox_fields}

            file_name_to_bounding_box_dict[img_file_path] = bounding_box_pos

    # if requested, limit the number of images (sampling without replacement)
    if no_imgs is not None and len(file_name_to_bounding_box_dict) > no_imgs:
        random_img_keys = random.sample(list(file_name_to_bounding_box_dict.keys()), no_imgs)
        file_name_to_bounding_box_dict = {key:file_name_to_bounding_box_dict[key] for key in random_img_keys}

    return file_name_to_bounding_box_dict

if __name__ == '__main__':

    if len(sys.argv) < 4:
        print(f'Usage: {sys.argv[0]} voc_root_dir num_of_images class_name_1 class_name_2 ...')
        sys.exit(1)

    voc_root_dir = sys.argv[1] # the directory where VOCdevkit was unpacked
    img_number = int(sys.argv[2]) # the number of images to be pulled
    voc_classes = sys.argv[3:] # the names of the classes

    voc_class_dir = voc_root_dir + '/VOCdevkit/VOC2012/ImageSets/Main/'
    voc_annotation_dir = voc_root_dir + '/VOCdevkit/VOC2012/Annotations/'
    voc_jpeg_dir = voc_root_dir + '/VOCdevkit/VOC2012/JPEGImages/'

    print('**')
    print(f'** Collecting {img_number} images of the following classes: {voc_classes}')
    print(f'** assuming that VOC2012 directory is here: {voc_root_dir}')
    print('**')

    # check if directories with VOC2012 exists and contain requested classes
    sanitize_data_sources(voc_class_dir, voc_annotation_dir, voc_jpeg_dir, voc_classes)

    # collect dictionaries from per class file paths to bounding boxes
    no_images_per_class = img_number//len(voc_classes)
    for curr_class in voc_classes:
        print(f'-- collecting image data for class {curr_class}')
        class_images = collect_dir_for_class(voc_class_dir, voc_annotation_dir, voc_jpeg_dir, \
                                             curr_class, no_images_per_class)
        print(f'-- found {len(class_images)} images that have a single object and bounding box')

