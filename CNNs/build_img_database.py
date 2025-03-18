#!/usr/bin/env python3

import sys
import os
import shutil
import random
import json
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
# - it then returns a list with dicts that contain details about images (filename, 
#   class, bounding box)
# - (if no_imgs is not None, then it tries to return a draw of no_imgs from the dict)
def collect_dir_for_class(voc_class_dir, voc_annotation_dir, voc_jpeg_dir,\
                            class_name, no_imgs = None):

    class_annotations = []
 
    # read class spec. file and collect the names of image files labeled with '1' 
    # (these contain easily recognizable objects of the class)
    path_to_file = voc_class_dir + class_name + '_trainval.txt'
    with open(path_to_file, 'r') as f:
        class_file_names = {sline[0] for line in f.readlines() \
                      if (sline := line.strip().split()) and len(sline) == 2 \
                        and sline[1] == '1'}

    # now, collect from these class file names only those whose annotations
    # show that they describe a single object only 
    for img_file in class_file_names:
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

            img_description_dict = {'file': img_file + '.jpg', 'class': class_name}

            # requirement: there is only one bounding box
            bounding_box = img_object.findall('./bndbox') 
            if len(objects) != 1:
                continue 
            bounding_box = bounding_box[0]

            # read the bounding box details
            bbox_fields = ['xmin', 'ymin', 'xmax', 'ymax']
            bounding_box_pos = {field:int(bounding_box.find(field).text) \
                                for field in bbox_fields}

            img_description_dict.update(bounding_box_pos)
            class_annotations.append(img_description_dict)

    # if requested, limit the number of images (sampling without replacement)
    if no_imgs is not None and len(class_annotations) > no_imgs:
        class_annotations = random.sample(class_annotations, no_imgs)

    return class_annotations

# This function will copy all images from class_annotations to target_dir and save
# class_annotations as .json metadata about the files
def copy_selected_images_and_save_metadata(class_annotations, voc_jpeg_dir, target_dir):

    # make target_dir and clean it if exists
    try:
        os.makedirs(target_dir)
    except:
        shutil.rmtree(target_dir)
        os.makedirs(target_dir)
    
    copy_ctr = 0
    # copy files from the original dataset to the new one
    for image_details in class_annotations:
        image_file = image_details['file']
        old_path = voc_jpeg_dir + '/' + image_file 
        new_path = target_dir + '/' + image_file 
        shutil.copyfile(old_path, new_path)
        copy_ctr += 1

    print(f'-- {copy_ctr} files copied')
    # save class_annotations as .json file
    metadata_filename = 'metadata.json'
    with open(metadata_filename, 'w') as jsonf:
        json.dump(class_annotations, jsonf)
        print(f'-- saved images metadata (class to filenames to bounding boxes) as {metadata_filename}')

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

    # this dict will map class name to a dict that assigns image files to bounding boxes 
    class_annotations = {}  

    # collect dictionaries from per class file paths to bounding boxes
    no_images_per_class = img_number//len(voc_classes)
    class_annotations = []
    for curr_class in voc_classes:
        print(f'-- collecting image data for class {curr_class},', end=' ')
        class_images = collect_dir_for_class(voc_class_dir, voc_annotation_dir, voc_jpeg_dir, \
                                             curr_class, no_images_per_class)
        print(f'found {len(class_images)} images that have a single object and bounding box')

        class_annotations.extend(class_images)

    # copy all the images selected in the previous step, together with .json metadata
    # to 'images' directory, for further processing
    copy_selected_images_and_save_metadata(class_annotations, voc_jpeg_dir, 'images')

    print('** all done')
