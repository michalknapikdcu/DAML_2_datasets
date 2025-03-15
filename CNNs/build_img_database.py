#!/usr/bin/env python3

import sys
import os

def sanitize_data_sources(voc_class_files, voc_annotation_files, voc_jpeg_files, voc_classes):

    data_dirs = [(voc_class_files, '.txt class'), (voc_annotation_files, 'annotations'), \
                 (voc_jpeg_files, 'images')]

    # check that directories exist 
    for dir, purpose in data_dirs:
        if not os.path.exists(dir):
            print(f'error: cannot find {purpose} directory under {dir}')
            sys.exit(1)

    # check that class .txt files are there
    for voc_class in voc_classes:
        voc_class_file = voc_class_files + voc_class + '_trainval.txt'

        if not os.path.exists(voc_class_file):
            print(f'error: cannot find {voc_class} file under {voc_class_file}')
            sys.exit(1)

if __name__ == '__main__':

    if len(sys.argv) < 4:
        print(f'Usage: {sys.argv[0]} voc_root_dir num_of_images class_name_1 class_name_2 ...')
        sys.exit(1)

    voc_root_dir = sys.argv[1] # the directory where VOCdevkit was unpacked
    img_number = int(sys.argv[2]) # the number of images to be pulled
    voc_classes = sys.argv[3:] # the names of the classes

    voc_class_files = voc_root_dir + '/VOCdevkit/VOC2012/ImageSets/Main/'
    voc_annotation_files = voc_root_dir + '/VOCdevkit/VOC2012/Annotations/'
    voc_jpeg_files = voc_root_dir + '/VOCdevkit/VOC2012/JPEGImages/'

    print('**')
    print(f'** Collecting {img_number} images of the following classes: {voc_classes}')
    print(f'** assuming that VOC2012 directory is here: {voc_root_dir}')
    print('**')

    # check if directories with VOC2012 exists and contain requested classes
    sanitize_data_sources(voc_class_files, voc_annotation_files, voc_jpeg_files, voc_classes)

