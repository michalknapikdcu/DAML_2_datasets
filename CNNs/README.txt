** The original source of images

The images collected here originate from The PASCAL VOC database. 
The project's website is here: http://host.robots.ox.ac.uk/pascal/VOC/
Selected images have been copied from the project's repository together
with annotations, archived, and pushed to Github in order to fulfill
the assignment's requirements.

** On image annotations

PASCAL VOC database provides for each image an .xml file that contains 
various annotations. The structure of these is rather self-explanatory,
moreover, within this assignment we're interested only in the following 
parts of the <object> sections:

<object>
[the name of the object is put here]
</object>

...
<bndbox>
<xmax> [the location of the max x of the object's bounding box] </xmax>
<xmin> [the location of the min x of the object's bounding box] </xmin>
<ymax> [the location of the max y of the object's bounding box] </ymax>
<ymax> [the location of the min y of the object's bounding box] </ymax>
</bndbox>
...
</object>

** On the original train/validate splits

The original database contains .txt files that split files into classes
and train/validate sets. For instance, the 'aeroplane' class consists of 
images that contain aeroplanes. There are three files provided for 
'aeroplane' class:

- aeroplane_train.txt: images selected for training
- aeroplane_val.txt: images selected for validation 
- aeroplane_trainval.txt: the union of the above

Entries in these files are further labeled where -1,0,1 means that the 
given object (here an aeroplane) is respectively: not present, present 
but possibly hard to detect, present. For instance the following means 
that only 2008_005796.jpg contains a picture of easily recognizable 
airplane: 

file: aeroplane_trainval.txt
...
2008_005779  0
2008_005794 -1
2008_005796  1
... 

** On image selection for this assignment project 

In order to make training and analysis feasible, we select 2000 images
from 5 classes, equally represented. We also ensure that each image 
selected does have a matching object entry with a bounding box. 
Moreover, we are ensuring that the selected images contain only a single
object. The latter is a limitation due to time constraints - it would
probably be possible to directly extend what is presented in this
solution to a multi-box object detection task. 

The process is fully automated. A provided python script iterates 
*_trainval.txt files that correspond to provided class names,
and randomly selects image files whose annotations satisfy our requirements.

These files are then copied to local images directory. The script 
will also create a json file with the following entries copied from the
original .xml annotations:

{
"filename": [the name of the image file],
"object": [the name of the object],
"bndbox": {
    "xmax": [the location of the max x of the object's bounding box],
    "xmin": [the location of the min x of the object's bounding box],
    "ymax": [the location of the max y of the object's bounding box],
    "ymax": [the location of the min y of the object's bounding box]
    }
}

** The script's usage

Assuming that we wish to obtain 2000 images selected from classes 
aeroplane person dog cat bird we only need to run:

./build_img_database.py [path to unpacked VOCdevkit] 2000 aeroplane person dog cat bird

The output is as follows:

**
** Collecting 2000 images of the following classes: ['aeroplane', 'person', 'dog', 'cat', 'bird']
** assuming that VOC2012 directory is here: ../..
**
-- collecting image data for class aeroplane, found 400 images that have a single object and bounding box
-- collecting image data for class person, found 400 images that have a single object and bounding box
-- collecting image data for class dog, found 400 images that have a single object and bounding box
-- collecting image data for class cat, found 400 images that have a single object and bounding box
-- collecting image data for class bird, found 400 images that have a single object and bounding box
-- copying files from aeroplane to images, done
-- copying files from person to images, done
-- copying files from dog to images, done
-- copying files from cat to images, done
-- copying files from bird to images, done
-- saved images metadata (class to filenames to bounding boxes) as metadata.json
** all done

(Note that we selected the classes in such a way that they are balanced!)

Now, a directory 'images' will contain the images randomly drawn from PASCAL VOC and file
'metadata.json' will contain entries that describe bounding boxes for each image, of each
selected class:

{
  "aeroplane": {
    "2011_001624.jpg": {
      "xmin": 115,
      "ymin": 148,
      "xmax": 353,
      "ymax": 223
    },
    "2010_006032.jpg": {
      "xmin": 1,
      "ymin": 1,
      "xmax": 500,
      "ymax": 334
    },
...

