
unga bunga - v8 2024-10-08 6:38am
==============================

This dataset was exported via roboflow.com on October 8, 2024 at 6:39 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 8051 images.
Plate are annotated in YOLOv11 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Randomly crop between 0 and 24 percent of the image
* Random shear of between -10° to +10° horizontally and -10° to +10° vertically
* Random brigthness adjustment of between -23 and +23 percent
* Random Gaussian blur of between 0 and 2.5 pixels
* Salt and pepper noise was applied to 1.72 percent of pixels


