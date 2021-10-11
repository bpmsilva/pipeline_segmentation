"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Run -> python3 balloon.py train --dataset=/home/diego/Desktop/CNN/Python_Code/My_Code/R-CNN_Project/My_Mask-RCNN/images/ --weights=imagenet


------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py inference --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py inference --weights=last --video=<URL or path to file>
"""

import os
import sys
#import json #Biblioteca do código original para carregar as anotações. Como utilizamos CSV, uso a lib abaixo
import csv
import datetime
import numpy as np
from numpy.lib.polynomial import poly
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "pipeline"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + pipeline

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

############################################################
#  Functions - Caio
############################################################

#SIZE IS THE SAME FOR EVERY IMAGE
height = 360
width = 640 

def read_CsvAnnotations(pathFile):

    with open(pathFile, newline='') as file:
        reader = csv.reader(file)
        annotations = list(map(tuple, reader))

    return annotations

def getPolygonPoints(line):

    line.pop(0) #Retiro o primeiro elemento que representa o Frame a ser usado!!

    line = np.array(line)
    listOfPoints_y = []
    listOfPoints_x = []

    for i in range(0,2): #Calcula os pontos x e y das 2 linhas

        m = (int(line[3])-int(line[1]))/(int(line[2])-int(line[0]) +0.000001)
        b = int(line[1]) -m*int(line[0])

        if m == 0: #Caso onde o frame anotado é para ser ignorado (anotamos com linhas horizontais)
            Polygon = False
            return Polygon

        y = np.array([0,width])
        x = np.array([(y[0]-b)/m,(y[1]-b)/m])

        #Como a ordem dos pontos importa, na iteração 1 adiciono p1, p2. Já na iteração 2 adiciono p2, p1

        if not i:
            listOfPoints_x.append(int(x[0])) 
            listOfPoints_x.append(int(x[1]))

            listOfPoints_y.append(int(y[0]))
            listOfPoints_y.append(int(y[1]))

        else:
            listOfPoints_x.append(int(x[1])) 
            listOfPoints_x.append(int(x[0]))

            listOfPoints_y.append(int(y[1]))
            listOfPoints_y.append(int(y[0])) 
        
        line = line[4:]

    Polygon = [ [listOfPoints_y[0], listOfPoints_x[0]], [listOfPoints_y[1], listOfPoints_x[1]], [listOfPoints_y[2], listOfPoints_x[2]], [listOfPoints_y[3], listOfPoints_x[3]] ]
    return Polygon


############################################################
#  Dataset
############################################################

class BalloonDataset(utils.Dataset):

    def load_balloon(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        #Assigning size values as defined global 
        global height
        global width

        # Add classes. We have only one class to add.
        self.add_class("pipeline", 1, "pipeline") #Source, Class_ID, Class Name

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # CSV Database saves each annotation in the form:
        # FRAME | X1 | Y1 | X2 | Y2 | X3 | Y3 | X4 | Y4 
        # We don't need to use frame number for mask generation, so we can discart this information
        # And we do it in getPolygonPoints function
        # We mostly care about the x and y coordinates of each pipeline

        subdirectories = os.listdir(path=dataset_dir) #Get a list of subdirectories to go and read annotation

        for vid in subdirectories: #Vid is the name of CSVs files, wich are the name of the folders too!!
            
            currentDirectory = os.path.join(dataset_dir, vid) #Join the root directory with folder name in use
            annotationFile = currentDirectory + "/" + vid + ".csv" #Add the rest of path to file
            
            annotations = read_CsvAnnotations(annotationFile) #Recebo na variável annotations uma lista que contem todo o CSV lido

            frameId = int(annotations[0][0]) #get first frame from that video

            for frame in annotations[:]: #Loop para percorrer as linhas do CSV file
                #Here frame meaning is the line of CSV file
                #Add images
                # Get the x, y coordinaets of points of the polygons that make up
                # the outline of each object instance. These are stores in lines of
                # CSVs files / Format-> FRAME, XI1, YI1, XF1, YF1, XI2, YI2, XF2, YF2
                Polygon = getPolygonPoints(list(frame)) 

                if Polygon == False: #We simple need to discart this frame. Annotation won't be relevant
                    frameId += 4
                    continue

                imageName = "Image" + str(frameId) + "_" + vid

                image_path = currentDirectory + "/" + imageName + ".png"
                #image = skimage.io.imread(image_path) DONT NEED TO READ THE IMAGE CAUSE ALREADY KNOW SIZE
                #Image Size if defined below in Functions Caio Section. Since it is the same for every image
                #we don't need to change it

                self.add_image(
                    "pipeline",
                    image_id=imageName, #We will use file name as a unique image id. So every image must have a unique name
                    path=image_path,
                    width=width, height=height,
                    polygons=Polygon) #Polygon[0] is where all y points is. Polygon[1] have x

                frameId += 4

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a pipeline dataset image, delegate to parent class.
        image_info = self.image_info[image_id]

        if image_info["source"] != "pipeline":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]

        mask = np.zeros([info["height"], info["width"], 3],dtype=np.uint8) #Dim: Height, Width, 3 (RGB - 3 color channels)
        
        #Get indexes of pixels inside the polygon and set them to 1
        calculatedMask = skimage.draw.polygon2mask(mask.shape, info["polygons"])
        
        mask[calculatedMask] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "pipeline":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BalloonDataset()
    dataset_val.load_balloon(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detectPipeline(model, image_path=None, video_path=None):
    
    assert image_path or video_path

    # Image or video?
    if image_path:
        
        # Run model detection and generate the image detection
        print("Running on {}".format(args.image))
        
        # Read image
        image = skimage.io.imread(args.image)
        
        # Detect objects
        r = model.detect([image], verbose=1)

        ### Color Splash Not Interesting for me #########

        # Color splash
        #splash = color_splash(image, r['masks'])
        
        #################################################
    
    elif video_path:
        
        import cv2
        
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                
                ### Color Splash Not Interesting for me

                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1

                #not sure where comment here because this is a video. i need to see this latter
                #################################################

        vwriter.release()
    
    #print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'inference'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "inference":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BalloonConfig()
    else:
        class InferenceConfig(BalloonConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "inference":
        detectPipeline(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'inference'".format(args.command))
