##General utilities
import datetime as dt
import os
from uuid import uuid4
import os, os.path
import cv2
import random
import time

##Labelbox utilities
import labelbox as lb

##Facebook Detectron2 utilities
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from matplotlib import pyplot as plt


import valohai

inputs = {
    "images": [
        "s3://labelbox-sample/raw-images/146.jpg"
    ],
    "model": "datum://018489a3-b5f0-f833-bae5-b640e617cf4c"
}
valohai.prepare(step="inference", image="drazend/labelbox", environment="aws-eu-west-1-p2-xlarge", default_inputs=inputs)

## get project ontology from labelbox
def get_ontology(project_id):
    response = client.execute(
                """
                query getOntology (
                    $project_id : ID!){ 
                    project (where: { id: $project_id }) { 
                        ontology { 
                            normalized 
                        } 
                    }
                }
                """,
                {"project_id": project_id})
            
    ontology = response['project']['ontology']['normalized']['tools']

    ##Return list of tools and embed category id to be used to map classname during training and inference
    mapped_ontology = []
    thing_classes = []
    
    i=0
    for item in ontology:
#         if item['tool']=='superpixel' or item['tool']=='rectangle':
        item.update({'category': i})
        mapped_ontology.append(item)
        thing_classes.append(item['name'])
        i=i+1         

    return mapped_ontology, thing_classes


# # Object detection example
PROJECT_ID=os.getenv("LB_PROJECT_ID") #labelbox project id
## Generate API key: https://app.labelbox.com/account/api-keys
LB_API_KEY = os.getenv("LB_API_KEY")

client = lb.Client(LB_API_KEY, "https://api.labelbox.com/graphql")

## Get labelbox project
project = client.get_project(PROJECT_ID)

ontology, thing_classes = get_ontology(PROJECT_ID)

# These can be set to anything. As long as this process doesn't have 
# another dataset with these names
train_ds_name = "custom_coco_train"
test_ds_name = "custom_coco_test"

DETECTRON_DATASET_TRAINING_NAME = 'prelabeling-train'
DETECTRON_DATASET_VALIDATION_NAME = 'prelabeling-val'


try:
    MetadataCatalog.get(DETECTRON_DATASET_TRAINING_NAME).thing_classes=thing_classes
    MetadataCatalog.get(DETECTRON_DATASET_VALIDATION_NAME).thing_classes=thing_classes
except Exception as e:
    print(e)

metadata = MetadataCatalog.get(DETECTRON_DATASET_TRAINING_NAME)
model = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.DATASETS.TRAIN = (DETECTRON_DATASET_TRAINING_NAME,)
cfg.DATASETS.TEST = (DETECTRON_DATASET_VALIDATION_NAME,)     
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
cfg.MODEL.WEIGHTS = os.path.join(valohai.inputs("model").path())
# Create predictor
predictor = DefaultPredictor(cfg)

print(metadata)

# Let's perform inferencing on random samples and preview the predictions before proceeding.
for d in valohai.inputs("images").paths():    
    im = cv2.imread(d)
    outputs = predictor(im)
    categories = outputs["instances"].to("cpu").pred_classes.numpy()
    predicted_boxes = outputs["instances"].to("cpu").pred_boxes

    print(categories)
    print(outputs)

    if len(categories) != 0:
        for i in range(len(categories)):
            classname = thing_classes[categories[i]]
            for item in ontology:
                if classname==item['name']:
                    schema_id = item['featureSchemaId']

    v = Visualizer(im[:, :, ::-1],
                metadata=metadata, 
                scale=2, 
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    ## For paperspace cloud notebooks. Cloud notebooks do not support cv2.imshow.
    plt.rcParams['figure.figsize'] = (12, 24)
    plt.imshow(v.get_image()[:, :, ::-1])
    cv2.imwrite(valohai.outputs().path(os.path.basename(d)), v.get_image()[:, :, ::-1])