import torch


import datetime as dt
import os
import sys
import random
import time
from itertools import cycle
from uuid import uuid4
import requests
from pprint import pprint
from multiprocessing.pool import ThreadPool
import os, os.path
import numpy as np
import cv2
from skimage import io
import simplejson as json
import random
from datetime import datetime
import time
import shutil
from matplotlib import pyplot as plt
from pycocotools import mask
from PIL import Image

##Labelbox utilities
import labelbox as lb
from labelbox import Project, Dataset
from labelbox.schema.bulk_import_request import BulkImportRequest
from labelbox.schema.enums import BulkImportRequestState

##Facebook Detectron2 utilities
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import HookBase

import valohai

class ValohaiHook(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()

    def after_step(self):
        with torch.no_grad():
            latest_metrics = self.trainer.storage.latest()
            try :
                print(json.dumps({
                    "loss_box_reg": latest_metrics["loss_box_reg"],
                    "loss_rpn_cls": latest_metrics["loss_rpn_cls"],
                    "total_loss": latest_metrics["total_loss"],
                }))
            except:
                pass

    def after_train(self):
        with torch.no_grad():
            with open(valohai.outputs().path("model-config.yaml"), "w") as f:
                f.write(self.cfg.dump())

parameters = {
    "datarow_id": "",
}
valohai.prepare(step="train", image="drazend/labelbox", default_parameters=parameters, environment="aws-eu-west-1-p2-xlarge")

## Generate API key: https://app.labelbox.com/account/api-keys
LB_API_KEY = os.getenv("LB_API_KEY")

### Uncomment the object detection example to use with object detection projects
client = lb.Client(LB_API_KEY, "https://api.labelbox.com/graphql")

## Get labelbox project
PROJECT_ID=next(client.get_projects(where=(Project.name == "Vessel detection"))).uid #labelbox project id

project = client.get_project(PROJECT_ID)

# # Object detection example
DATASETS=[client.get_datasets(where=(Dataset.name == "Geospatial vessel detection"))] #labelbox dataset ids attached to the project
MODE = 'object-detection' 
DATA_LOCATION = './obj-data/'

## Universal configuration
DOWNLOAD_IMAGES = True # Download data from labelbox. Set false for re-runs when data already exists locally
VALIDATION_RATIO = 0.2 # Validation data / training data ratio
NUM_CPU_THREADS = 8 # for multiprocess downloads
NUM_SAMPLE_LABELS = 0 # Use 0 to use all of the labeled training data from project. Otherwise specify number of labeled images to use. Use smaller number for faster iteration. 
PRELABELING_THRESHOLD = 0.6 # minimum model inference confidence threshold to be uploaded to labelbox
HEADLESS_MODE = False # Set True to skip previewing data or model results

DETECTRON_DATASET_TRAINING_NAME = 'prelabeling-train'
DETECTRON_DATASET_VALIDATION_NAME = 'prelabeling-val'


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

## Creates a new export request to get all labels from labelbox. 
def get_labels(project_id):
    should_poll = 1
    while(should_poll == 1):
        response = client.execute(
                    """
                    mutation export(
                    $project_id : ID!    
                    )
                    { 
                        exportLabels(data:{ projectId: $project_id }){ 
                            downloadUrl 
                            createdAt 
                            shouldPoll 
                        }
                    }
                    """,
                    {"project_id": project_id})
        
        if response['exportLabels']['shouldPoll'] == False:
            should_poll = 0
            url = response['exportLabels']['downloadUrl']
            headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}

            r = requests.get(url, headers=headers)
            
            print('Export generated')
            ## writing export to disc for easier debugging
            open('export.json', 'wb').write(r.content)
            return r.content
        else:
            print('Waiting for export generation. Will check back in 10 seconds.')    
            time.sleep(10)

    return response

## Get all previous predictions import (bulk import request). 
def get_current_import_requests():
    response = client.execute(
                    """
                    query get_all_import_requests(
                        $project_id : ID! 
                    ) {
                      bulkImportRequests(where: {projectId: $project_id}) {
                        id
                        name
                      }
                    }
                    """,
                    {"project_id": PROJECT_ID})
    
    return response['bulkImportRequests']

## Delete all current predictions in a project and dataset. We want to delete them and start fresh with predictions from the latest model iteration
def delete_import_request(import_request_id):
    response = client.execute(
                    """
                        mutation delete_import_request(
                            $import_request_id : ID! 
                        ){
                          deleteBulkImportRequest(where: {id: $import_request_id}) {
                            id
                            name
                          }
                        }
                    """,
                    {"import_request_id": import_request_id})
    
    return response

## function to return the difference between two lists. This is used to compute the queued datarows to be used for inference. 
def diff_lists(li1, li2): 
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2] 
    return li_dif 

## Generic data download function
def download_files(filemap):
    path, uri = filemap    
    ## Download data
    if not os.path.exists(path):
        r = requests.get(uri, stream=True)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in r:
                    f.write(chunk)
    return path

## Converts binary image mask into COCO RLE format
def rle_encode(mask_image):
    size = list(mask_image.shape)
    pixels = mask_image.flatten()
    
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask, 
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    
    rle = {'counts': runs.tolist(), 'size': size}
    return rle


def load_set(dir):
    with open(dir+"dataset.json") as json_file:
        dataset_dicts = json.loads(json_file)
    return dataset_dicts

def cv2_imshow(a, **kwargs):
#     a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    return plt.imshow(a, **kwargs)


## Convert and load labelbox labels into Detectron2 format
def load_detectron2_dataset(labels, ontology, thing_classes, dir):
    dataset_dicts = []
    i = 0
    total = len(labels)

    print("Num labels processing: " + str(total))
    time.sleep(1)
    
    for label in labels:
        
        try:
            record = {}
            filename = os.path.join(dir, label['External ID'])
            
            ##scikit needed to raise exception if unable to read the image
            _ = io.imread(filename)
            
            height, width = cv2.imread(filename).shape[:2]

            record["file_name"] = filename
            record["height"] = height
            record["width"] = width
            record["image_id"] = label['ID']

            objs = []

            for instance in label['Label']['objects']:
                category_id = thing_classes.index(instance['title'])
#                 print(category_id)
                if MODE == 'object-detection':
                    obj = {
                            "bbox": [instance['bbox']['left'], instance['bbox']['top'], instance['bbox']['width'], instance['bbox']['height']],
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "segmentation": [],
                            "category_id": category_id,
                        }
                    objs.append(obj)

                if MODE == 'segmentation-rle':
                    path = DATA_LOCATION+masks+'/'+label['External ID']
                    mask_URI = instance['instanceURI']
                    downloaded_path = download_files((path, mask_URI))
                    im = cv2.imread(downloaded_path,0)

                    binary = np.array(im)

                    rle = mask.encode(np.asfortranarray(binary))
                    ground_truth_bounding_box = mask.toBbox(rle)

                    obj = {
                            "bbox": ground_truth_bounding_box.tolist(),
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "segmentation": rle,
                            "category_id": category_id,
                            "iscrowd": 0
                        }
                    objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)
            

            i=i+1
        except Exception as e:
            print('Exception: ', e)

            
    ## Write detectron2 dataset file to disk for easier debugging
    f = open(dir+"dataset_dict.json","w")
    f.write(json.dumps(dataset_dicts))
    f.close()
    
    return dataset_dicts

class CocoTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)

start_time = time.time()

if os.path.exists('coco_eval'):
    shutil.rmtree('coco_eval')
    

##Get ontology
ontology, thing_classes = get_ontology(PROJECT_ID)
print('Available classes: ', thing_classes)

##Get labels
labels = json.loads(get_labels(PROJECT_ID))

##Split training and validation labels
if NUM_SAMPLE_LABELS !=0:
    val_sample = int(VALIDATION_RATIO*NUM_SAMPLE_LABELS)
    val_labels = random.sample(labels, val_sample)
    train_labels = random.sample(labels, NUM_SAMPLE_LABELS)
else:
    split = int(VALIDATION_RATIO*len(labels))
    val_labels = labels[:split]
    train_labels = labels[split:]

## Check and create folders for downloading data from Labelbox
train = 'train'
val = 'val'
inference = 'inference'
masks = 'masks'
tmp = 'tmp'

if not os.path.exists(DATA_LOCATION):
    os.makedirs(DATA_LOCATION)

if not os.path.exists(DATA_LOCATION+train):
    os.makedirs(DATA_LOCATION+train)
    os.makedirs(DATA_LOCATION+train+"/positive_image_set")
    
if not os.path.exists(DATA_LOCATION+val):
    os.makedirs(DATA_LOCATION+val)
    os.makedirs(DATA_LOCATION+val+"/positive_image_set")
    
if not os.path.exists(DATA_LOCATION+inference):
    os.makedirs(DATA_LOCATION+inference)
    os.makedirs(DATA_LOCATION+inference+"/positive_image_set")
    
if not os.path.exists(DATA_LOCATION+masks):
    os.makedirs(DATA_LOCATION+masks)
    os.makedirs(DATA_LOCATION+masks+"/positive_image_set")
    
if not os.path.exists(DATA_LOCATION+tmp):
    os.makedirs(DATA_LOCATION+tmp)
    os.makedirs(DATA_LOCATION+tmp+"/positive_image_set")
    
##Download training and validation labels in parallel
train_urls = []
for label in train_labels:
    train_urls.append((DATA_LOCATION+'train/' + label['External ID'], label['Labeled Data']))

val_urls = []
for label in val_labels:
    val_urls.append((DATA_LOCATION+'val/' + label['External ID'], label['Labeled Data']))


if(DOWNLOAD_IMAGES==True):
    print('Downloading training and validation data... \n')
    
    results_train = ThreadPool(NUM_CPU_THREADS).imap_unordered(download_files, train_urls)
    results_val = ThreadPool(NUM_CPU_THREADS).imap_unordered(download_files, val_urls)
    
    for item in results_train:
        pass
    for item in results_val:
        pass
    
    print('Finished downloading training and validation data... \n')
    

### Begin FB Detectron code. 

#Load dataset into Detectron2
try:
    DatasetCatalog.register(DETECTRON_DATASET_TRAINING_NAME, lambda: load_detectron2_dataset(train_labels, ontology, thing_classes, DATA_LOCATION+'train/' ))
    DatasetCatalog.register(DETECTRON_DATASET_VALIDATION_NAME, lambda: load_detectron2_dataset(val_labels, ontology, thing_classes, DATA_LOCATION+'val/' ))
    MetadataCatalog.get(DETECTRON_DATASET_TRAINING_NAME).thing_classes=thing_classes
    MetadataCatalog.get(DETECTRON_DATASET_VALIDATION_NAME).thing_classes=thing_classes
except Exception as e:
    print(e)
    
if MODE == 'object-detection':
    model = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
    
if MODE == 'segmentation-rle':
    model = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'


##Load data and metadata for visualization and inference
dataset_dicts = DatasetCatalog.get(DETECTRON_DATASET_TRAINING_NAME)
dataset_dicts_val = DatasetCatalog.get(DETECTRON_DATASET_VALIDATION_NAME)
metadata = MetadataCatalog.get(DETECTRON_DATASET_TRAINING_NAME)


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.DATASETS.TRAIN = (DETECTRON_DATASET_TRAINING_NAME,)
cfg.DATASETS.TEST = (DETECTRON_DATASET_VALIDATION_NAME,)   
cfg.TEST.EVAL_PERIOD = 150
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00125
cfg.SOLVER.MAX_ITER = 1500    # 300
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
cfg.OUTPUT_DIR = valohai.outputs().dir_path

vh_hook = ValohaiHook(cfg)

if MODE=='segmentation-rle':
    cfg.INPUT.MASK_FORMAT='bitmask'

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg) 

trainer.register_hooks([vh_hook])
trainer.resume_or_load(resume=False)
trainer.train()

valohai_file_tags = {
    "valohai.tags": ["model", "object-detection"],
    "valohai.alias": "production-model"
}

valohai_file_metadata_path = "model_final.pth.metadata.json"

with open(valohai.outputs().path(valohai_file_metadata_path), "w") as outfile:
    json.dump(valohai_file_tags, outfile)

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = PRELABELING_THRESHOLD  # set threshold for this model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

# Create predictor
predictor = DefaultPredictor(cfg)

for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    categories = outputs["instances"].to("cpu").pred_classes.numpy()
    predicted_boxes = outputs["instances"].to("cpu").pred_boxes

    print(categories)
    print(outputs)
    if MODE=='segmentation-rle':
        pred_masks = outputs["instances"].to("cpu").pred_masks.numpy()

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
    plt.savefig(valohai.outputs().path(d["file_name"]))
    cv2.imwrite(valohai.outputs().path("new_" + d["file_name"]), v.get_image()[:, :, ::-1])