##General utilities
import datetime as dt
import os
from uuid import uuid4
import os, os.path
import cv2
import random
import time
from multiprocessing.pool import ThreadPool
import json
import requests

##Labelbox utilities
import labelbox as lb
from labelbox import Project, Dataset

##Facebook Detectron2 utilities
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from matplotlib import pyplot as plt


import valohai

inputs = {
    "model": "datum://production-model"
}
valohai.prepare(step="autolabel", image="drazend/labelbox", environment="aws-eu-west-1-p3-2xlarge", default_inputs=inputs)

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


# # Object detection example
PROJECT_ID=os.getenv("LB_PROJECT_ID") #labelbox project id
## Generate API key: https://app.labelbox.com/account/api-keys
LB_API_KEY = os.getenv("LB_API_KEY")

client = lb.Client(LB_API_KEY, "https://api.labelbox.com/graphql")

## Get labelbox project
project = client.get_project(PROJECT_ID)

ontology, thing_classes = get_ontology(PROJECT_ID)
labels = json.loads(get_labels(PROJECT_ID))

# These can be set to anything. As long as this process doesn't have 
# another dataset with these names
train_ds_name = "custom_coco_train"
test_ds_name = "custom_coco_test"

DETECTRON_DATASET_TRAINING_NAME = 'prelabeling-train'
DETECTRON_DATASET_VALIDATION_NAME = 'prelabeling-val'
NUM_CPU_THREADS = 8 # for multiprocess downloads
DATA_LOCATION = './obj-data/'
DATASETS=[client.get_datasets(where=(Dataset.name == "Geospatial vessel detection"))] #labelbox dataset ids attached to the project
MODE = 'object-detection' 

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

all_import_requests = get_current_import_requests()

for task in all_import_requests:
    response = delete_import_request(task['id'])
    print(response)

## Get datarows that needs to be pre-labeled. We are performing a subtraction (all datarows in project - labeled datarows)
datarow_ids_with_labels = []

for label in labels:
    datarow_ids_with_labels.append(label['DataRow ID'])
    
all_datarow_ids = []
all_datarows = []

dataset = next(client.get_datasets(where=(Dataset.name == "Geospatial vessel detection")))

for data_row in dataset.data_rows():
    all_datarow_ids.append(data_row.uid)
    all_datarows.append(data_row)

datarow_ids_queued = diff_lists(all_datarow_ids, datarow_ids_with_labels)

print('Number of datarows to be pre-labeled: ', len(datarow_ids_queued))

## Download queued datarows that needs to be pre-labeled

data_row_queued = []
data_row_queued_urls = []

for datarow in all_datarows:
    for datarow_id in datarow_ids_queued:
        if datarow.uid == datarow_id:
            data_row_queued.append(datarow)
            extension = os.path.splitext(datarow.external_id)[1]
            filename = datarow.uid + extension
            data_row_queued_urls.append((DATA_LOCATION+'inference/' + filename, datarow.row_data))

print('Downloading queued data for inferencing...\n')
filepath_inference = ThreadPool(NUM_CPU_THREADS).imap_unordered(download_files, data_row_queued_urls)
print('Success...\n')

## Inferencing on queued datarows and create labelbox annotation import file (https://labelbox.com/docs/automation/model-assisted-labeling)

predictions = []
counter = 1

print("Inferencing...\n")
time.sleep(1)

for datarow in data_row_queued:
    extension = os.path.splitext(datarow.external_id)[1]
    filename = DATA_LOCATION+'inference/' + datarow.uid + extension
    print(filename)
    print(datarow)
    im = cv2.imread(filename)
    
    ##Predict using FB Detectron2 predictor
    outputs = predictor(im)
    
    categories = outputs["instances"].to("cpu").pred_classes.numpy()
    predicted_boxes = outputs["instances"].to("cpu").pred_boxes

    if len(categories) != 0:
        for i in range(len(categories)):
            
            classname = thing_classes[categories[i]]
            
            for item in ontology:
                if classname==item['name']:
                    schema_id = item['featureSchemaId']
                
            bbox = predicted_boxes[i].tensor.numpy()[0]
            bbox_dimensions = {'left': int(bbox[0]), 'top': int(bbox[1]), 'width': int(bbox[2]-bbox[0]), 'height': int(bbox[3]-bbox[1])}
            predictions.append({"uuid": str(uuid4()),'schemaId': schema_id, 'bbox': bbox_dimensions, 'dataRow': { 'id': datarow.uid }})
    
    v = Visualizer(im[:, :, ::-1],
                metadata=metadata, 
                scale=2, 
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    ## For paperspace cloud notebooks. Cloud notebooks do not support cv2.imshow.
    plt.rcParams['figure.figsize'] = (12, 24)
    plt.imshow(v.get_image()[:, :, ::-1])
    cv2.imwrite(valohai.outputs().path(filename), v.get_image()[:, :, ::-1])

    counter = counter + 1
          
time.sleep(1)
print('Total annotations predicted: ', len(predictions))