import labelbox
from labelbox.schema.ontology import OntologyBuilder, Tool
from labelbox.data.serialization import NDJsonConverter
from labelbox.schema.annotation_import import LabelImport
from labelbox.schema.media_type import MediaType
from labelbox.data.annotation_types import (
    Label,
    Point,
    LabelList,
    ImageData,
    Rectangle,
    ObjectAnnotation,
)
from labelbox.schema.data_row_metadata import (
    DataRowMetadataField,
)

import requests
import json
import os
import time
import datetime
import random

import valohai

valohai.prepare(step="generate-labeled-data", image="drazend/labelbox")

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

## Generate API key: https://app.labelbox.com/account/api-keys
LB_API_KEY = os.getenv("LB_API_KEY")

client = labelbox.Client(LB_API_KEY)

DATA_ROWS = "https://storage.googleapis.com/labelbox-datasets/VHR_geospatial/geospatial_datarows.json"
ANNOTATIONS = "https://storage.googleapis.com/labelbox-datasets/VHR_geospatial/geospatial_annotations.json"

download_files(("data_rows.json", DATA_ROWS))
download_files(("annotations.json", ANNOTATIONS))

with open('data_rows.json', 'r') as fp:
    data_rows = json.load(fp)

with open('annotations.json', 'r') as fp:
    annotations = json.load(fp)

dataset = client.create_dataset(name="Geospatial vessel detection")

# Here is an example of adding two metadata fields to your Data Rows: a "captureDateTime" field with datetime value, and a "tag" field with string value
metadata_ontology = client.get_data_row_metadata_ontology()
datetime_schema_id = metadata_ontology.reserved_by_name["captureDateTime"].uid
tag_schema_id = metadata_ontology.reserved_by_name["tag"].uid
tag_items = ["WorldView-1", "WorldView-2", "WorldView-3", "WorldView-4"]

for datarow in data_rows:
    dt = datetime.datetime.utcnow() + datetime.timedelta(days=random.random()*30) # this is random datetime value
    tag_item = random.choice(tag_items) # this is a random tag value
    
    metadata_fields = [
                       DataRowMetadataField(schema_id=datetime_schema_id, value=dt), 
                       DataRowMetadataField(schema_id=tag_schema_id, value=tag_item)
                       ]

    datarow["metadata_fields"] = metadata_fields

task = dataset.create_data_rows(data_rows)
task.wait_till_done()

datarow = next(dataset.data_rows())

ontology = OntologyBuilder()

for tool in annotations['categories']:
  print(tool['name'])
  ontology.add_tool(Tool(tool = Tool.Type.BBOX, name = tool['name']))

ontology = client.create_ontology("Vessel detection ontology", ontology.asdict())
project = client.create_project(name="Vessel detection", media_type=MediaType.Image)
project.setup_editor(ontology)
ontology_from_project = OntologyBuilder.from_project(project)

data_rows = [dr.uid for dr in list(dataset.export_data_rows())]

# Randomly select 200 Data Rows
sampled_data_rows = random.sample(data_rows, 200)

batch = project.create_batch(
  "Initial batch", # name of the batch
  sampled_data_rows, # list of Data Rows
  1 # priority between 1-5
)

queued_data_rows = project.export_queued_data_rows()
ground_truth_list = LabelList()

for datarow in queued_data_rows:
  annotations_list = []
  folder = datarow['externalId'].split("/")[0]
  id = datarow['externalId'].split("/")[1]
  if folder == "positive_image_set":
    for image in annotations['images']:
      if (image['file_name']==id):
        for annotation in annotations['annotations']:
          if annotation['image_id'] == image['id']:
            bbox = annotation['bbox']
            id = annotation['category_id'] - 1
            class_name = ontology_from_project.tools[id].name
            annotations_list.append(ObjectAnnotation(
                name = class_name,
                value = Rectangle(start = Point(x = bbox[0], y = bbox[1]), end = Point(x = bbox[2]+bbox[0], y = bbox[3]+bbox[1])),
            ))
  image = ImageData(uid = datarow['id'])
  ground_truth_list.append(Label(data = image, annotations = annotations_list))

ground_truth_ndjson = list(NDJsonConverter.serialize(ground_truth_list))

start_time = time.time()
## Upload annotations
upload_task = LabelImport.create_from_objects(client, project.uid, "geospatial-import-job-1", ground_truth_ndjson)
print(upload_task)

#Wait for upload to finish (Will take up to five minutes)
upload_task.wait_until_done()
print(upload_task.errors)
print("--- Finished in %s mins ---" % ((time.time() - start_time)/60))
