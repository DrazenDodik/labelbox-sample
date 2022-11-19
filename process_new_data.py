import labelbox
from labelbox import Dataset
import os
from datetime import datetime
import json

import valohai

inputs = {
  "images": "s3://labelbox-sample/raw-images/146.jpg"
}

valohai.prepare(step="process-new-data", default_inputs=inputs, image="python:3.9")

LB_API_KEY = os.getenv("LB_API_KEY")
PROJECT_ID = os.getenv("LB_PROJECT_ID")

client = labelbox.Client(LB_API_KEY)

## Get labelbox project
project = client.get_project(PROJECT_ID)
dataset = next(client.get_datasets(where=(Dataset.name == "Geospatial vessel detection")))

data_rows = []
for path in valohai.inputs("images").paths():
      row = {
          "row_data": path,
          "external_id": os.path.basename(path)
      }
      print(row)
      data_rows.append(row)

task = dataset.create_data_rows(data_rows)
task.wait_till_done()

data_rows = [dr.uid for dr in list(dataset.export_data_rows())][-1:]
print("New data added")
print(json.dumps({"datarow_id": data_rows[0]}))
