import json
import urllib.parse
import boto3
import requests
import os
 
print('Loading function')
 
s3 = boto3.client('s3')
 
def lambda_handler(event, context):
    # The lambda event will contain a list of files (Records)
    # We'll take the first file object and find the S3 bucket name
    bucket = event['Records'][0]['s3']['bucket']['name']
    # Then we'll extract the path to the file inside the bucket
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    # Generate a URL to the new file using the extracted data
    # e.g. s3://mybucket/images/animals/horse.jpeg
    url_to_new_file = f's3://{bucket}/{key}'
 
    # Get the Valohai API token you created earlier.
    # Here it has been stored in a local environment variable.
    # Remember to follow your organization's security standards when handling the token.
    auth_token = os.environ['VH_API_TOKEN']
    headers = {'Authorization': 'Token %s' % auth_token}
    
    new_pipeline_json = {
        "edges": [
            {
            "source_node": "process-new-data",
            "source_key": "datarow_id",
            "source_type": "metadata",
            "target_node": "train",
            "target_type": "parameter",
            "target_key": "datarow_id"
            }
        ],
        "nodes": [
            {
            "name": "process-new-data",
            "type": "execution",
            "template": {
                "commit": "~ddd739f6bc5515833660cb14fc8c5a9d10639fc2303fcdb5c1f808730156f345",
                "step": "process-new-data",
                "inputs": {
                "images": [
                    url_to_new_file
                ]
                },
            },
            "on_error": "stop-all"
            },
            {
            "name": "train",
            "type": "execution",
            "template": {
                "commit": "~ddd739f6bc5515833660cb14fc8c5a9d10639fc2303fcdb5c1f808730156f345",
                "step": "train",
                "parameters": {
                "datarow_id": ""
                },
            },
            "on_error": "stop-all",
            "actions": [
                {
                "if": [],
                "then": [
                    "require-approval"
                ],
                "when": [
                    "node-starting"
                ]
                }
            ]
            }
        ],
        "project": "0184859e-885c-6258-d358-ca8438ded615",
        "title": "New Data - Label & Train"
        }
    
    # Send the JSON payload to the right Valohai API
    resp = requests.post('https://app.valohai.com/api/v0/pipelines/', headers=headers, json=new_pipeline_json)
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=4))