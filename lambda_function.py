import json
import urllib.parse
import boto3
import requests
import os

print('Loading function')

s3 = boto3.client('s3')

def lambda_handler(event, context):

    auth_token = os.environ["VH_TOKEN"]

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

    # Next we'll generate the JSON payload that we'll send to Valohai
    # See the Valohai API docs for more information
    # or click on the "Show as API call" button in the UI
    # You'll find the button next to the Create execution or Create pipeline buttons
    new_execution_json = {
      "project": "0184859e-885c-6258-d358-ca8438ded615",
      "commit": "~90523edc4f53ab6f1a316b0211b71855a957bcbfe25ef2bc6ee3f13d0e49c806",
      "step": "train-model",
      "inputs": {
        "images": [
          url_to_new_file
          ]
        }
    }

    # Send the JSON payload to the right Valohai API
    resp = requests.post('https://app.valohai.com/api/v0/executions/', headers=headers, json=new_execution_json)
    resp.raise_for_status()

    print(json.dumps(resp.json(), indent=4))