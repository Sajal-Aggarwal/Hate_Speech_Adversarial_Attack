!pip install google-api-python-client

from googleapiclient import discovery
import json

API_KEY = 'ENTER YOUR API KEY'

client = discovery.build(
"commentanalyzer",
"vlalphal", 
developerKey=API_KEY,
discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest",
static_discovery=False,
)

analyze_request = {
'comment': { 'text': 'Sample comment to test toxicity using Perspective API' },
'requestedAttributes': {'TOXICITY': {}}
}

response = client.comments().analyze(body=analyze_request).execute()

print(json.dumps(response, indent=2))
