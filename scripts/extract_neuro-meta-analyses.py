""" Extract details from neuro-meta-analyses datasets such as inclusion/exclusion reasons """
import sys
import os
sys.path.append('../')
import json
from publang.extract import extract_from_text
from openai import OpenAI
from nipub_templates.neuro_meta_analyses.prompts import META_ANALYSIS_INCLUSION_CRITERIA_PROMPT
from pathlib import Path
import pandas as pd


# Read JSON lines
INPUT_DIR = Path('../../labelbuddy-annotations/projects/neuro-meta-analyses/documents')

# Load all batches
docs = []
for batch_file in INPUT_DIR.glob('*.jsonl'):
    with open(batch_file, 'r') as f:
        for line in f:
            docs.append(json.loads(line))

# Convert to DataFrame
docs = pd.DataFrame(docs)

OUTPUT_DIR = Path('../outputs/neuro_meta_analyses')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set up OpenAI clients
openai_client = OpenAI(api_key=os.getenv('MYOPENAI_API_KEY'))


def _run(extraction_model, extraction_client, docs, prepend='', **extract_kwargs):
    # Extract
    predictions = extract_from_text(
        docs['text'].to_list(),
        model=extraction_model, client=extraction_client,
        **extract_kwargs
    )

    # Add abstract id to predictions
    outputs = []
    for i, pred in enumerate(predictions):
        metadata = docs.iloc[i]['metadata']
        if pred:    
            if 'pmcid' in metadata:
                pred['pmcid'] = metadata['pmcid']
            if 'doi' in metadata:
                pred['doi'] = metadata['doi']
            if 'pmid' in metadata:
                pred['pmid'] = metadata['pmid']
            outputs.append(pred)

    name = f"full_{prepend}_{extraction_model.split('/')[-1]}"
    predictions_path = OUTPUT_DIR / f'{name}.json'

    json.dump(outputs, open(predictions_path, 'w'))


models = [
    ("gpt-5-mini", openai_client, {'temperature': 1}),
]

for model_name, client, kwargs in models:
    _run(model_name, client, docs, prepend='study-criteria',
         num_workers=10, **kwargs, **META_ANALYSIS_INCLUSION_CRITERIA_PROMPT
         )
