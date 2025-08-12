import pandas as pd
import os
from pathlib import Path
from publang.extract import extract_from_text
from openai import OpenAI
import json


# Change directory for importing
import sys
sys.path.append('../')
from nipub_templates.pain import ZERO_SHOT_PAIN_DESIGN_FC
from nipub_templates.clean import clean_predictions_pain

# Load 10 rows
docs = pd.read_csv('/data/alejandro/projects/ns-pond/source/mega-ni-dataset/pubget_searches/fmri_journal/query_875641cf4cbc22f32027447cd62fca27/subset_allArticles_extractedData/text_subset.csv')
                   

output_dir = Path('../outputs/extractions/all')
openai_client = OpenAI(api_key=os.getenv('MYOPENAI_API_KEY'))

def _run(extraction_model, extraction_client, docs, prepend='', **extract_kwargs):
    prepend += '_'
    short_model_name = extraction_model.split('/')[-1]

    extract_kwargs.pop('search_query', None)

    name = f"full_{prepend}{short_model_name}"
    predictions_path = output_dir / f'{name}.json'
    clean_predictions_path = output_dir / f'{name}_clean.csv'

    # Extract
    predictions = extract_from_text(
        docs['body'].to_list(),
        model=extraction_model, client=extraction_client,
        **extract_kwargs
    )

    # Add abstract id to predictions
    outputs = []
    for pred, _id in zip(predictions, docs['pmcid']):
        if pred:    
            pred['pmcid'] = _id
            outputs.append(pred)

    json.dump(outputs, open(predictions_path, 'w'))

    clean_predictions_pain(predictions).to_csv(
        clean_predictions_path, index=False
    )


models = [
    ("gpt-4o-mini-2024-07-18", openai_client),
]


for model_name, client in models:
    _run(model_name, client, docs, prepend='md_paindesign-zeroshot',
         **ZERO_SHOT_PAIN_DESIGN_FC, num_workers=20)
