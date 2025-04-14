import sys
import os
sys.path.append('../')
import json
from publang.extract import extract_from_text
from openai import OpenAI
from nipub_templates.nv_task.schemas import StudyMetadataModel
from nipub_templates.evidence.schemas import create_evidence_model
from nipub_templates.evidence.prompts import MESSAGES
from pathlib import Path
import pandas as pd


OUTPUT_DIR = Path('../outputs/nv_task/evidence')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read JSON lines
docs = pd.read_json('../../labelbuddy-annotations/projects/nv_task/documents/task_0.jsonl', lines=True)

output_dir = Path('../outputs/nv_task/extractions')
output_dir.mkdir(parents=True, exist_ok=True)

# Set up OpenAI clients
openai_client = OpenAI(api_key=os.getenv('MYOPENAI_API_KEY'))

StudyMetadataEvidenceModel = create_evidence_model(StudyMetadataModel)

previous_outputs = pd.read_json(output_dir / 'full_lb_nv_taskstructured-zeroshot_gpt-4o-mini-2024-07-18.jsonl', lines=True)
previous_outputs = {d.pop('pmcid'): d for d in previous_outputs}

doc_ids = [d['pmcid'] for d in docs['metadata']],

# Sort previous inputs by d['pmcid'] based on doc_ids
input_args = []
for doc_id in doc_ids:
    if doc_id in previous_outputs:
        input_args.append({'prompt_subsitutions': previous_outputs[doc_id]})
    else:
        input_args.append({'prompt_subsitutions': {'pmcid': doc_id, 'text': ''}})


def _run(extraction_model, extraction_client, docs, prepend='', **extract_kwargs):
    # Extract
    predictions = extract_from_text(
        docs['text'].to_list(),
        model=extraction_model, client=extraction_client,
        **extract_kwargs
    )

    # Add abstract id to predictions
    pmcids = [d['pmcid'] for d in docs['metadata']]
    outputs = []
    for pred, _id in zip(predictions, pmcids):
        if pred:    
            pred['pmcid'] = _id
            outputs.append(pred)

    name = f"full_{prepend}_{extraction_model.split('/')[-1]}"
    predictions_path = output_dir / f'{name}.json'

    json.dump(outputs, open(predictions_path, 'w'))


models = [
    ("gpt-4o-mini-2024-07-18", openai_client),
    # ("anthropic/claude-3.5-sonnet", openrouter_client),
]

for model_name, client, kwargs in models:
    _run(model_name, client, docs, prepend='nv_taskstructured-evidence',
         messages=MESSAGES, output_schema=StudyMetadataEvidenceModel.model_json_schema(),
         ids=doc_ids
         num_workers=10, input_args=input_args, **kwargs
         )
