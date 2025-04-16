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
docs = pd.read_json('../../labelbuddy-annotations/projects/nv_task/documents/batch_0.jsonl', lines=True)
# Limit to 2 documents for testing
docs = docs.iloc[:2]
# Get doc_ids from the metadata
doc_ids = [d['pmcid'] for d in docs['metadata']]

output_dir = Path('../outputs/nv_task/')
output_dir.mkdir(parents=True, exist_ok=True)

# Set up OpenAI clients
openai_client = OpenAI(api_key=os.getenv('MYOPENAI_API_KEY'))

evidence_schema = create_evidence_model(StudyMetadataModel).model_json_schema()


def prepare_input_args(
    doc_ids, previous_outputs_path
):
    """Prepare input arguments for the extraction process."""
    # Load previous outputs
    with open(previous_outputs_path) as f:
        previous_outputs = json.loads(f.read())
    previous_outputs = {d.pop('pmcid'): d for d in previous_outputs}

    # Prepare input arguments
    input_args = []
    for doc_id in doc_ids:
        if doc_id in previous_outputs:
            input_args.append(
                {
                    'prompt_substitutions': {
                        'previous_extraction_json': json.dumps(previous_outputs[doc_id])
                    }
                }
            )
        else:
            input_args.append(
                {
                    'prompt_substitutions': {
                        'previous_extraction_json': '{}'
                    }
                }
            )
    return input_args

# Prepare input arguments
previous_outputs_path = output_dir / 'extractions/full_lb_nv_taskstructured-zeroshot_gpt-4o-mini-2024-07-18.jsonl'
input_args = prepare_input_args(
    doc_ids, previous_outputs_path
)


def _run(extraction_model, extraction_client, docs, prepend='', **extract_kwargs):
    # Extract
    predictions = extract_from_text(
        docs['text'].to_list(),
        model=extraction_model, client=extraction_client,
        **extract_kwargs
    )

    name = f"full_{prepend}_{extraction_model.split('/')[-1]}"
    predictions_path = output_dir / 'evidence' / f'{name}.json'

    json.dump(predictions, open(predictions_path, 'w'))


models = [
    ("gpt-4o-mini-2024-07-18", openai_client)
    # ("anthropic/claude-3.5-sonnet", openrouter_client),
]

for model_name, client in models:
    _run(model_name, client, docs, prepend='nv_taskstructured-evidence',
         messages=MESSAGES, output_schema=evidence_schema,
         ids=doc_ids, num_workers=1, input_args=input_args)
