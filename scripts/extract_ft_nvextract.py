import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from pathlib import Path
import json
import pandas as pd

# Change directory for importing
import sys
sys.path.append('../')
from nipub_templates.nv_task.schemas import StudyMetadataModel
from nipub_templates.conversion import schema_to_template


# Read JSON lines
docs = pd.read_json('../../labelbuddy-annotations/projects/nv_task/documents/batch_0.jsonl', lines=True)

output_dir = Path('../outputs/nv_task/extractions')
output_dir.mkdir(parents=True, exist_ok=True)
                 
def predict_NuExtract(model, tokenizer, texts, template, batch_size=1, max_length=10_000, max_new_tokens=4_000):
    template = json.dumps(template, indent=4)
    prompts = [f"""<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>""" for text in texts]
    
    outputs = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_encodings = tokenizer(batch_prompts, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(model.device)

            pred_ids = model.generate(**batch_encodings, max_new_tokens=max_new_tokens)
            outputs += tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    results = []
    for output in outputs:
        try:
            results.append(json.loads(output))
        except:
            results.append({})
    return results

def run_NuExtract(
    texts: list,
    schema: str,
    model_name: str = "numind/NuExtract-tiny-v1.5",
    template_version: str = "v1",
    device = "cuda",
    batch_size: int = 1,
    max_length: int = 10_000,
    max_new_tokens: int = 4_000
):
    template = schema_to_template(schema.model_json_schema(), version=template_version)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return predict_NuExtract(model, tokenizer, texts, template, batch_size, max_length, max_new_tokens)


def _run(extraction_model, docs, schema, prepend='', **extract_kwargs):
    prepend += '_'
    short_model_name = extraction_model.split('/')[-1]
    
    pmcids = [d['pmcid'] for d in docs['metadata']]

    name = f"full_{prepend}{short_model_name}"
    predictions_path = output_dir / f'{name}.json'

    # Extract
    predictions = run_NuExtract(
        docs['text'].to_list(),
        schema, model_name=extraction_model,
        **extract_kwargs
    )

    # Add abstract id to predictions
    outputs = []
    for pred, _id in zip(predictions, pmcids):
        if pred:    
            pred['pmcid'] = _id
            outputs.append(pred)

    json.dump(outputs, open(predictions_path, 'w'))


if __name__ == '__main__':
    _run(
        "numind/NuExtract-tiny-v1.5", docs, StudyMetadataModel, 
        prepend='lb_nv_taskstructured-zeroshot',
        template_version="v1"
    )
