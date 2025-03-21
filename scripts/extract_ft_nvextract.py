import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
                 

def _clean_json_text(text):
    text = text.strip()
    text = text.replace("\#", "#").replace("\&", "&")
    return text

def _clean_output(output):
    output = output.split("<|output|>")
    if len(output) == 1:
        output = {}
    else:
        output = _clean_json_text(output[1])

    return output
    

def _tokenize_and_predict(prompt, model, tokenizer, max_length, max_new_tokens):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to('cuda')
    return tokenizer.decode(model.generate(**input_ids, max_new_tokens=max_new_tokens)[0], skip_special_tokens=True)

def _predict_chunk(text, template, current, model, tokenizer, max_length, max_new_tokens):
    current = _clean_json_text(current)

    input_llm =  f"<|input|>\n### Template:\n{template}\n### Current:\n{current}\n### Text:\n{text}\n\n<|output|>" + "{"
    output = _tokenize_and_predict(input_llm, model, tokenizer, max_length, max_new_tokens)

    return _clean_output(output)

def split_document(document, window_size, overlap, tokenizer):
    tokens = tokenizer.tokenize(document)
    print(f"\tLength of document: {len(tokens)} tokens")

    chunks = []
    if len(tokens) > window_size:
        for i in range(0, len(tokens), window_size-overlap):
            print(f"\t{i} to {i + len(tokens[i:i + window_size])}")
            chunk = tokenizer.convert_tokens_to_string(tokens[i:i + window_size])
            chunks.append(chunk)

            if i + len(tokens[i:i + window_size]) >= len(tokens):
                break
    else:
        chunks.append(document)
    print(f"\tSplit into {len(chunks)} chunks")

    return chunks

def handle_broken_output(pred, prev):
    try:
        if all([(v in ["", []]) for v in json.loads(pred).values()]):
            # if empty json, return previous
            pred = prev
    except:
        # if broken json, return previous
        pred = prev

    return pred

def sliding_window_prediction(text, template, model, tokenizer, window_size=2900, overlap=128, max_length=20_000, max_new_tokens=6000):
    chunks = split_document(text, window_size, overlap, tokenizer)

    # iterate over text chunks
    prev = template
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i}...")
        pred = _predict_chunk(chunk, template, prev, model, tokenizer, max_length, max_new_tokens)

        # handle broken output
        pred = handle_broken_output(pred, prev)
            
        # iterate
        prev = pred

    return pred


def full_text_prediction(text, template, model, tokenizer, max_length=20_000, max_new_tokens=6000):
    prompt = f"""<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>"""
    output =  _tokenize_and_predict(prompt, model, tokenizer, max_length, max_new_tokens)
    return _clean_output(output)


    
def predict_NuExtract(model, tokenizer, texts, template, mode, **kwargs):
    template = json.dumps(template, indent=4)
    
    outputs = []
    with torch.no_grad():
        for text in texts:
            if mode == 'full_text':
                output = full_text_prediction(text, template, model, tokenizer, **kwargs)
            elif mode == 'sliding_window':
                output = sliding_window_prediction(text, template, model, tokenizer, **kwargs)

            outputs.append(output)

    results = []
    for output in outputs:
        try:
            results.append(json.loads(output))
        except:
            results.append({})
            print(f"Error parsing")
    return results


def run_NuExtract(
    texts: list,
    schema: str,
    model_name: str = "numind/NuExtract-v1.5",
    mode = "sliding_window",
    **kwargs
):

    template = schema_to_template(schema.model_json_schema(), version="v1")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    return predict_NuExtract(model, tokenizer, texts, template, mode, **kwargs)


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
        "numind/NuExtract-v1.5", docs, StudyMetadataModel, 
        prepend='lb_nv_taskstructured-zeroshot'
    )
