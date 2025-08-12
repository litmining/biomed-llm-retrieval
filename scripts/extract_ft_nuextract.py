import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
# Change directory for importing
import sys
sys.path.append('../')
from nipub_templates.nv_task.schemas import StudyMetadataModelNoMC
from nipub_templates.conversion import schema_to_template

# Read JSON lines
docs = pd.read_json('../../labelbuddy-annotations/projects/nv_task/documents/batch_0.jsonl', lines=True)

OUTPUT_DIR = Path('../outputs/nv_task/extractions')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                 

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
    

def nuextract_generate(prompt, model, tokenizer, max_length, max_new_tokens):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to('cuda')
    return tokenizer.decode(model.generate(**input_ids, max_new_tokens=max_new_tokens)[0], skip_special_tokens=True)

def _predict_chunk(text, template, current, model, tokenizer, max_length, max_new_tokens):
    current = _clean_json_text(current)

    input_llm =  f"<|input|>\n### Template:\n{template}\n### Current:\n{current}\n### Text:\n{text}\n\n<|output|>" + "{"
    output = nuextract_generate(input_llm, model, tokenizer, max_length, max_new_tokens)
    torch.cuda.empty_cache()

    return _clean_output(output)

def split_document(document, window_size, overlap, tokenizer):
    tokens = tokenizer.tokenize(document)
    print(f"\tLength of document: {len(tokens)} tokens")

    chunks = []
    if len(tokens) > window_size:
        for i in range(0, len(tokens), window_size-overlap):
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

def sliding_window_prediction(text, template, model, tokenizer, window_size=2000, overlap=128, max_length=10_000, max_new_tokens=4000):
    chunks = split_document(text, window_size, overlap, tokenizer)

    # iterate over text chunks
    prev = template
    for chunk in tqdm(chunks):
        pred = _predict_chunk(chunk, template, prev, model, tokenizer, max_length, max_new_tokens)

        # handle broken output
        pred = handle_broken_output(pred, prev)
            
        # iterate
        prev = pred

    return pred


def full_text_prediction(text, template, model, tokenizer, max_length=10_000, max_new_tokens=4000):
    prompt = f"""<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>"""
    output =  nuextract_generate(prompt, model, tokenizer, max_length, max_new_tokens)
    return _clean_output(output)

def load_model(model_name):
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def run_NuExtract(
    texts: list,
    schema: str,
    model_name: str = "numind/NuExtract-v1.5",
    mode = "sliding_window",
    ids = None,
    outfile = None,
    window_size = 2000,
    **kwargs
):
    template = schema_to_template(schema.model_json_schema(), version="v1")
    model, tokenizer = load_model(model_name)
    
    results = []
    with torch.no_grad():
        for i, text in tqdm(enumerate(texts)):
            try:
                if mode == 'full_text':
                    _func = full_text_prediction
                else:
                    _func = sliding_window_prediction
                    kwargs['window_size'] = window_size

                result = json.loads(
                    _func(
                        text, template, model, tokenizer, **kwargs
                    )
                )

                if ids:
                    result['id'] = ids[i]

                results.append(result)

                if outfile:
                    with open(outfile, 'w') as f:
                        json.dump(results, f)

            except Exception as e:
                print(f"Error: {e}")

    return results


def _run(extraction_model, docs, schema, prepend='', **extract_kwargs):
    short_model_name = extraction_model.split('/')[-1]
    
    # Add extract_kwargs to the output filename
    extract_kwargs_str = '_'.join([f"{k}={v}" for k, v in extract_kwargs.items()])
    
    outfile = OUTPUT_DIR / \
        f"full_{prepend}_{short_model_name}_{extract_kwargs_str}.jsonl"

    run_NuExtract(
        docs['text'].to_list(),
        schema, model_name=extraction_model,
        ids=[d['pmcid'] for d in docs['metadata']],
        outfile=outfile,
        **extract_kwargs
    )


if __name__ == '__main__':
    _run(
        "numind/NuExtract-v1.5", docs, StudyMetadataModelNoMC, 
        prepend='lb_nv_taskstructuredNOMC-zeroshot', window_size=5000,
    )
