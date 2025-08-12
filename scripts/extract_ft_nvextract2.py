import torch
import json
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Change directory for importing
import sys
sys.path.append('../')
from nipub_templates.conversion import schema_to_template
from nipub_templates.nv_task.schemas import StudyMetadataModel, StudyMetadataModelNoMC
from nipub_templates.nv_task.examples import EXAMPLES, EXAMPLESNOMC


# Read JSON lines
docs = pd.read_json('../../labelbuddy-annotations/projects/nv_task/documents/batch_0.jsonl', lines=True)

OUTPUT_DIR = Path('../outputs/nv_task/extractions')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                 
    
def robust_json_decode(raw_string):
    """
    Attempt to decode a JSON string that may have extra wrapping quotes
    or excessive escaping. Returns the parsed JSON object on success,
    otherwise raises a JSONDecodeError.
    """
    # 1) Trim leading/trailing whitespace
    s = raw_string.strip()
    
    # 2) If the entire string is enclosed in a single set of quotes,
    #    remove them. (Covers cases like '"{...}"' or "'{...}'".)
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    
    # 3) Sometimes the string has embedded escape sequences for quotes.
    #    Replacing `\"` with just `"` is a common fix (careful not to
    #    over-correct if your data legitimately uses backslash).
    s = s.replace(r'\"', '"')
    s = s.replace(r"\'", "'")
    
    # 4) Now try parsing the potentially cleaned-up string as JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        pass

    # 5) If we still can't parse, try removing the last character
    try:
        return json.loads(s.strip()[:-1])
    except json.JSONDecodeError:
        pass

    # 6) If we still can't parse, try adding a trailing }
    try:
        return json.loads(s.strip() + "}")
    except json.JSONDecodeError:
        pass

    raise json.JSONDecodeError("Failed to parse JSON", s, 0)



def nuextract_generate(
        model, tokenizer, prompt, generation_config
        ):
    """
    Generate responses NuExtract prompt.
    
    Args:
        model: The vision-language model
        tokenizer: The tokenizer for the model
        prompt: Prompt
        generation_config: GenerationConfig object
        
    Returns:
        List of generated responses
    """    
    img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    model.img_context_token_id = img_context_token_id

    # process all prompts in a single batch
    tokenizer.padding_side = 'left'
    model_inputs = tokenizer(prompt, return_tensors='pt', padding=True)

    input_ids = model_inputs['input_ids'].to(model.device)
    attention_mask = model_inputs['attention_mask'].to(model.device)
        
    # generate outputs
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
    )[0]
    
    # Decode responses
    responses = tokenizer.decode(output, skip_special_tokens=True)
    
    return responses


def construct_prompt(text, template, tokenizer, examples=None, max_chars=None):
    """
    Construct the individual NuExtract message texts, prior to chat template formatting.
    """
    # check if text is too long by tokenizing
    if max_chars is not None and len(tokenizer(text)['input_ids']) > max_chars:
        print(f"Text is too long ({len(text)} chars), truncating to {max_chars} chars.")
        text = text[:max_chars]

    # add few-shot examples if needed
    if examples is not None and len(examples) > 0:
        icl = "# Examples:\n"
        for row in examples:
            icl += f"## Input:\n{row['input']}\n## Output:\n{row['output']}\n"
    else:
        icl = ""
        
    message =  f"""# Template:\n{template}\n{icl}# Context:\n{text}"""
    message =  {"role": "user", "content": message}

    # dump to JSON to debug
    with open('_prompt.json', 'w') as f:
        json.dump(message, f, indent=4)

    return tokenizer.apply_chat_template(
        [message],
        tokenize=False,
        add_generation_prompt=True
    )

def load_model(model_name):
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, 
                                                torch_dtype=torch.bfloat16,
                                                attn_implementation="flash_attention_2",
                                                device_map="auto").eval()

    return model, tokenizer


def run_NuExtract(
    texts: list,
    schema: str = None,
    template: str = None,
    model_name: str = "numind/NuExtract-2-2B",
    examples: list = None,
    ids = None,
    outfile = None,
    max_chars = None,
    **kwargs
):
    if schema:
        template = schema_to_template(schema.model_json_schema(), version="v2")

    model, tokenizer = load_model(model_name)

    generation_config = GenerationConfig(
        max_new_tokens=4000,
        **kwargs
    )

    print(f"Running NuExtract with model: {model_name}")
    print(f"kwargs: {kwargs}")

    results = []
    unparsed = []
    with torch.no_grad():
        for i, text in tqdm(enumerate(texts)):
            prompt = construct_prompt(text, template, tokenizer, examples, max_chars=max_chars)
            output = nuextract_generate(model, tokenizer, prompt, generation_config)

            try:
                result = robust_json_decode(output)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for text {i}")
                unparsed.append(output)
                continue

            if ids:
                result['id'] = ids[i]

            results.append(result)

            if outfile:
                with open(outfile, 'w') as f:
                    json.dump(results, f)

    if unparsed:
        # Append _unparsed to the output filename, and change extension to .txt
        outfile = outfile.with_name(outfile.stem + '_unparsed').with_suffix('.txt')
        with open(outfile, 'w') as f:
            f.write('\n'.join(unparsed))

    return results

def _run(extraction_model, docs, schema, prepend='', examples=None, **extract_kwargs):
    short_model_name = extraction_model.split('/')[-1]

    # Add extract_kwargs to the output filename
    extract_kwargs_str = '_'.join([f"{k}={v}" for k, v in extract_kwargs.items()])
    
    outfile = OUTPUT_DIR / \
        f"full_{prepend}_{short_model_name}_{extract_kwargs_str}.jsonl"

    run_NuExtract(
        docs['text'].to_list(),
        schema=schema, model_name=extraction_model,
        ids=[d['pmcid'] for d in docs['metadata']],
        outfile=outfile,
        examples=examples,
        **extract_kwargs
    )


if __name__ == '__main__':
    _run(
        "numind/NuExtract-2-4B", docs, StudyMetadataModel, 
        prepend='lb_nv_taskstructured-noexample-zeroshot',
        examples=None, temperature=0, do_sample=False
    )

    _run(
        "numind/NuExtract-2-4B", docs, StudyMetadataModelNoMC,
        prepend='lb_nv_taskstructuredNOMC-noexample-zeroshot',
        examples=None, temperature=0, do_sample=False
    )