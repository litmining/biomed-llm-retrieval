""" Extract participant demographics from HTML files. """
import os
from publang.extract import extract_from_text
from openai import OpenAI
from pathlib import Path
import json
import pandas as pd

# Change directory for importing
import sys
sys.path.append('../')
from nipub_templates.nv_task.prompts import ZERO_SHOT_TASK


output_dir = Path('../outputs/nv_task/extractions')
output_dir.mkdir(parents=True, exist_ok=True)

# Set up OpenAI clients
openai_client = OpenAI(api_key=os.getenv('MYOPENAI_API_KEY'))


def _run(extraction_model, extraction_client, csv_path, prepend='', batch_size=1000, **extract_kwargs):
    """
    Read the CSV file in chunks of `batch_size`. For each chunk:
      - Run the extraction.
      - Save the results to a separate JSON file.
    This way, the entire file is never loaded into memory at once.
    """
    prepend += '_'
    short_model_name = extraction_model.split('/')[-1]
    name_prefix = f"full_{prepend}{short_model_name}"

    chunk_index = 0
    # Read CSV in chunks
    for chunk in pd.read_csv(csv_path, chunksize=batch_size):
        # Convert columns to lists for extraction
        pmcids = chunk['study_id'].tolist()
        texts = chunk['body'].tolist()

        # Extract
        predictions = extract_from_text(
            texts,
            model=extraction_model,
            client=extraction_client,
            **extract_kwargs
        )

        # Add abstract id to predictions
        outputs = []
        for pred, pmcid in zip(predictions, pmcids):
            if pred:
                pred['pmcid'] = pmcid
                outputs.append(pred)

        # Save this chunk's predictions to a JSON file
        predictions_path = output_dir / f"{name_prefix}_batch{chunk_index}.json"
        with open(predictions_path, 'w') as f_out:
            json.dump(outputs, f_out)

        chunk_index += 1


# Define your models
models = [
    ("gpt-4o-mini", openai_client, {"temperature": 1}),
    # ("anthropic/claude-3.5-sonnet", openrouter_client),
    # ("gpt-4o-2024-05-13", openai_client)
]

# Path to your CSV file
csv_path = '../data/pubmed_texts_combined_julio.csv'

# Run for each model
for model_name, client, kwargs in models:
    _run(
        extraction_model=model_name,
        extraction_client=client,
        csv_path=csv_path,
        prepend='combinedpm_taskstructured-zeroshot',
        batch_size=1000,
        num_workers=10,         # if supported by your extraction function
        **ZERO_SHOT_TASK,
        **kwargs
    )