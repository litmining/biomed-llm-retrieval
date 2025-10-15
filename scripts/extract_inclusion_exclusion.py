import pandas as pd
import os
from pathlib import Path
from publang.extract import extract_from_text
from openai import OpenAI
import json

# Change directory for importing
import sys
sys.path.append('../')
from nipub_templates.demographics.prompts import INCLUSION_EXCLUSION_CRITERIA

# Load input data
docs = pd.read_csv('/data/alejandro/projects/ns-pond/source/mega-ni-dataset/pubget_searches/original/fmri_journal/query_875641cf4cbc22f32027447cd62fca27/subset_allArticles_extractedData/text.csv')

docs = docs.head(1000)

# Set up output directory
output_dir = Path('../outputs/demographics/inclusion_exclusion')
output_dir.mkdir(parents=True, exist_ok=True)

# Set up OpenAI clients
openai_client = OpenAI(api_key=os.getenv('MYOPENAI_API_KEY'))


def run_extraction(extraction_model, extraction_client, docs, prepend='', **extract_kwargs):
    """Run full-text inclusion/exclusion criteria extraction."""
    prepend += '_' if prepend else ''
    short_model_name = extraction_model.split('/')[-1]

    # Remove search_query if present (not needed for full-text extraction)
    extract_kwargs.pop('search_query', None)

    name = f"full_{prepend}{short_model_name}"
    predictions_path = output_dir / f'{name}.json'
    results_path = output_dir / f'{name}.csv'

    # Extract from full text
    predictions = extract_from_text(
        docs['body'].to_list(),
        model=extraction_model, 
        client=extraction_client,
        **extract_kwargs
    )

    # Add pmcid to predictions
    outputs = []
    for pred, _id in zip(predictions, docs['pmcid']):
        if pred:
            pred['pmcid'] = _id
            outputs.append(pred)

    # Save JSON
    json.dump(outputs, open(predictions_path, 'w'), indent=2)

    # Convert to DataFrame and save CSV
    results = []
    for pred in outputs:
        result = {
            'pmcid': pred.get('pmcid'),
            'inclusion_criteria': pred.get('inclusion_criteria'),
            'exclusion_criteria': pred.get('exclusion_criteria'),
            'has_dedicated_section': pred.get('has_dedicated_section'),
            'criteria_location': pred.get('criteria_location')
        }
        results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv(results_path, index=False)
    
    print(f"Extraction complete. Results saved to:")
    print(f"  JSON: {predictions_path}")
    print(f"  CSV: {results_path}")
    return df


# Define models to use for extraction
models = [
    ("gpt-4o-mini-2024-07-18", openai_client),
]


if __name__ == "__main__":
    # Run extraction with different models
    for model_name, client in models:
        print(f"\nRunning full-text extraction with {model_name}...")
        run_extraction(
            model_name, 
            client, 
            docs,
            prepend='inclusion_exclusion',
            **INCLUSION_EXCLUSION_CRITERIA, 
            num_workers=20
        )