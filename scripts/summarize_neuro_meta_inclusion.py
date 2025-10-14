# Use OpenAI GPT-5 to summarize inclusion / exclusion criteria across meta-analyses
# Uses JSON containing extracted metadata from neuro-meta-analyses
import sys
import os

import json
from pathlib import Path
from openai import OpenAI

# Set up OpenAI client
openai_client = OpenAI(api_key=os.getenv('MYOPENAI_API_KEY'))

# JSON file containing extracted metadata
INPUT_FILE = Path('../outputs/neuro_meta_analyses/gpt5_cognitive_domain_inclusion_exclusion.json')
OUTPUT_FILE = Path('../outputs/neuro_meta_analyses/summarized_inclusion_exclusion.json')

BASE_MESSAGE = """
You will be provided with a JSON file containing extracted metadata from neuro-meta-analyses.
Using those annotations, summarize for me the most common inclusion and exclusion criteria across the studies.

The goal is to build a common set of inclusion and exclusion criteria that can be used to filter studies in future analyses.

The JSON file is delimited with triple backticks.

````
${json_content}
````
"""


def summarize_inclusion_exclusion(json_content):
    messages = [
        {
            "role": "user",
            "content": BASE_MESSAGE.replace('${json_content}', json_content)
        }
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",  # Use the appropriate model
        messages=messages,
        temperature=0.5
    )

    return response.choices[0].message.content


def main():
    # Load the JSON file
    with open(INPUT_FILE, 'r') as f:
        json_content = f.read()

    # Summarize inclusion/exclusion criteria
    summary = summarize_inclusion_exclusion(json_content)

    # Save the summary to a new JSON file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({"summary": summary}, f, indent=4)
    print(f"Summary saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
