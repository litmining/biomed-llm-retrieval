# The goal is to convert the "None of the above / Other" category into newly defined categories from Gemini

import json
import pandas as pd
from skllm.config import SKLLMConfig
from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier
from pathlib import Path

results_df = pd.read_csv('pd_normalized.csv')

gemini_diagnoses = json.load(open('onvoc-gemini-suggestions.json', 'r'))

categories_with_examples = [item.strip() for sublist in [[f"{k}:{v}" for k,v in a.items()]  for a in gemini_diagnoses.values()] for item in sublist]
categories_with_examples += ['None of the above / Other']

open_ai_key = Path('~/.keys/open_ai.key').expanduser()
open_ai_org = Path('~/.keys/open_ai_org.key').expanduser()

# Configure the credentials
SKLLMConfig.set_openai_key(open_ai_key.open().read().strip())
SKLLMConfig.set_openai_org(open_ai_org.open().read().strip())

unique_not_mapped = results_df[results_df.onvoc_diagnosis == 'None of the above / Other'].diagnosis_resolved.unique()

clf = ZeroShotGPTClassifier(model="gpt-4o-mini")
clf.fit([], categories_with_examples)
mapped = clf.predict(unique_not_mapped)

mappings = dict(zip(unique_not_mapped, mapped))

# Save
with open('onvoc-gemini-mappings.json', 'w') as f:
    json.dump(mappings, f, indent=2)


results_df['onvoc_diagnosis_gemini+'] = results_df['diagnosis_resolved'].map(mappings)

results_df.to_csv('pd_normalized.csv', index=False)