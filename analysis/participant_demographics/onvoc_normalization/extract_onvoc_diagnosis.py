import json
import pandas as pd
from skllm.config import SKLLMConfig
from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier
from pathlib import Path

results_df = pd.read_csv('pd_normalized.csv')

onvoc = json.load(open('onvoc-diseases.json', 'r'))

vals = [item for sublist in onvoc.values() for item in sublist]

open_ai_key = Path('~/.keys/open_ai.key').expanduser()
open_ai_org = Path('~/.keys/open_ai_org.key').expanduser()

# Configure the credentials
SKLLMConfig.set_openai_key(open_ai_key.open().read().strip())
SKLLMConfig.set_openai_org(open_ai_org.open().read().strip())

vals += ['Healthy Controls', 'None of the above / Other']
unique_diags = results_df[(results_df.group_name == 'patients') & (results_df.diagnosis_resolved != 'Healthy')].diagnosis_resolved.unique()

clf = ZeroShotGPTClassifier(model="gpt-4o-mini")
clf.fit([], vals)
normalized = clf.predict(unique_diags)

# Save
with open('../analysis/participant_demographics/onvoc-normalized.json', 'w') as f:
    json.dump(dict(zip(unique_diags, normalized)), f, indent=2)

from collections import defaultdict
mappings = defaultdict(list)

for k, v in normalized.items():
    mappings[v].append(k)
results_df['onvoc_diagnosis'] = results_df['diagnosis_resolved'].map(normalized)

results_df.to_csv('pd_normalized.csv', index=False)