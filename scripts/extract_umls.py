# ==============================================================================
# UMLS Code Extraction Script
# ==============================================================================
#
# Purpose:
# This script processes participant demographic information previously extracted
# from research papers and saved into CSV files. Specifically, it focuses on
# diagnosis strings associated with 'patients' groups. For each diagnosis string,
# it utilizes the `scispacy.CandidateGenerator` (based on UMLS) to identify
# potential matching UMLS concepts (CUIs).
#
# Workflow:
# 1. Reads input CSV files containing demographic extractions (e.g., diagnosis text,
#    PMCID, participant count) located in '../outputs/demographicsextractions'.
#    It specifically targets files matching '*_*zeroshot*_noabbrev.csv'.
# 2. For each relevant diagnosis string, it generates UMLS candidates using scispaCy.
# 3. Filters these candidates based on similarity scores, with optional stricter
#    filtering for candidates lacking definitions in the UMLS knowledge base.
# 4. Saves the top N filtered UMLS predictions (CUI, name, score/probability)
#    to new output CSV files (named '*_*zeroshot*_umls.csv' in the same directory).
# 5. The output links the UMLS predictions back to the original PMCID, diagnosis text,
#    participant count, and original data row index (as 'group_ix').
#
# This script generates the UMLS prediction files that are later evaluated for
# accuracy in the corresponding analysis notebook.
#
# ==============================================================================


from scispacy.candidate_generation import CandidateGenerator
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# Initialize the CandidateGenerator with the UMLS knowledge base.
# This object will be used to find potential UMLS concepts matching text strings.
generator = CandidateGenerator(name='umls')


def get_candidates(
        generator, target, k=30, threshold=0.5, no_definition_threshold=0.95, 
        filter_for_definitions=True, max_entities_per_mention=5):
    """
    Retrieves and filters potential UMLS candidates for a given target text string.

    Args:
        generator (CandidateGenerator): The initialized CandidateGenerator instance.
        target (str): The text string (e.g., a diagnosis) to find UMLS candidates for.
        k (int): The maximum number of initial candidates to retrieve from the generator.
        threshold (float): The minimum similarity score for a candidate to be considered.
        no_definition_threshold (float): A potentially higher threshold applied to candidates
                                         that lack a definition in the UMLS knowledge base,
                                         if filter_for_definitions is True.
        filter_for_definitions (bool): If True, applies the no_definition_threshold to
                                       candidates without definitions.
        max_entities_per_mention (int): The maximum number of final candidates to return
                                        after filtering and sorting.

    Returns:
        tuple: A tuple containing:
            - The original target text string.
            - A list of tuples, where each inner tuple represents a filtered UMLS candidate
              (concept_id, canonical_name, similarity_score), sorted by score descending.
    """
    # We can use the CandidateGenerator to get the UMLS entities

    candidates = generator([target], k)[0]
    predicted = []
    for cand in candidates:
        score = max(cand.similarities)
        if (
            filter_for_definitions
            and generator.kb.cui_to_entity[cand.concept_id].definition is None
            and score < no_definition_threshold
        ):
            continue
        if score > threshold:
            name = cand.canonical_name if hasattr(cand, 'canonical_name') else cand.aliases[0]
            predicted.append((cand.concept_id, name, score))
    sorted_predicted = sorted(predicted, reverse=True, key=lambda x: x[2])
    return target, sorted_predicted[: max_entities_per_mention]


def run_extraction(predictions):
    """
    Processes a DataFrame of demographic predictions, extracts UMLS candidates for
    diagnoses associated with patient groups, and returns the results.

    Args:
        predictions (pd.DataFrame): DataFrame containing demographic extractions.
                                    Must include columns like 'pmcid', 'group_name',
                                    'diagnosis', 'count', and potentially 'start_char',
                                    'end_char'. The index of this DataFrame is used as 'group_ix'.

    Returns:
        list: A list of dictionaries, where each dictionary represents a single
              UMLS candidate prediction linked to an original diagnosis entry.
    """
    results = []
    for pmcid, doc_preds in tqdm(predictions.groupby('pmcid')):
        for ix, pred in doc_preds.iterrows():
            # Get the UMLS entities that match the targettarg
            if pred['group_name'] == 'patients' and pd.isna(pred['diagnosis']) == False:
                resolved_target, target_ents = get_candidates(
                    generator, pred['diagnosis'])
                for ent in target_ents:
                    results.append({
                        "pmcid": int(pmcid),
                        "diagnosis": resolved_target,
                        "umls_cui": ent[0],
                        "umls_name": ent[1],
                        "umls_prob": ent[2],
                        "count": pred['count'],
                        "group_ix": ix,
                        "start_char": pred['start_char'] if 'start_char' in pred else None,
                        "end_char": pred['end_char'] if 'end_char' in pred else None,
                    })

    return results

# --- Main Execution Block ---

# Define the directory where the input demographic extraction files are located.
extractions_dir = Path('../outputs/demographicsextractions')

# Find all relevant demographic extraction files using glob patterns.
# Selects specific zeroshot extractions (chunked and full text) that have '_noabbrev' suffix.
all_files = list(extractions_dir.glob('chunked_*zeroshot*_noabbrev.csv')) + list(extractions_dir.glob('full_*zeroshot*_noabbrev.csv'))

# Loop through each identified demographic extraction file path.
for pred_path in all_files:
    print(f"Processing {pred_path}")
    out_name = Path(str(pred_path).replace('_noabbrev', '_umls'))
    predictions = pd.read_csv(pred_path)
    results = run_extraction(predictions)
    results_df = pd.DataFrame(results)

    # Remove _clean from the filename
    results_df.to_csv(out_name, index=False)
