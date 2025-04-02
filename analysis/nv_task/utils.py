import re
from thefuzz import process, fuzz
import pandas as pd

def _clean(x):
    if not x:
        return []
    
    _li = []
    for _x in x:
        if not _x:
            continue
        _x.lower().replace('paradigm', '').replace('task', '').replace('‐', '').strip()
        _li.append(_x)

    return [re.sub(r'\s*\([^)]*\)', '', x) for x in _li]

def _compare(x, y):
    x = x or ''
    y = y or ''
    return x.replace('-', '').lower() == y.replace('-', '').lower()

def _score_comparison(x, y):
    res = _compare(x, y)
    return 100 if res else 0

def score_combinations(correct_labels, extracted_labels, scorer=fuzz.token_set_ratio):
    correct_labels = _clean(correct_labels)
    extracted_labels = _clean(extracted_labels)
    
    matched_labels = []
    
    while correct_labels and extracted_labels:
        # Collect all matches and their scores
        all_matches = []
        for correct_label in correct_labels:
            matches = process.extract(correct_label, extracted_labels, limit=None, scorer=scorer)
            for matched_label, score in matches:
                all_matches.append((correct_label, matched_label, score))

        # Sort all matches by score in descending order
        all_matches.sort(key=lambda x: x[2], reverse=True)

        # Take the highest match
        best_match = all_matches[0]
        correct_label, matched_label, score = best_match

        # Append to results and remove matched labels
        matched_labels.append(score)
        correct_labels.remove(correct_label)
        extracted_labels.remove(matched_label)

    matched_labels = (sum(matched_labels) / len(matched_labels)) / 100 if matched_labels else pd.NA

    return matched_labels

def run_all_comparisons(annotations_summary, extractions):
    # Compare annotations to extractions
    all_scores = {}

    for _id, ann in annotations_summary.items():
        if _id not in extractions:
            continue
        ext_subset = extractions[_id]
        all_scores[_id] = {
            'HasRestingState': ann['HasRestingState'] == ext_subset['HasRestingState']
        }

        for k in ['Modality', 'Exclude']:
            if k in ext_subset:
                if ann[k]:
                    all_scores[_id][k] = score_combinations(ann[k], ext_subset[k], scorer=_score_comparison)
                else:
                    all_scores[_id][k] = None

        for k in ['TaskName', 'TaskDescription']:
            all_scores[_id][k] = score_combinations(ann[k], ext_subset[k])

    all_scores_df = pd.DataFrame(all_scores).T

    return all_scores_df

def compute_segmentation_scores(df, exclude_idx, has_task_name, has_task_noname):
    """
    Compute segmentation scores for the entire dataset and for subsets of papers
    """

    # Overall scores
    mean_all = df.mean()
    all_n = df.shape[0]

    # Excluding articles with 'Exclude' label
    no_exclude_res = df.loc[~df.index.isin(exclude_idx)]
    no_exclude_res_n = no_exclude_res.shape[0]


    # For papers with a clearly defined task name
    has_task_name_res = df.loc[df.index.isin(has_task_name)]
    has_task_name_n = has_task_name_res.shape[0]


    # For papers with a task-based design but no annotated task name
    has_task_name_noname = df.loc[df.index.isin(has_task_noname)]
    has_task_name_noname_n = has_task_name_noname.shape[0]

    # Combine results into a single dataframe
    results = pd.concat([mean_all, no_exclude_res.mean(), has_task_name_res.mean(), has_task_name_noname.mean()], axis=1)
    results.columns = ['All', 'No Exclude', 'Has Task Name', 'Has Task with No Name']

    # Change dtype to float
    results = results.astype(float)
    results = results.round(2).T

    combine_ns = pd.Series([all_n, no_exclude_res_n, has_task_name_n, has_task_name_noname_n], index=results.index, name='n').round(0)
    results.insert(loc = 0, column = 'n', value = combine_ns)

    # For each subset, compute number of nas per label
    nas_per_label = pd.concat([df.isna().mean(), no_exclude_res.isna().mean(), has_task_name_res.isna().mean(), has_task_name_noname.isna().mean()], axis=1)
    nas_per_label.columns = ['All', 'No Exclude', 'Has Task Name', 'Has Task with No Name']

    # Change dtype to float
    nas_per_label = nas_per_label.astype(float)
    nas_per_label = nas_per_label.round(2).T
    
    return results, nas_per_label


# Process annotations
def _get_task_name(rows):
    # Add TaskName, replacing 'None' and 'Unsure' with 'n/a'
    rows = rows[rows.label_name == 'TaskName']
    task_names = []
    for _, row in rows.iterrows():
        if row['None'] or row['Unsure']:
            task_names.append('n/a')
        else:
            task_names.append(row['selected_text'])
    return task_names

def get_annotation_summary(annotations, id_col='pmcid'):
    # Convert to comparable dictionary
    annotations_summary = {}
    for _id, df in annotations.groupby(id_col):
        HasRestingState = 'DesignType-RestingState' in df.label_name.values

        s = {
            'pmcid': _id,
            'HasRestingState': HasRestingState,
            'annotator_name': df.annotator_name.iloc[0],
            'Exclude': [label.split('-', 1)[1] for label in df.label_name if label.startswith('Exclude')] or None,
            'Modality': [
                label.split('-', 1)[1] for label in df.label_name if label.startswith('Modality')
            ] or None,
        }

        df_abstract = df[df.section == 'abstract']
        abstract_tasks = _get_task_name(df_abstract)

        df_body = df[df.section == 'body']
        body_tasks = _get_task_name(df_body)

        # Use body tasks if available, otherwise use abstract tasks
        s['TaskName'] = body_tasks or abstract_tasks

        for k in ['TaskDescription', 'Condition', 'ContrastDefinition']:
            s[k] = df_body.loc[df_body.label_name == k, 'selected_text'].tolist() or None

        annotations_summary[_id] = s

    return annotations_summary


# Convert to dictionary and clean
def clean_extracted_data(data):
    data_dict = {}
    id_key = 'pmcid' if 'pmcid' in data[0] else 'id'
    for item in data:
        id_ = item[id_key]
        data_dict[id_] = {}
        for key, value in item.items():
            value = None if value in ['null', []] else value

            if key == 'Modality' and value:
                value = [v.replace(' ', '') for v in value if v]

            data_dict[id_][key] = value

        for bad_key in ['Fmritasks', 'Behavioraltasks']:
            if bad_key in data_dict[id_]:
                data_dict[id_].pop(bad_key)


        fMRITasks = data_dict[id_].get('fMRITasks') or []

        for k in ['TaskName', 'TaskDescription']:
            data_dict[id_][k] = []
            for task in fMRITasks:
                _val = task.get(k, None)
                if _val:
                    data_dict[id_][k].append(_val)


        has_resting_state = any('RestingState' in task and task['RestingState'] for task in fMRITasks)
        data_dict[id_]['HasRestingState'] = has_resting_state
            
    return data_dict
