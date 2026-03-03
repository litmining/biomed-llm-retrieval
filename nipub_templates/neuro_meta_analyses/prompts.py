from .schemas import MetaAnalysisModel, StudyCriteriaModel, IncludedStudyListModel

base_message = """
You will be provided with a text sample from a neuroimaging meta-analysis
The sample is delimited with triple backticks.

Your task is to identify information about the design of the meta-analysis, with a focus on the inclusion and exclusion criteria, and any other relevant metadata about the study.
If any information is missing or not explicitly stated in the text, return `null` for that field.

For any extracted text, maintain fidelity to the source. Avoid inferring information not explicitly stated. If a field cannot be completed, return `null`.

Text sample: ${text}
"""


META_ANALYSIS_PROMPT = {
    "messages": [
        {
            "role": "user",
            "content": base_message + "\n Call the extractData function to save the output."
        }
    ],
    "output_schema": MetaAnalysisModel.model_json_schema()
}

META_ANALYSIS_INCLUSION_CRITERIA_PROMPT = {
    "messages": [
        {
            "role": "user",
            "content": base_message + "\n Call the extractData function to save the output."
        }
    ],
    "output_schema": StudyCriteriaModel.model_json_schema()
}

message_included_excluded_studies = """
You will be given a text sample from a neuroimaging meta-analysis, delimited by triple backticks.

Your task is to extract only information that is explicitly supported by the provided text.
Do not infer or guess details that are not stated.
If the text does not provide enough information for a field, return null for that field.
Return false only when the text explicitly states that something was not shared or is unavailable.

Extract the following:

1. Study list reporting

Determine whether the meta-analysis provides an explicit list of:
- included studies
- excluded studies
- both
- neither

Definitions:
- "included studies" means the paper explicitly lists the studies included in the meta-analysis, such as in a table, appendix, supplement, or study summary.
- "excluded studies" means the paper explicitly lists excluded studies or provides a table/appendix of excluded records with reasons.
- A PRISMA flowchart alone does not count as an excluded-studies list unless individual excluded studies are explicitly named.
- General references or background citations do not count as an included-studies list unless they are clearly presented as the studies included in the meta-analysis.

Set:
- `study_list.presence = "included"` if only included studies are explicitly listed
- `study_list.presence = "excluded"` if only excluded studies are explicitly listed
- `study_list.presence = "both"` if both are explicitly listed
- `study_list.presence = "none"` if neither is explicitly listed

For locations:
- `included_studies_location` should describe where the included-studies list appears
- `excluded_studies_location` should describe where the excluded-studies list appears
- Use `Main Text` if the explicit list appears in the main article text, tables, or figures
- Use `Supplementary File` if the explicit list appears in supplementary materials
- Use null if that type of list is not present or its location is not stated

Important:
- Included and excluded study lists may have different locations.
- Example: included studies in the main text and excluded studies in the supplement should be represented separately.

2. Coordinate/input data sharing

Determine whether the paper reports sharing the meta-analytic input data, such as coordinates, study tables, or files used as input to the meta-analysis.

Examples of coordinate/input data:
- coordinate tables
- ALE, SDM, Sleuth, or NiMARE input files
- CSV, Excel, or JSON tables containing study-level coordinates or extracted inputs
- raw study tables used to build the meta-analysis dataset

Set:
- `coordinate_data.shared = true` only if the text explicitly states that coordinate/input data are available
- `coordinate_data.shared = false` only if the text explicitly states that coordinate/input data are not shared or unavailable
- otherwise set `coordinate_data.shared = null`

For `coordinate_data.sharing_location`:
- use `Public Repository` if the data are shared only in a repository
- use `Supplementary Material` if the data are shared only in supplementary files
- use `Both` if both are explicitly stated
- otherwise null

For repository fields:
- fill `repository_name` and `repository_url` only for a public repository explicitly mentioned in the text
- if the data are shared only as supplementary material, set both repository fields to null

For format:
- fill `data_format` only if the file format or structured format is explicitly stated
- do not guess the format from the repository alone

3. Result/map data sharing

Determine whether the paper reports sharing meta-analytic result data, such as statistical maps, thresholded maps, output images, or downloadable result files.

Examples of result/map data:
- NeuroVault statistical maps
- NIfTI files
- ZIP archives containing meta-analytic outputs
- CSV, Excel, or JSON result tables
- downloadable result images or map files explicitly described as analysis outputs

Set:
- `result_data.shared = true` only if the text explicitly states that result/map data are available
- `result_data.shared = false` only if the text explicitly states that result/map data are not shared or unavailable
- otherwise set `result_data.shared = null`

For `result_data.sharing_location`:
- use `Public Repository` if the data are shared only in a repository
- use `Supplementary Material` if the data are shared only in supplementary files
- use `Both` if both are explicitly stated
- otherwise null

For repository fields:
- fill `repository_name` and `repository_url` only for a public repository explicitly mentioned in the text
- if the data are shared only as supplementary material, set both repository fields to null

For format:
- fill `data_format` only if the file format is explicitly stated
- do not guess the format from the repository alone unless the text explicitly names it

Cross-field rules:
- Keep coordinate/input data and result/map data separate.
- If raw tables are shared in OSF and maps are shared in NeuroVault, record OSF under `coordinate_data` and NeuroVault under `result_data`.
- If coordinates are shared as a supplementary CSV and not in a repository, set:
  - `coordinate_data.shared = true`
  - `coordinate_data.sharing_location = "Supplementary Material"`
  - `coordinate_data.data_format = "CSV"` if explicitly stated
  - `coordinate_data.repository_name = null`
  - `coordinate_data.repository_url = null`
- If maps are shared in a repository and no coordinate/input data sharing is mentioned, do not mark coordinate data as shared.

Important rules:
- Use only the provided text.
- Do not guess missing repository names, URLs, locations, or formats.
- If sharing is mentioned but the format or repository details are not stated, set those specific fields to null.
- If the text is ambiguous, prefer null rather than guessing.

Text sample:
```${text}```
"""


INCLUDED_EXCLUDED_STUDIES_PROMPT = {
    "messages": [
        {
            "role": "user",
            "content": message_included_excluded_studies + "\n Call the extractData function to save the output."
        }
    ],
    "output_schema": IncludedStudyListModel.model_json_schema()
}