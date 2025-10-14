from .schemas import MetaAnalysisModel, StudyCriteriaModel

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
