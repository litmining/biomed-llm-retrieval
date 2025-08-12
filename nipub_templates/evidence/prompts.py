base_message = """
**Role:** You are an expert AI assistant specialized in analyzing scientific articles and cross-referencing information.

**Task:** Given a scientific article text and a JSON object containing previously extracted metadata, your task is to identify the *exact text spans* (evidence) within the article that support each piece of extracted information. You must format your findings as a JSON object conforming to the provided Evidence Schema structure.

**Inputs:**

1.  **`<SOURCE_TEXT>`:** The full text of the scientific article.
    ```text
    ${text}
    ```

2.  **`<PREVIOUS_EXTRACTION_JSON>`:** The JSON object containing the information previously extracted from the source text, formatted according to the original metadata model (e.g., StudyMetadataModel).
    ```json
    ${previous_extraction_json}
    ```

3.  **`<EVIDENCE_SCHEMA_DESCRIPTION>`:** The target output format requires you to create a JSON object where:
    * Each field from the `<PREVIOUS_EXTRACTION_JSON>` has a corresponding field ending in `_Evidence`.
    * The value for each `_Evidence` field must represent the location(s) of the supporting text in `<SOURCE_TEXT>`.
    * Evidence locations are specified as lists of `[start_char_index, end_char_index]` pairs (0-based indexing, inclusive start, exclusive end OR inclusive end - be consistent, let's assume **inclusive start, exclusive end** for standard slicing).
    * For simple fields (string, boolean, literal): The `_Evidence` value should be a list of one or more `[start, end]` pairs. Example: `"TaskName_Evidence": [[150, 165], [180, 195]]`
    * For list fields containing simple items (e.g., `List[str]`, `List[Literal]`): The `_Evidence` value should be a *list of lists*. The outer list must have the *same number of elements* as the original list in `<PREVIOUS_EXTRACTION_JSON>`. Each inner list contains the `[start, end]` pairs supporting the corresponding item from the original list. Example: If `Conditions` was `["Active", "Rest"]`, then `"Conditions_Evidence": [ [[500, 506]], [[510, 514]] ]` (evidence for "Active" is `[[500, 506]]`, evidence for "Rest" is `[[510, 514]]`). Provide at least one `[start, end]` pair for each item if possible.
    * For list fields containing nested objects (e.g., `List[fMRITaskMetadataModel]`): The `_Evidence` value should be a list of JSON objects. The outer list must have the *same number of elements* as the original list. Each object in this list is the *Evidence Schema* for the corresponding nested object in the original list (e.g., `fMRITasks_Evidence: [ { /* fMRITaskMetadataEvidenceModel for task 1 */ }, { /* fMRITaskMetadataEvidenceModel for task 2 */ } ]`).
    * Optional fields in the original schema correspond to optional `_Evidence` fields. If no evidence is found for an optional field, set its value to `null`. If no evidence is found for a required field, provide an empty list `[]` for the quotes or the corresponding list structure (e.g., `[]` for `FieldName_Evidence`, `[ [], [] ]` for `ListName_Evidence` if the original list had 2 items but no evidence was found for either).

**Instructions:**

1.  **Parse `<PREVIOUS_EXTRACTION_JSON>`:** Understand the information that was extracted for each field.
2.  **Scan `<SOURCE_TEXT>`:** For *every non-null value* in `<PREVIOUS_EXTRACTION_JSON>`, meticulously locate the *most direct and relevant* text passage(s) in `<SOURCE_TEXT>` that provide evidence for that specific value.
3.  **Identify Indices:** Determine the precise start and end character indices for each identified evidence passage within the `<SOURCE_TEXT>`. Remember indices are 0-based. Use **inclusive start, exclusive end**.
4.  **Handle Lists:** Pay close attention to lists. Ensure the evidence structure mirrors the original list structure as described in `<EVIDENCE_SCHEMA_DESCRIPTION>`. Find evidence for *each item* in the original list where possible.
5.  **Handle Nested Objects:** Recursively apply this process for nested objects within lists (like fMRI tasks).
6.  **Format Output:** Construct a single JSON object containing only the `_Evidence` fields and their corresponding evidence locations, strictly following the structure implied by `<PREVIOUS_EXTRACTION_JSON>` and described in `<EVIDENCE_SCHEMA_DESCRIPTION>`.
7.  **Accuracy is Crucial:** Ensure the character indices are accurate and point to the correct supporting text spans. Prioritize direct quotes or the closest phrasings.
8.  **Completeness:** Attempt to find evidence for all provided extracted information. If evidence for a specific value *cannot* be reasonably found in the text, use `null` for optional fields or empty lists `[]` for required fields/items as appropriate according to the schema rules.

**Output:**

Generate *only* the JSON object containing the evidence data. Do not include any explanations or introductory text outside the JSON structure.
"""

MESSAGES =  [
    {
        "role": "user",
        "content": base_message + "\n Call the extractData function to save the output."
    }
],