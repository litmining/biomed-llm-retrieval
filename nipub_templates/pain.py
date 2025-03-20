from typing import List
from pydantic import BaseModel, Field


base_message = """
You will be provided with a text sample from a scientific journal.
The sample is delimited with triple backticks.

Your task is to identify charachteristics of the study design for a neuroimaging study.

For each experiment identify:
    - the neuroimaging modality used (fMRI-BOLD, Diffusion MRI, PET, etc.)
    - if fMRI, identify the design type ('task-based', 'resting-state', 'connectivity', etc.)
    - if task-based, identify the event design ('block', 'event-related', 'mixed', etc.)
    - if task-based, identify the task type ('motor', 'cognitive', 'emotional', 'pain', etc.)
    - if task-based, identify the condition type ('painful', 'non-painful', 'neutral', etc.)
    - if task-based, identify the conditions ('painful', 'non-painful', 'neutral', '2-back', '0-back', etc.)
    - identify the total length of the experiment (in seconds)
    - for MRI, identify the field strength (1.5T, 3T, 7T, etc.)
    - for MRI, identify the Pulse sequence type sequence type (gradient echo, spin echo, etc.)
    - for MRI, identify the imaging type (echo planar imaging (EPI), spiral, 3D, etc.)
    - for MRI, identify the resolution (in mm)
    - identify the analysis software used (SPM, FSL, AFNI, etc.)
    - identify the preprocessing steps (motion correction, slice timing correction, etc.)
    - identify the statistical model used (GLM, ICA, MVPA, encoding model, etc.)
    - if the study is a pain study, identify the pain type ('acute', 'chronic', 'experimental', etc.)
    - if the study is a pain study, identify the pain location ('hand', 'leg', 'back', etc.)
    - if the study is a pain study, identify the pain modality ('thermal', 'mechanical', 'electrical', etc.)
    - if the study is a pain study, identify the pain intensity ('low', 'medium', 'high', etc.)
    - if the study is a pain study, identify the pain duration ('short', 'medium', 'long', etc.)

    If any of the information is missing, return `null` for that field.         

Text sample: ${text}
"""

class ExperimentDetails(BaseModel):
    neuroimaging_modality: str = Field(None, description="The neuroimaging modality used (fMRI-BOLD, Diffusion MRI, PET, etc.)")
    fmri_design_type: str = Field(None, description="If fMRI, the design type ('task-based', 'resting-state', 'connectivity', etc.)")
    event_design: str = Field(None, description="If task-based, the event design ('block', 'event-related', 'mixed', etc.)")
    task_type: str = Field(None, description="If task-based, the task type ('motor', 'cognitive', 'emotional', 'pain', etc.)")
    condition_type: str = Field(None, description="If task-based, the condition type ('painful', 'non-painful', 'neutral', etc.)")
    conditions: List[str] = Field(None, description="If task-based, the conditions ('painful', 'non-painful', 'neutral', '2-back', '0-back', etc.)")
    total_length_seconds: int = Field(None, description="The total length of the experiment (in seconds)")
    mri_field_strength: str = Field(None, description="For MRI, the field strength (1.5T, 3T, 7T, etc.)")
    pulse_sequence_type: str = Field(None, description="For MRI, the Pulse sequence type (gradient echo, spin echo, etc.)")
    imaging_type: str = Field(None, description="For MRI, the imaging type (echo planar imaging (EPI), spiral, 3D, etc.)")
    resolution_mm: str = Field(None, description="For MRI, the resolution (in mm)")
    analysis_software: str = Field(None, description="The analysis software used (SPM, FSL, AFNI, etc.)")
    preprocessing_steps: List[str] = Field(None, description="The preprocessing steps (motion correction, slice timing correction, etc.)")
    statistical_model: str = Field(None, description="The statistical model used (GLM, ICA, MVPA, encoding model, etc.)")
    pain_type: str = Field(None, description="If the study is a pain study, the pain type ('acute', 'chronic', 'experimental', etc.)")
    pain_location: str = Field(None, description="If the study is a pain study, the pain location ('hand', 'leg', 'back', etc.)")
    pain_modality: str = Field(None, description="If the study is a pain study, the pain modality ('thermal', 'mechanical', 'electrical', etc.)")
    pain_intensity: str = Field(None, description="If the study is a pain study, the pain intensity ('low', 'medium', 'high', etc.)")
    pain_duration: str = Field(None, description="If the study is a pain study, the pain duration ('short', 'medium', 'long', etc.)")

class BasePainSchema(BaseModel):
    groups: List[ExperimentDetails]

ZERO_SHOT_PAIN_DESIGN_FC = {
    "messages": [
        {
            "role": "user",
            "content": base_message + "\n Call the extractData function to save the output."
        }
    ],
    "output_schema": BasePainSchema.model_json_schema()
}