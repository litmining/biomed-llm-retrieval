from typing import List
from pydantic import BaseModel, Field


class GroupBase(BaseModel):
    count: int = Field(description="Number of participants in this group")
    diagnosis: str = Field(description="Diagnosis of the group, if any")
    group_name: str = Field(description="Group name, healthy or patients",
                            enum=["healthy", "patients"])
    subgroup_name: str = Field(description="Subgroup name")
    male_count: int = Field(description="Number of male participants in this group")
    female_count: int = Field(description="Number of female participants in this group")
    age_mean: float = Field(description="Mean age of participants in this group")
    age_range: str = Field(description="Age range of participants in this group, separated by a dash")
    age_minimum: int = Field(description="Minimum age of participants in this group")
    age_maximum: int = Field(description="Maximum age of participants in this group")
    age_median: int = Field(description="Median age of participants in this group")


class GroupImaging(GroupBase):
    imaging_sample: str = Field(
        description="Did this subgroup undergo fMRI, MRI or neuroimaging, yes or no", enum=["yes", "no"])


class BaseDemographicsSchema(BaseModel):
    groups: List[GroupImaging]


class RelevantSentSchema(BaseDemographicsSchema):
    relevant_sentences: str = Field(
        description="Highlight the most relevant sentence from the text. Maximum 50 charachters.")


class GroupAssesmentType(GroupBase):
    assesment_type: str = Field(
        description="Was the assesment for this group behavioral, imaging, or other? ",
        enum=["behavioral", "imaging", "other"])


class FewShot2Schema(BaseModel):
    groups: List[GroupAssesmentType]


class InclusionExclusionCriteria(BaseModel):
    """Schema for extracting verbatim inclusion and exclusion criteria from studies."""
    
    inclusion_criteria: str | None = Field(
        description="The VERBATIM text describing inclusion criteria for participant selection. "
        "Extract the exact text as written in the article. If no specific inclusion criteria are mentioned, return null.",
        default=None
    )
    
    exclusion_criteria: str | None = Field(
        description="The VERBATIM text describing exclusion criteria for participant selection. "
        "Extract the exact text as written in the article. If no specific exclusion criteria are mentioned, return null.",
        default=None
    )
    
    has_dedicated_section: bool = Field(
        description="Indicates whether the article has a dedicated section explicitly outlining inclusion and exclusion criteria separately "
        "(e.g., headers like 'Inclusion Criteria' or 'Exclusion Criteria'), or if the criteria are only mentioned in prose within the text.",
        default=False
    )
    
    criteria_location: str | None = Field(
        description="Brief description of where the criteria were found (e.g., 'Methods - Participants section', 'dedicated subsection', 'embedded in participant description')",
        default=None
    )
