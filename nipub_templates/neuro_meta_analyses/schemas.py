from typing import List
from pydantic import BaseModel, Field
from typing_extensions import Literal, Optional, Dict


class MetaAnalysisModel(BaseModel):
    """ Metadata for the study as a whole """
    Modality: List[Literal["fMRI-BOLD", "StructuralMRI", "DiffusionMRI", "PET FDG", "PET [15O]-water", "fMRI-CBF", "fMRI-CBV", "MEG", "EEG", "Other"]] = Field(
        description="Modality of the neuroimaging data",
    )
    CognitiveDomain: Optional[List[str]] = Field(
        description="Cognitive, affective, perceptual, or motor domain(s) targeted by the meta-analysis (e.g., working memory, emotion processing, language, social cognition)."
    )
    PopulationDescription: Optional[str] = Field(
        description="Brief description of the study population(s) across included studies, e.g., healthy adults, clinical group (schizophrenia), mixed."
    )
    AgeRange: Optional[str] = Field(
        description="Age range of participants in the studies included in the meta-analysis. This may include specific age groups or ranges, such as '18-35 years' or 'adolescents'."
    )
    ClinicalStatus: Optional[List[Literal["Healthy", "Clinical", "Mixed"]]] = Field(
        description="Clinical status of the participants in the studies included in the meta-analysis. This may include 'Healthy' for healthy controls or 'Clinical' for participants with specific clinical conditions."
    )
    SearchQuery: Optional[str] = Field(
        description="The search query used to identify relevant studies for the meta-analysis. This may include specific keywords, phrases, or combinations of terms. Repeat verbatim PubMed queries if available"
    )
    DatesOfSearch: Optional[str] = Field(
        description="The dates during which the search for relevant studies was conducted. This may include the start and end dates of the search period."
    )
    SearchDatabase: Optional[str] = Field(
        description="The database or databases used to conduct the search for relevant studies. This may include PubMed, Scopus, Web of Science, or other academic databases."
    )
    AdditionalIdentificationMethods: Optional[str] = Field(
        description="Any additional methods used to identify studies beyond the search query. This may include manual searches, reference list checks, or other strategies employed to ensure comprehensive study identification."
    )
    InclusionCriteria: Optional[str] = Field(
        description="Criteria used to include studies in the meta-analysis. This may include specific conditions, populations, or methodological standards that studies must meet to be considered."
    )
    ExclusionCriteria: Optional[str] = Field(
        description="Criteria used to exclude studies from the meta-analysis. This may include specific conditions, populations, or methodological limitations that were not met by certain studies."
    )
    TotalIdentifiedStudies: Optional[int] = Field(
        description="The total number of studies identified through the search query. This includes all studies that were initially considered for inclusion in the meta-analysis."
    )
    FinalNumberOfStudies: Optional[int] = Field(
        description="The final number of studies included in the meta-analysis after applying the inclusion and exclusion criteria."
    )
    CoordinateSpace: Optional[Literal["MNI", "Talairach", "Mixed", "Not reported"]] = Field(
        description="The stereotaxic coordinate space used across included studies."
    )
    MetaAnalysisMethod: Optional[str] = Field(
        description="The statistical method or approach used to conduct the neuroimaging-meta-analysis. This may include specific algorithms, models, or software used to analyze the data."
    )
    MetaAnalysisSoftware: Optional[str] = Field(
        description="The software or tools used to perform the meta-analysis. This may include specific packages, libraries, or platforms utilized for data analysis."
    )
    AnalysesConducted: Optional[List[str]] = Field(
        description="A list of specific analyses conducted as part of the meta-analysis. This may include statistical tests, comparisons, or other analytical procedures performed on the data."
    )
   

class InclusionCriteriaModel(BaseModel):
    # Language / publication
    published_in_english: Optional[bool] = Field(
        description="True if studies were required to be published in English."
    )
    peer_reviewed_only: Optional[bool] = Field(
        description="True if only peer-reviewed studies were included."
    )
    original_research_only: Optional[bool] = Field(
        description="True if only original research articles (not reviews/meta-analyses) were included."
    )

    # Participants
    allowed_age_groups: Optional[List[Literal["Adults", "Children", "Adolescents", "OlderAdults"]]] = Field(
        description="Age groups allowed in included studies."
    )
    allowed_conditions: Optional[List[Literal["HealthyControls", "MDD", "Schizophrenia", "Anxiety", "PTSD", "OtherClinical"]]] = Field(
        description="Participant conditions/diagnoses allowed."
    )

    # Methodological
    imaging_modalities: Optional[List[Literal["fMRI", "PET", "StructuralMRI", "DiffusionMRI"]]] = Field(
        description="Neuroimaging modalities required."
    )
    whole_brain_required: Optional[bool] = Field(
        description="True if whole-brain analyses were required."
    )
    coordinates_required: Optional[bool] = Field(
        description="True if reporting of peak coordinates was required."
    )
    coordinate_space: Optional[List[Literal["MNI", "Talairach"]]] = Field(
        description="Accepted stereotaxic coordinate spaces."
    )
    diagnostic_framework: Optional[List[Literal["DSM", "ICD"]]] = Field(
        description="Diagnostic frameworks required for clinical inclusion."
    )

    # Design / outcomes
    accepted_designs: Optional[List[Literal["RCT", "CaseControl", "CrossSectional", "Other"]]] = Field(
        description="Study designs allowed."
    )
    accepted_tasks: Optional[List[Literal["WorkingMemory", "EmotionRegulation", "DecisionMaking", "Other"]]] = Field(
        description="Experimental tasks accepted."
    )
    required_outcome_measures: Optional[List[Literal["ALFF", "ReHo", "FunctionalConnectivity", "Other"]]] = Field(
        description="Required neuroimaging outcome measures."
    )


class ExclusionCriteriaModel(BaseModel):
    # Study type
    excluded_non_original: Optional[bool] = Field(
        description="True if non-original research (reviews, meta-analyses, case studies) was excluded."
    )
    excluded_conference_abstracts: Optional[bool] = Field(
        description="True if conference abstracts were excluded."
    )

    # Participants
    excluded_comorbid_conditions: Optional[bool] = Field(
        description="True if participants with comorbid psychiatric/neurological conditions were excluded."
    )
    excluded_children: Optional[bool] = Field(
        description="True if children/adolescents were excluded (unless specifically targeted)."
    )

    # Methodological
    excluded_roi_only: Optional[bool] = Field(
        description="True if ROI-only studies were excluded."
    )
    excluded_non_fmri_pet: Optional[bool] = Field(
        description="True if non-fMRI/PET studies were excluded."
    )
    excluded_no_coordinates: Optional[bool] = Field(
        description="True if studies without stereotaxic coordinates were excluded."
    )

    # Data quality
    excluded_small_sample: Optional[bool] = Field(
        description="True if studies with insufficient sample sizes (e.g., <10) were excluded."
    )
    excluded_null_findings: Optional[bool] = Field(
        description="True if studies with only null results (without usable data) were excluded."
    )
    excluded_no_multiple_comparison_correction: Optional[bool] = Field(
        description="True if studies not correcting for multiple comparisons were excluded."
    )

    # Specificity
    excluded_irrelevant_tasks: Optional[bool] = Field(
        description="True if tasks irrelevant to target cognitive/emotional processes were excluded."
    )


class StudyCriteriaModel(BaseModel):
    """Structured yes/no criteria for inclusion and exclusion in neuroimaging meta-analyses."""
    inclusion: InclusionCriteriaModel
    exclusion: ExclusionCriteriaModel
