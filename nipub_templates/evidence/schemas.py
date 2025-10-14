import inspect
from pydantic import BaseModel, Field, create_model
# FieldInfo is not directly used by user code here usually
# from pydantic.fields import FieldInfo
from typing import List, Optional, Tuple, Type, get_origin, get_args, Union, Any, Dict
import json # For printing schema

# --- NEW: Define Evidence Quote as a Model ---
class EvidenceQuoteModel(BaseModel):
    """Represents a single text span evidence (start and end character)."""
    start: int = Field(description="Start character index of the evidence text span.")
    end: int = Field(description="End character index (exclusive) of the evidence text span.")
    # You could add model_config here if needed, but defaults are likely fine

# --- UPDATED: Type alias now uses the Model ---
EvidenceQuote = EvidenceQuoteModel

EvidenceForValue = EvidenceQuote

# --- Function to Create Evidence Models (No changes needed inside) ---

# Cache for generated evidence models
_evidence_model_cache: Dict[Type[BaseModel], Type[BaseModel]] = {}

def create_evidence_model(
    original_model: Type[BaseModel],
    base_model_class: Type[BaseModel] = BaseModel
) -> Type[BaseModel]:
    """
    Dynamically creates a Pydantic model for capturing evidence locations
    corresponding to the fields of an original Pydantic model.
    (Code remains the same as the previous version, it will now naturally
     use EvidenceQuoteModel where EvidenceQuote/EvidenceForValue are used)
    """
    if original_model in _evidence_model_cache:
        return _evidence_model_cache[original_model]

    evidence_fields: Dict[str, Tuple[Type, FieldInfo]] = {} # type: ignore # Use FieldInfo type hint if available/needed
    original_fields = original_model.model_fields # Pydantic v2

    for field_name, field_info in original_fields.items():
        original_annotation = field_info.annotation
        origin = get_origin(original_annotation)
        args = get_args(original_annotation)

        evidence_field_name = f"{field_name}_Evidence"
 
        evidence_type: Type = Any
        is_optional = False
        processed_type = original_annotation

        if origin is Union and type(None) in args:
            is_optional = True
            non_none_args = tuple(t for t in args if t is not type(None))
            if len(non_none_args) == 1:
                processed_type = non_none_args[0]
            else:
                processed_type = Union[non_none_args] # type: ignore
            origin = get_origin(processed_type)
            args = get_args(processed_type)

        evidence_value_origin = None

        if origin is list or origin is List:
            if not args: continue
            list_item_type = args[0]

            if inspect.isclass(list_item_type) and issubclass(list_item_type, BaseModel):
                nested_evidence_model = create_evidence_model(list_item_type, base_model_class)
                evidence_type = List[nested_evidence_model]
                evidence_value_origin = list
            else:
                # List of simple types -> Evidence is List[EvidenceForValue]
                # EvidenceForValue is now List[EvidenceQuoteModel]
                evidence_type = List[EvidenceForValue]
                evidence_value_origin = list

        elif inspect.isclass(processed_type) and issubclass(processed_type, BaseModel):
             nested_evidence_model = create_evidence_model(processed_type, base_model_class)
             evidence_type = nested_evidence_model
             evidence_value_origin = object # Using 'object' loosely here

        else:
            # Simple Types -> Evidence is EvidenceForValue
            # EvidenceForValue is now List[EvidenceQuoteModel]
            evidence_type = EvidenceForValue
            evidence_value_origin = list


        final_evidence_type = Optional[evidence_type] if is_optional else evidence_type

        field_kwargs: Dict[str, Any] = {}
        if is_optional:
            field_kwargs["default"] = None
        elif evidence_value_origin is list:
            field_kwargs["default"] = []

        # Pydantic's FieldInfo type isn't directly needed for the field definition tuple here
        evidence_fields[evidence_field_name] = (
            final_evidence_type,
            Field(**field_kwargs) # type: ignore
        )

    evidence_model_name = f"{original_model.__name__}Evidence"
    created_model = create_model(
        evidence_model_name,
        __base__=base_model_class,
        __module__=original_model.__module__,
        **evidence_fields # type: ignore
    )

    _evidence_model_cache[original_model] = created_model
    return created_model