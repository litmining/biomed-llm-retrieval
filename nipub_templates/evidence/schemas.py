import inspect
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
from typing import List, Optional, Tuple, Type, get_origin, get_args, Union, Any, Dict

# Type alias for a single piece of evidence (start char, end char)
EvidenceQuote = Tuple[int, int]

# Type alias for the evidence supporting a single atomic value (list of quotes)
EvidenceForValue = List[EvidenceQuote]

# --- Function to Create Evidence Models ---

# Cache for generated evidence models to avoid redundant creation
_evidence_model_cache: Dict[Type[BaseModel], Type[BaseModel]] = {}

def create_evidence_model(
    original_model: Type[BaseModel],
    base_model_class: Type[BaseModel] = BaseModel
) -> Type[BaseModel]:
    """
    Dynamically creates a Pydantic model for capturing evidence locations
    corresponding to the fields of an original Pydantic model.

    Args:
        original_model: The original Pydantic model class used for extraction.
        base_model_class: The base class to use for the generated model (defaults to pydantic.BaseModel).

    Returns:
        A new Pydantic model class designed for evidence capture.
    """
    if original_model in _evidence_model_cache:
        return _evidence_model_cache[original_model]

    evidence_fields: Dict[str, Tuple[Type, FieldInfo]] = {}
    original_fields = original_model.model_fields # Pydantic v2

    for field_name, field_info in original_fields.items():
        original_type = field_info.annotation
        origin = get_origin(original_type)
        args = get_args(original_type)

        evidence_field_name = f"{field_name}_Evidence"
        evidence_description = (
            f"List of text spans (start_char, end_char tuples) from the source text "
            f"providing evidence for the extracted '{field_name}' field."
        )

        evidence_type: Type = Any # Default type
        is_optional = False

        # Check if the original field is Optional
        if origin is Union and type(None) in args:
            is_optional = True
            # Get the non-None type
            original_type = next(t for t in args if t is not type(None)) # noqa: E721
            origin = get_origin(original_type) # Re-evaluate origin and args
            args = get_args(original_type)

        if origin is list or origin is List:
            # --- Handle List Types ---
            list_item_type = args[0]
            list_item_origin = get_origin(list_item_type)
            list_item_args = get_args(list_item_type)

            # Check if the list item is a Pydantic model
            if inspect.isclass(list_item_type) and issubclass(list_item_type, BaseModel):
                # Recursively create the evidence model for the nested type
                nested_evidence_model = create_evidence_model(list_item_type, base_model_class)
                # The evidence field will be a list of these nested evidence models
                evidence_type = List[nested_evidence_model]
                evidence_description = (
                    f"List of evidence structures, one for each item in the original '{field_name}' list. "
                    f"Each structure contains text spans (start_char, end_char tuples) "
                    f"providing evidence for the corresponding item's fields."
                )
            else:
                # It's a list of simple types (str, int, Literal, etc.)
                # Evidence should be a list where each element corresponds to an item
                # in the original list, and contains a list of quotes for that item.
                evidence_type = List[EvidenceForValue]
                evidence_description = (
                    f"List of evidence lists, one for each item in the original '{field_name}' list. "
                    f"Each inner list contains text spans (start_char, end_char tuples) "
                    f"providing evidence for the corresponding item."
                )

        elif inspect.isclass(original_type) and issubclass(original_type, BaseModel):
             # --- Handle Nested Pydantic Models (not in a List) ---
             # This case might be less common for top-level fields but included for completeness
             nested_evidence_model = create_evidence_model(original_type, base_model_class)
             evidence_type = nested_evidence_model # Evidence is the nested evidence model itself
             evidence_description = (
                 f"Evidence structure for the nested '{field_name}' object. "
                 f"Contains text spans (start_char, end_char tuples) for each field within '{field_name}'."
             )

        else:
            # --- Handle Simple Types (str, int, bool, Literal, etc.) ---
            evidence_type = EvidenceForValue # List of (start, end) tuples
            evidence_description = (
                 f"List of text spans (start_char, end_char tuples) from the source text "
                 f"providing evidence for the extracted '{field_name}' value."
             )

        # Wrap in Optional if the original field was Optional
        final_evidence_type = Optional[evidence_type] if is_optional else evidence_type

        evidence_fields[evidence_field_name] = (
            final_evidence_type,
            Field(
                default=None if is_optional else ([] if get_origin(evidence_type) is list else None),
                description=evidence_description
            )
        )

    # Create the new evidence model dynamically
    evidence_model_name = f"{original_model.__name__}Evidence"
    created_model = create_model(
        evidence_model_name,
        __base__=base_model_class,
        __module__=original_model.__module__, # Keep it in the same module scope if needed
        **evidence_fields # type: ignore
    )

    # Add a docstring to the generated model
    created_model.__doc__ = f"""
    Model for capturing text evidence locations for fields extracted using {original_model.__name__}.
    Each field ending in '_Evidence' corresponds to a field in the original model.
    Evidence is provided as lists of (start_char, end_char) tuples referencing the source text.
    For list fields in the original model, the evidence field mirrors the list structure.
    """

    _evidence_model_cache[original_model] = created_model
    return created_model
