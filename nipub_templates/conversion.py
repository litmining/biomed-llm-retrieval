""""
This module contains functions for converting JSON schemas into JSON templates, whice are primarily used by 
the NuExtract model for zero-shot information extraction tasks.


V1 Template Example:
"Model": {
    "Name": "",
    "Number of parameters": "",
    "Number of max token": "",
    "Architecture": []
},
"Usage": {
    "Use case": [],
    "Licence": ""
}

V2 Template Example:
{
  "first_name": "verbatim-string",
  "last_name": "verbatim-string",
  "description": "string",
  "age": "integer",
  "gpa": "number",
  "birth_date": "date-time",
  "nationality": ["France", "England", "Japan", "USA", "China"],
  "languages_spoken": [["English", "French", "Japanese", "Mandarin", "Spanish"]]
}

"""


def schema_to_template(schema: dict, version: str = "v1") -> dict:
    """
    Convert a JSON Schema into a JSON "template" dictionary, 
    filling in required fields with simple placeholder values.
    """

    # A helper dictionary for default placeholder values by JSON type.
    DEFAULTS_V1 = {
        "string": "",
        "boolean": False,
        "number": 0,
        "integer": 0,
        "object": {},
        "array": [],
        "null": None
    }

    DEFAULTS_V2 = {
        "string": "verbatim-string",
        "boolean": ["true", "false"],
        "number": "number",
        "integer": "integer",
        "object": {},
        "array": [],
        "null": None
    }

    # Extract top-level $defs (used for resolving $ref)
    defs = schema.get("$defs", {})

    def resolve_ref(ref_path: str) -> dict:
        """
        Resolve a $ref of the form '#/$defs/SomeDefinition' 
        and return the corresponding schema dictionary.
        """
        # Basic handling: assume local refs only, of form '#/$defs/SomeName'
        parts = ref_path.strip("#/").split("/")
        # e.g., ["$defs", "SomeDefinition"]
        # navigate to that definition
        sub_schema = schema
        for p in parts:
            sub_schema = sub_schema[p]
        return sub_schema

    def pick_schema_from_any_of(any_of_list: list) -> dict:
        """
        Given an 'anyOf' list, pick the first schema that is 
        not simply 'type: null'. 
        If everything includes null or there's no suitable match, 
        return the first entry as fallback.
        """
        for sub in any_of_list:
            if sub.get("type") and sub["type"] != "null":
                return sub
            # or if there's 'items' or 'properties', we might pick that
            if "properties" in sub or "items" in sub:
                return sub
        # Fallback: just return the first
        return any_of_list[0]

    def build_template(subschema: dict) -> any:
        """
        Recursively build a template based on subschema.
        """

        defaults = DEFAULTS_V1 if version == "v1" else DEFAULTS_V2

        if "$ref" in subschema:
            # Resolve reference and build from the real subschema
            real_schema = resolve_ref(subschema["$ref"])
            return build_template(real_schema)

        # If 'anyOf' is present, pick one option and proceed
        # Typically, in our schemas this is used for optional fields
        if "anyOf" in subschema:
            chosen = pick_schema_from_any_of(subschema["anyOf"])
            # Merge chosen schema with any outer properties 
            # (sometimes schemas combine anyOf + properties)
            merged = dict(chosen, **{k: v for k, v in subschema.items() if k not in ("anyOf",)})
            return build_template(merged)

        # Identify type
        json_type = subschema.get("type")

        # If there's an enum, return empty string
        if "enum" in subschema:
            if version == "v1":
                return ""
            else:
                # Return all values in [[...]] format
                return [str(v) for v in subschema["enum"]]

        # If no "type" is specified, assume "object" if we see "properties"
        if not json_type and "properties" in subschema:
            json_type = "object"

        # If no type and no properties, fallback to string
        if not json_type:
            json_type = "string"

        # Handle a top-level "null" type
        # or anyOf => null. We could return None in that case:
        if json_type == "null":
            return None

        # If it's an object:
        if json_type == "object":
            required_fields = subschema.get("required", [])
            props = subschema.get("properties", {})

            template_object = {}
            for prop_name in required_fields:
                prop_schema = props.get(prop_name, {})
                template_object[prop_name] = build_template(prop_schema)
            return template_object

        # If it's an array:
        if json_type == "array":
            # We can either return an empty list or a list with a single example item
            items_schema = subschema.get("items", {})
            # If items is itself a list of schemas, we'd have to handle each. 
            # But here we assume it's a single schema or $ref.
            return [build_template(items_schema)]

        # If it's a basic type (string, number, boolean, integer)
        return defaults.get(json_type, "")

    # Now build the template from the top-level schema
    return build_template(schema)
