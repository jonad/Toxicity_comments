from marshmallow import Schema, fields
from marshmallow import ValidationError

import typing as t
import json

class InvalidInputError(Exception):
    pass

class ToxicCommentSchema(Schema):
    comment_text = fields.Str()
    

def _filter_error_rows(errors: dict, validated_input: t.List[dict]) -> t.List[dict]:
    indexes = errors.keys()
    for index in sorted(indexes, reverse=True):
        del validated_input[index]
    return validated_input

def validate_inputs(input_json):
    schema = ToxicCommentSchema(many=True)
    input_data = json.dumps(input_json)
    input_data = json.loads(input_data)
    
    errors = None
    try:
        schema.loads(input_data)
    except ValidationError as exc:
        errors = exc.messages
        
    if errors:
        validated_input = _filter_error_rows(
            errors=errors,
            validated_input=input_data)
    else:
        validated_input = input_data

    return validated_input, errors
    