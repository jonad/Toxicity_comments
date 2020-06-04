from flask import Blueprint, request, jsonify
from toxicity_model.predict import make_prediction
from toxicity_model import __version__ as _version
from packages.ml_api.api import __version__ as api_version
from packages.ml_api.api import validation


from packages.ml_api.api.config import get_logger

_logger = get_logger(logger_name=__name__)



prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        return 'ok'



@prediction_app.route('/v1/predict/toxicity', methods=['POST'])
def predict():
    if request.method == 'POST':
        json_data = request.get_json()
        _logger.info(f'Inputs: {json_data}')
        
        # validate data
        input_data, errors = validation.validate_inputs(input_json=json_data)
        
        result = make_prediction(input_data=input_data)
        _logger.info(f'Outputs: {result}')

        predictions = result.get('predictions')
        version = result.get('version')

        return jsonify({'predictions': predictions,
                        'version': version})
    
@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': _version,
                        'api_version': api_version})