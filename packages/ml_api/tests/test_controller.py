from toxicity_model import __version__ as _version
from packages.ml_api.api import __version__ as api_version
import warnings
import json
import pandas as pd
from toxicity_model.processing.clean_data import load_dataset
from toxicity_model.config import config
warnings.simplefilter(action='ignore', category=FutureWarning)



def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200
    
def test_prediction_end_point_return_prediction(flask_test_client):
    # Given
    
    
    test_data = load_dataset(filename=config.TESTING_API_DATA_FILE)
    post_json = test_data[0:1].to_json(orient='records')
    
    #When
    response = flask_test_client.post('/v1/predict/toxicity',
                                      json=post_json)
    
    #Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    predictions = response_json['predictions']
    response_version = response_json['version']
    dict_keys = list(predictions.keys())
    lstm_predictions = predictions['lstm']
    bilstm_predictions = predictions['bilstm']
    attention_predictions = predictions['attentionnet']
    
    assert set(dict_keys) == set(['lstm', 'bilstm', 'attentionnet'])
    assert all(x >= 0.0 and x <= 1.0 for x in lstm_predictions)
    assert all(x >= 0.0 and x <= 1.0 for x in bilstm_predictions)
    assert all(x >= 0.0 and x <= 1.0 for x in attention_predictions)
    assert response_version == _version
    
def test_version_endpoint_returns_version(flask_test_client):
    # When
    
    response = flask_test_client.get('/version')
    
    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json['model_version'] == _version
    assert response_json['api_version'] == api_version
    





