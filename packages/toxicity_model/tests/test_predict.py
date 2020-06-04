from toxicity_model.predict import make_prediction
from toxicity_model.processing.clean_data import load_dataset
from toxicity_model.config import config

def test_make_single_prediction():
    # Given
    test_data = load_dataset(filename=config.TESTING_DATA_FILE)
    single_test_json = test_data[0:1].to_json(orient='records')

    # When
    subject = make_prediction(input_data=single_test_json)

    # Then
    assert subject is not None
    
    
def test_make_prediction_multiple_predictions():
    # Given
    test_data = load_dataset(filename=config.TESTING_DATA_FILE)
    multiple_test_json = test_data[0:100].to_json(orient='records')
    
    # When
    subject = make_prediction(input_data=multiple_test_json)
    
    # Then
    assert subject is not None
    assert len(subject.get('predictions')) == 3
    assert len(subject.get('predictions')['lstm']) == 100
    assert len(subject.get('predictions')['bilstm']) == 100
    assert len(subject.get('predictions')['attentionnet']) == 100
    
    
    