import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":
    test_data_id = "1bxClRIw9IzQWXozxsMjx7jqn729rTUDO"
    train_data_id = "1h9jMnlgKLBEv4D8MPJUU16el0r2BUIgA"
    validation_data_id = "1UcGlUFHY1Q7nxRj8DCQu2N6W7yByIQw3"
    tokenizer_id = "1q_Y72ARaEQHakZ6ubZ3nk7Ye2_hl5Fl8"
    embeddings_id = "1aLJZj6OkwouX5lPFKXsiqI5Dsya9SfO0"
    lstm_id = "1FGGI6uwkzZyuPXhE1fKI7IY03jaXCwQc"
    bilstm_id = "1rQj1t66OLM7tqB2Trasfcsr14Cbj5njt"
    attention_id = "1hKVKqnUWY7M6SET1zd9zVI8xKyPjCzSU"
    
    root = 'packages/toxicity_model/toxicity_model'
    
    test_data_dest = f'{root}/datasets/test.csv'
    train_data_dest = f'{root}/datasets/train.csv'
    validation_data_dest = f'{root}/datasets/validation.csv'

    embeddings_dest = f'{root}/embeddings/glove_embeddings.pkl'
    tokenizer_dest = f'{root}/tokenizers/tokenizer.pkl'

    lstm_dest = f'{root}/trained_models/state_dict_lstm0.1.0.pt'
    bilstm_dest = f'{root}/trained_models/state_dict_bilstm0.1.0.pt'
    attention_dest = f'{root}/trained_models/state_dict_attention0.1.0.pt'
    
    ids = [test_data_id, train_data_id, validation_data_id, tokenizer_id, embeddings_id, lstm_id, bilstm_id, attention_id]
    dests = [test_data_dest, train_data_dest, validation_data_dest, tokenizer_dest, embeddings_dest, lstm_dest, bilstm_dest, attention_dest]
    
    for file_id, dest in zip(ids, dests):
        download_file_from_google_drive(file_id, dest)