from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense


def create_model(embedding_input_dim, embedding_output_dim, embedding_weights):
    model = Sequential([
        Embedding(input_dim=embedding_input_dim,
                  output_dim=embedding_output_dim,
                  weights=[embedding_weights],
                  trainable=False,
                  mask_zero=True),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model
