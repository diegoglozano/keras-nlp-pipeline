from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense


def create_model(self_vocab_size, self_embedding_weights):
    model = Sequential([
        Embedding(input_dim=self_vocab_size,
                  output_dim=300,
                  weights=[self_embedding_weights],
                  trainable=False,
                  mask_zero=True),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model
