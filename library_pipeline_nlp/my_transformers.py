from sklearn.base import BaseEstimator, TransformerMixin

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class TokenizerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.tokenizer = Tokenizer()

    def fit(self, X, y=None):
        self.tokenizer.fit_on_texts(X)
        return self

    def transform(self, X, y=None):
        X_transformed = self.tokenizer.texts_to_sequences(X)
        return X_transformed


class PadSequencesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, maxlen):
        self.maxlen = maxlen

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_padded = pad_sequences(X, maxlen=self.maxlen)
        return X_padded

