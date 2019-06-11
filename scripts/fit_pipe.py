import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from gensim.models import KeyedVectors
from keras.wrappers.scikit_learn import KerasClassifier

from library_pipeline_nlp.my_transformers import TokenizerTransformer, TokenizerTransformerInherit, PadSequencesTransformer
from library_pipeline_nlp.my_model import create_model


# PATHS
DATA_PATH = '../data/texts.csv'
EMBEDDING_PATH = '../data/embedding.bin'
PIPE_PATH = '../objects/pipeline.joblib'

# DATA
df = pd.read_csv(DATA_PATH, index_col='id', nrows=100)

# X / y
X = df['tweet'].copy()
y = df['label'].copy()

# DIMENSIONS
longest_text = X.str.split().str.len().max()  # Para padsequences()
EMBEDDING_DIM = 300  # Dimensión Word2Vec

# PRETRAINED WORD2VEC
word_vectors = KeyedVectors.load_word2vec_format(EMBEDDING_PATH, limit=1000, binary=True)

# VOCABULARY SIZE (+1)
vocab_size = len(word_vectors.vocab.keys()) + 1
embedding_weights = np.vstack([np.zeros(word_vectors.vectors.shape[1]), word_vectors.vectors])


# DECLARE ESTIMATORS
my_tokenizer = TokenizerTransformer()
my_padder = PadSequencesTransformer(maxlen=longest_text)
my_model = KerasClassifier(build_fn=create_model,
                           epochs=2,
                           embedding_input_dim=vocab_size,
                           embedding_output_dim=300,
                           embedding_weights=embedding_weights)

# DECLARE PIPELINE
pipeline = Pipeline([
    ('tokenizer', my_tokenizer),
    ('padder', my_padder),
    ('model', my_model)
])

# TRAIN
pipeline.fit(X, y)

# DUMP OBJECT
joblib.dump(pipeline, PIPE_PATH)
