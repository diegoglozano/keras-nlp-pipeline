import joblib

from library_pipeline_nlp.my_transformers import TokenizerTransformer, PadSequencesTransformer
from library_pipeline_nlp.my_model import create_model


# PATHS
PIPE_PATH = '../objects/pipeline.joblib'

# LOAD PIPELINE
pipeline = joblib.load(PIPE_PATH)

# PREDICT NEW TEXT
preds = pipeline.predict(['This is a new text'])
print(preds)
