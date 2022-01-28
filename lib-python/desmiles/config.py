import os

# The default location of the data is DESMILES/data
# DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

if "DESMILES_DATA_DIR" in os.environ.keys():
    DATA_DIR = os.environ['DESMILES_DATA_DIR']
elif "DESMILES_DOWNLOAD_LOCATION" in os.environ.keys():
    DATA_DIR = os.path.join(os.environ['DESMILES_DOWNLOAD_LOCATION'], 'data')
else:    
    DATA_DIR = "/workspace/DESMILES-test/data"

# We need the data_dir to be an absolute path:
DATA_DIR = os.path.abspath(DATA_DIR)
    
# TODO: Jacob: check for existence of data/pretrained and fail with helpful error if not there.
    
# The path for the optimal model trained on the older molecules (train+val1)
MODEL_train_val1 = os.path.join(DATA_DIR, 'pretrained', 'model_2000_400_2000_5')


