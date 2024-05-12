import fasttext
import fasttext.util
import sys
sys.path.insert(0, '/app2/config')

from config import get_local_config


model = fasttext.load_model(get_local_config()["FastText"]["path_full_model"])
print(model.get_dimension())
model = fasttext.util.reduce_model(model, 5) # 100 is the embedding dim.
print(model.get_dimension())
model.save_model(get_local_config()["FastText"]["path_reduced_model"])