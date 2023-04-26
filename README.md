# LayersRanking
Dataset of models for layer ranking evaluation. In this repository, you will find:
- [x] the raw data in `models_dataset.zip` which contains a fodler `Data` and one sub-folder per DNN architecture type and random perturbation distribution
- [x] a minimalist code base to load the models as a dataset iterator in `src`

The current version only supports tensorflow, which is the only requirement. We plan on providing support for torch as soon as possible.
The intended use is the following:

```python
from LayersRanking.src.TFLayersRanking import LayerRankingDataset
import os


dataset = LayerRankingDataset(data_path=os.path.join("LayersRanking","Data", "transfo_dirac"))
for models, rankings in dataset:
    models # is a list of tf.keras.Model
    rankings # for each models[i], the values rankings[i]
             # is the ranking from highest to least important
             # fully connected layer.
```