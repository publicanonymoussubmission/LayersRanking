from typing import Any, Tuple, List, Union
import tensorflow as tf
from .CustomTFObjects import DensePerturbated, StochasticDepth
import numpy as np
import os


def unit_load_data(x: str, y: str) -> Tuple[tf.keras.Model, np.ndarray]:
    """
    retrives the model and corresponding label based on source paths

    Args:
        x: path to the saved model
        y: path to the saved labels
    """
    model_path = x
    model = tf.keras.models.load_model(filepath=model_path, compile=False)
    label_path = y
    label = np.load(label_path)
    return model, label


def convert_score_to_ranking(score: List[float]) -> List[int]:
    """
    given scores we return the ranking

    Args:
        score: score per layer to convert
    """
    return np.argsort(score)[::-1]


class LayerRankingDataset:
    def __init__(
        self,
        data_path: str,
        batch_size: int = 1,
        return_ranks: bool = True,
        *args: Any,
        **kwds: Any
    ) -> None:
        self.return_ranks = return_ranks
        self.data_path = data_path
        self.low = 0
        self.current = 0
        self.models, self.labels = self.get_models_and_labels()
        self.high = len(self.models)
        self.batch_size = batch_size

    def get_models_and_labels(
        self,
    ) -> Tuple[List[str], List[str]]:
        list_inputs = [
            os.path.join(self.data_path, e)
            for e in os.listdir(self.data_path)
            if ".h5" in e
        ]
        list_labels = [
            os.path.join(self.data_path, e)
            for e in os.listdir(self.data_path)
            if ".npy" in e
        ]
        list_inputs.sort()
        list_labels.sort()
        return list_inputs, list_labels

    def convert_to_rank(self, y: np.ndarray) -> np.ndarray:
        ranking = convert_score_to_ranking(y)
        return ranking

    def __iter__(self) -> None:
        return self

    def get_next(self) -> Tuple[List[tf.keras.Model], List[np.ndarray]]:
        output_x = []
        output_y = []
        for cpt in range(self.batch_size):
            if self.current + cpt < len(self.models):
                x, y = unit_load_data(
                    x=self.models[self.current + cpt], y=self.labels[self.current + cpt]
                )
                output_x.append(x)
                if self.return_ranks:
                    y = self.convert_to_rank(y=y)
                output_y.append(y)
        return output_x, output_y

    def __next__(self) -> Union[None, Tuple[List[tf.keras.Model], List[np.ndarray]]]:
        if self.current < self.high:
            output = self.get_next()
            self.current += self.batch_size
            return output
        self.current = 0
        raise StopIteration


if __name__ == "__main__":
    dataset = LayerRankingDataset(
        data_path=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "Data", "transfo_dirac"
        )
    )
    for x, y in dataset:
        x[0].summary()
        break
    dataset = LayerRankingDataset(
        data_path=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "Data", "skip_co_gaussian"
        )
    )
    for x, y in dataset:
        x[0].summary()
        break
    dataset = LayerRankingDataset(
        data_path=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "Data", "sto_depth_uniform"
        )
    )
    for x, y in dataset:
        x[0].summary()
        break
    dataset = LayerRankingDataset(
        data_path=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "Data", "dirac"
        )
    )
    for x, y in dataset:
        x[0].summary()
        break
    # python -m src.TFLayersRanking
