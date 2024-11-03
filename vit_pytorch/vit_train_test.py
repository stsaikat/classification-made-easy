import sys
import pathlib

if str(pathlib.Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

import train
import test

class VITTrainTest:
    def __init__(self) -> None:
        pass
    def train(self, dataset_path):
        train.main(dataset_path)
    def test(self, image):
        return test.main(image)