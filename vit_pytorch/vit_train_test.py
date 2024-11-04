import sys
import pathlib

if str(pathlib.Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

import train
import test

class VITTrainTest:
    def __init__(self) -> None:
        pass
    def train(self, dataset_path, outfolder = None):
        train.main(dataset_path, output_path=outfolder)
    def test(self, image, pretrained_folder):
        return test.main(image, pretrained_folder)