from vit_pytorch.vit_train_test import VITTrainTest

class TrainTest:
    def __init__(self, arch_name = 'vit') -> None:
        self.module = None
        if arch_name == 'vit':
            self.module = VITTrainTest()
    def train(self, dataset_path):
        self.module.train(dataset_path)
    def test(self, image):
        return self.module.test(image)