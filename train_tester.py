from vit_pytorch.vit_train_test import VITTrainTest
import os
import shutil
import random

class TrainTest:
    def __init__(self, arch_name = 'vit') -> None:
        self.module = None
        if arch_name == 'vit':
            self.module = VITTrainTest()
    
    def prepare_dataset(self, dataset_path):
        tmp_data_path = 'tmp_data'
        os.makedirs(tmp_data_path, exist_ok=True)
        
        class_folders = [ f.name for f in os.scandir(dataset_path) if f.is_dir() ]
        print(class_folders)
        
        for class_name in class_folders:
            class_images = os.listdir(os.path.join(dataset_path, class_name))
            class_images = [file for file in class_images if file.endswith(('png', 'jpg', 'jpeg'))]
            
            total_images = len(class_images)
            print('found', total_images, 'images in', class_name)
            
            class_path = os.path.join(tmp_data_path, class_name)
            os.makedirs(class_path, exist_ok=True)
            train_path = os.path.join(class_path, 'train')
            os.makedirs(train_path, exist_ok=True)
            test_path = os.path.join(class_path, 'test')
            os.makedirs(test_path, exist_ok=True)
            
            train_total_images = int(total_images * 0.9)
            
            random.shuffle(class_images)
            for file in class_images[:train_total_images]:
                shutil.copy(
                    os.path.join(dataset_path, class_name, file),
                    os.path.join(train_path, file)
                )
            for file in class_images[train_total_images:]:
                shutil.copy(
                    os.path.join(dataset_path, class_name, file),
                    os.path.join(test_path, file)
                )
        return tmp_data_path
    
    def cleanup(self, dataset_path):
        shutil.rmtree(dataset_path)
    
    def train(self, dataset_path):
        data_path = self.prepare_dataset(dataset_path)
        self.module.train(data_path)
        self.cleanup(data_path)
        
    def test(self, image):
        return self.module.test(image)