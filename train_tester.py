from vit_pytorch.vit_train_test import VITTrainTest
import os
import shutil
import random
import json

class TrainTest:
    def __init__(self, arch_name = 'vit') -> None:
        self.out_folder = 'output'
        os.makedirs(self.out_folder, exist_ok=True)
        self.arch_name = arch_name
        self.module = None
        if self.arch_name == 'vit':
            self.module = VITTrainTest()
        self.write_to_config('arch', self.arch_name)
    
    def prepare_dataset(self, dataset_path):
        tmp_data_path = 'tmp_data'
        os.makedirs(tmp_data_path, exist_ok=True)
        
        class_folders = [ f.name for f in os.scandir(dataset_path) if f.is_dir() ]
        class_folders.sort()
        self.write_to_config('classes', class_folders)
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
    
    def write_to_config(self, key, value):
        config_path = os.path.join(self.out_folder, 'config.json')
        config = {}
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
        except:
            pass
        
        config[key] = value
        
        with open(config_path, 'w+') as file:
            json.dump(config, file, indent=4)
    
    def train(self, dataset_path):
        data_path = self.prepare_dataset(dataset_path)
        self.module.train(data_path, outfolder=self.out_folder)
        self.cleanup(data_path)
        
    def test(self, image):
        return self.module.test(image)