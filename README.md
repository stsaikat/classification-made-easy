# classification-made-easy

a library to make your classification train, validation, test easier

## Installation
Create python virutal enviroment and install all requirements.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Dataset structure

```
dataset/
│
├── dogs/
│   ├── dog1.jpg
│   ├── dog2.jpg
│   ├── dog3.jpg
│   └── ... (more dog images)
│
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   ├── cat3.jpg
│   └── ... (more cat images)
│
├── birds/
│   ├── bird1.jpg
│   ├── bird2.jpg
│   ├── bird3.jpg
│   └── ... (more bird images)
│
└── ... (additional animal categories as needed)
```

### Training

```python
from train_tester import Trainer

trainer = Trainer()

# starts your training and save pretrain model to output folder
trainer.train('dataset')
```

### Testing

```python
from train_tester import Tester
from PIL import Image

# here 'output' is the pretrained model path
tester = Tester('output')
test_image = Image.open('test.jpg')
class_name = tester.test(test_image)
```

## Credits
- [vit-pytorch](https://github.com/lucidrains/vit-pytorch) for their awsome implementation of Vit model in pytorch.