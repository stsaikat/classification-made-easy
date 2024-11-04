from train_tester import Trainer, Tester
from PIL import Image

trainer = Trainer()
trainer.train('dataset')

tester = Tester('output')
tester.test(Image.open('test.jpg'))