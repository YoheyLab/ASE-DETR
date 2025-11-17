import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ASE-DETR.yaml')
    model.train(data='data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=4,
                workers=4,
                device='0',
                project='runs/train',
                name='exp',
                )