import configparser
import os
import sys
import traceback
from torch.utils.data import Dataset, DataLoader
import cv2
from logger import Logger
import pandas as pd
import albumentations as A

# NN elements
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.nn as nn

SHOW_LOG = True

class CatsVsDogsDataset(Dataset):
    def __init__(self, df_images, transform=None):
        self.images_filepaths = df_images['image_path'].values
        self.images_labels = df_images['label'].values
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image = cv2.imread(self.images_filepaths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.images_labels[idx]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


class ResnetModel():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        self.experiments_path = os.path.join(os.getcwd(), 'experiments')
        self.model_path = os.path.join(self.experiments_path, 'model.pt')

        self.train_dataset = CatsVsDogsDataset(
            pd.read_csv(self.config['SPLITED DATA']['train']), 
            A.load(self.config['AUGMENTATION']['train'])
        )
        self.valid_dataset = CatsVsDogsDataset(
            pd.read_csv(self.config['SPLITED DATA']['valid']), 
            A.load(self.config['AUGMENTATION']['valid'])
        )
    
    def resnet(self, use_config: bool, device='cpu', lr=0.001, epochs=2, num_workers=0, batch_size=4, predict=False) -> bool:
        if use_config:
            try:
                device = self.config['RESNET50']["device"]
                lr = self.config.getfloat('RESNET50', 'lr')
                epochs = self.config.getint('RESNET50', 'epochs')
                batch_size = self.config.getint('RESNET50', "batch_size")
                num_workers = self.config.getint('RESNET50', "num_workers")
            except KeyError:
                self.log.error(traceback.format_exc())
                self.log.warning(f'Using config:{use_config}, no params')
                sys.exit(1)
    
        model = getattr(models, 'resnet50')(weights=False, num_classes=1,)
        model = model.to(device)
        criterion = nn.BCEWithLogitsLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.log.info('Model loaded')

        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        val_loader = DataLoader(
            self.valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )
        self.log.info('Training started ...')
        try:
            for epoch in range(1, epochs + 1):
                model.train()
                for images, target in train_loader:
                    images = images
                    target = target.float().view(-1, 1)
                    output = model(images)
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                self.log.info(f'Epoch {epoch} done')
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        if predict:
            model.eval()
            losses = []
            with torch.no_grad():
                for images, target in val_loader:
                    images = images
                    target = target.float().view(-1, 1)
                    output = model(images)
                    loss = criterion(output, target)
                    losses.append(loss)
            self.log.info(f'BCE Loss = {sum(losses)/len(losses)}')
        
        params = {
            'device': device,
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'path': self.model_path
        }
        return self.save_model(model, self.model_path, "RESNET50", params)

    def save_model(self, model, path: str, name: str, params: dict) -> bool:
        self.config[name] = params
        os.remove('config.ini')
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        torch.save(model, path)
        self.log.info(f'{path} is saved')
        return os.path.isfile(path)

if __name__ == "__main__":
    multi_model = ResnetModel()
    multi_model.resnet(use_config=False, predict=True)