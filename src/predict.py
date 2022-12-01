import configparser
import argparse
import os
import sys
import traceback
from torch.utils.data import Dataset, DataLoader
import cv2
from logger import Logger
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from datetime import datetime
import shutil
import yaml
import time
import base64
import numpy as np
# NN elements
import torch
from torch.utils.data import Dataset, DataLoader
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

class Predictor():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        self.parser = argparse.ArgumentParser(description="Predictor")

        self.parser.add_argument("-t",
                                 "--tests",
                                 type=str,
                                 help="Select tests",
                                 required=True,
                                 default="smoke",
                                 const="smoke",
                                 nargs="?",
                                 choices=["smoke", "func"])

        self.test_dataset = CatsVsDogsDataset(
            pd.read_csv(self.config['SPLITED DATA']['valid']),
            A.load(self.config['AUGMENTATION']['valid'])
        )
        self.log.info("Predictor is ready")

    def predict(self) -> bool:
        args = self.parser.parse_args()
        try:
            test_loader = DataLoader(
                self.test_dataset, batch_size=2, num_workers=0,
            )
            model = torch.load(self.config['RESNET50']["path"])
            model = model.to(self.config['RESNET50']["device"])
            criterion = nn.BCEWithLogitsLoss().to(self.config['RESNET50']["device"])
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        self.log.info("Model is ready")
        if args.tests == "smoke":
            try:
                losses = []
                model.eval()
                with torch.no_grad():
                    for image, label in test_loader:
                        output = model(image)
                        label = label.float().view(-1, 1)
                        loss = criterion(output, label)
                        losses.append(loss)
                self.log.info(f'BCE Loss = {sum(losses)/len(losses)}')
        
            except Exception:
                self.log.error(traceback.format_exc())
                sys.exit(1)
            self.log.info(f'{self.config["RESNET50"]["path"]} passed smoke tests')
        elif args.tests == "func":
            tests_path = os.path.join(os.getcwd(), "tests")
            exp_path = os.path.join(os.getcwd(), "experiments")
            for test in os.listdir(tests_path):
                with open(os.path.join(tests_path, test)) as f:
                    try:
                        data = json.load(f)
                        dec = base64.b64decode(data['X'])
                        nparr = np.frombuffer(dec, np.uint8)
                        X = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        X = cv2.resize(X, (128, 128))
                        X = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])(image=X)['image']
                        X = X[None, :,:,:]
                        y = data['y']
                        y = torch.Tensor([y]).unsqueeze(1)
                        losses = []
                        model.eval()
                        with torch.no_grad():
                            output = model(X)
                            loss = criterion(output, y)
                            losses.append(loss)
                        score = sum(losses)/len(losses)
                        print(f'{"RESNET50"} has {score} score')
                    except Exception:
                        self.log.error(traceback.format_exc())
                        sys.exit(1)
                    self.log.info(
                        f'{self.config["RESNET50"]["path"]} passed func test {f.name}')
                    exp_data = {
                        "model": 'RESNET50',
                        "model params": dict(self.config.items('RESNET50')),
                        "tests": args.tests,
                        "score": str(score),
                        "test path": self.config["SPLITED DATA"]["test"],
                    }
                    date_time = datetime.fromtimestamp(time.time())
                    str_date_time = date_time.strftime("%Y_%m_%d_%H_%M_%S")
                    exp_dir = os.path.join(exp_path, f'exp_{test[:6]}_{str_date_time}')
                    os.mkdir(exp_dir)
                    with open(os.path.join(exp_dir,"exp_config.yaml"), 'w') as exp_f:
                        yaml.safe_dump(exp_data, exp_f, sort_keys=False)
                    shutil.copy(os.path.join(os.getcwd(), "logfile.log"), os.path.join(exp_dir,"exp_logfile.log"))
                    shutil.copy(self.config['RESNET50']["path"], os.path.join(exp_dir,f'exp_RESNET50.pt'))
        
        return True

if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict()

        
