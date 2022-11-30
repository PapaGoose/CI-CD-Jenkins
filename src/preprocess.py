import sys
import traceback

from torch.utils.data import Dataset, DataLoader
from logger import Logger
import cv2
import configparser
import os

# augmentation libs
import albumentations as A
from albumentations.pytorch import ToTensorV2

SHOW_LOG = True
TEST_SIZE = 0.2

class CatsVsDogsDataset(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # labeling
        if 'cat' in os.path.normpath(image_filepath).split(os.sep)[-1]:
            label = 1.0
        else:
            label = 0.0
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label

class DataMaker():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.project_path = os.path.join(os.getcwd(), "data")
        self.test_path = os.path.join(self.project_path, 'test')
        self.train_path = os.path.join(self.project_path, 'train')
        self.log.info("DataMaker is ready")

    def get_data(self) -> set:
        train_images = os.listdir(self.train_path)
        test_images = os.listdir(self.test_path)
        if not train_images:
            self.log.error('There is no train data')
            return ()
        if not test_images:
            self.log.error('There is no test data')
            return ()

        self.config['DATA'] = {
            'train': self.train_path,
            'test': self.test_path
        }
        train_images_path = [os.path.join(self.train_path, image) for image in train_images]
        test_images_path = [os.path.join(self.test_path, image) for image in test_images]
        return (train_images_path, test_images_path)

    def split_data(self, test_size=TEST_SIZE) -> set:
        train_images, test_images = self.get_data()
        valid_shape = int(len(train_images) * test_size)
        train_shape = len(train_images) - valid_shape
        train = train_images[:train_shape]
        valid = train_images[train_shape:]
        return (train, valid)

    def get_train_dataset(self) -> set:
        train, valid = self.split_data()

        train_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=160),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.RandomCrop(height=128, width=128),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        val_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=160),
                A.CenterCrop(height=128, width=128),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        train_dataset = CatsVsDogsDataset(images_filepaths=train, transform=train_transform)
        val_dataset = CatsVsDogsDataset(images_filepaths=valid, transform=val_transform)

        self.log.info("Train and validation data are ready")
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)

        return (train_dataset, val_dataset)

if __name__ == '__main__':
    data_maker = DataMaker()
    data_maker.get_train_dataset()
    
        


    