import sys
import traceback
from random import shuffle
import pandas as pd
from logger import Logger
import configparser
import os

# augmentation libs
import albumentations as A
from albumentations.pytorch import ToTensorV2

SHOW_LOG = True
TEST_SIZE = 0.2


class DataMaker():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.project_path = os.path.join(os.getcwd(), "data")
        self.test_path = os.path.join(self.project_path, 'test')
        self.train_path = os.path.join(self.project_path, 'train')
        self.splited_path = os.path.join(self.project_path, 'splited')
        self.transforms_path = os.path.join(self.project_path, 'transforms')
        self.log.info("DataMaker is ready")

    def get_data(self) -> set:
        train_images = os.listdir(self.train_path)
        test_images = os.listdir(self.test_path)
        if not train_images:
            self.log.error('There is no train data')
            return False
        if not test_images:
            self.log.error('There is no test data')
            return False

        self.config['DATA'] = {
            'train': self.train_path,
            'test': self.test_path
        }

        self.train_images_path = [os.path.join(self.train_path, image) for image in train_images]
        self.test_images_path = [os.path.join(self.test_path, image) for image in test_images]

        shuffle(self.train_images_path)
        return True

    def split_data(self, test_size=TEST_SIZE) -> set:
        try:
            self.get_data()
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        
        valid_shape = int(len(self.train_images_path) * test_size)
        train_shape = len(self.train_images_path) - valid_shape

        # train = train_images[:train_shape]
        train = self.train_images_path[:50]
        train_dict = {
            'image_path': train,
            'label': [1 if 'cat' in os.path.normpath(image).split(os.sep)[-1] else 0 for image in train]
        }
        train_csv_path = os.path.join(self.splited_path, 'train.csv')
        pd.DataFrame(train_dict).to_csv(train_csv_path, index=False)
        self.log.info(f'{train_csv_path} is saved')

        # valid = train_images[train_shape:]
        valid = self.train_images_path[50:70]
        valid_dict = {
            'image_path': valid,
            'label': [1 if 'cat' in os.path.normpath(image).split(os.sep)[-1] else 0 for image in valid]
        }
        valid_csv_path = os.path.join(self.splited_path, 'valid.csv')
        pd.DataFrame(valid_dict).to_csv(valid_csv_path, index=False)
        self.log.info(f'{valid_csv_path} is saved')

        test_csv_path = os.path.join(self.splited_path, 'test.csv')
        pd.DataFrame({'image_path': self.test_images_path[:5]}).to_csv(test_csv_path, index=False)
        self.log.info(f'{test_csv_path} is saved')

        self.config['SPLITED DATA'] = {
            'train': train_csv_path,
            'valid': valid_csv_path,
            'test': test_csv_path
        }

        self.save_augmentation()

        self.log.info("Train/validation/test data are ready")
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
    
        return os.path.isfile(train_csv_path) and os.path.isfile(valid_csv_path) and \
            os.path.isfile(test_csv_path)
    
    def save_augmentation(self) -> bool:
        try:
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
            train_transform_path = os.path.join(self.transforms_path, 'train.json')
            A.save(train_transform, train_transform_path)\

            val_transform = A.Compose(
                [
                    A.SmallestMaxSize(max_size=160),
                    A.CenterCrop(height=128, width=128),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
            valid_transform_path = os.path.join(self.transforms_path, 'valid.json')
            A.save(val_transform, valid_transform_path)

            test_transform = A.Compose(
                [
                    A.SmallestMaxSize(max_size=160),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2() 
                ]
            )
            test_transform_path = os.path.join(self.transforms_path, 'test.json')
            A.save(test_transform, test_transform_path)

            self.config['AUGMENTATION'] = {
                'train': train_transform_path,
                'valid': valid_transform_path,
                'test': test_transform_path
            }
            self.log.info("Augmentation saved")
            return True
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        

if __name__ == '__main__':
    data_maker = DataMaker()
    data_maker.split_data()
    
        


    