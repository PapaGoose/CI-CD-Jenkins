import configparser
import os
import unittest
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from train import ResnetModel

config = configparser.ConfigParser()
config.read("config.ini")


class TestResnetModel(unittest.TestCase):

    def setUp(self) -> None:
        self.resnet_model = ResnetModel()

    def test_log_reg(self):
        self.assertEqual(self.resnet_model.resnet(use_config=False), True)

if __name__ == "__main__":
    unittest.main()
