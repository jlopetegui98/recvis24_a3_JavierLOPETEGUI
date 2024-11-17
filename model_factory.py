"""Python file to instantite the model and the transform that goes with it."""

from data import data_transforms, resnet_transforms, data_aug_transforms
from model import Net
from resnet import ResnetClf


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "resnet":
            return ResnetClf()
        elif self.model_name == "dinov2_clf":
            return Dinov2CLF()
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms, data_transforms
        if self.model_name == "resnet" or self.model_name == "dinov2":
            return resnet_transforms, resnet_transforms
        elif self.model_name == "dinov2_aug":
            return data_aug_transforms, resnet_transforms
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
