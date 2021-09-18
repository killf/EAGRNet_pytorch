import torch
from torchvision import transforms

from utils.transforms import *


class Config:
    def __init__(self, task_id=1):
        self.task_id = task_id
        self.output_dir = "output"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_device = 1

        self.num_classes = 19
        self.dataset = "ImageFolder"
        self.data_dir = "/data/face/parsing/dataset/CelebAMask-HQ_processed2"
        self.sample_dir = "/data/face/parsing/dataset/testset_210720_aligned"
        self.image_size = (512, 512)
        self.crop_size = (473, 473)
        self.do_val = True

        self.lr = 1e-3
        self.batch_size = 2
        self.epochs = 100

        self.train_transform = Compose([RandomHorizontalFlip(),
                                        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                        RandomScale((0.75, 1.25)),
                                        RandomRotation(),
                                        RandomCrop(self.crop_size),
                                        ToTensor(),
                                        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.val_transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.model_name = "EAGRNet"
        self.model_args = Dict()

        self.loss_name = "CriterionCrossEntropyEdgeParsing"
        self.loss_args = Dict()

        self.scheduler_name = "LinearLR"
        self.scheduler_args = Dict()

        self.optimizer_name = "SGD"
        self.optimizer_args = Dict(
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

    def build(self, num_classes=None, **kwargs):
        if num_classes is not None:
            self.num_classes = num_classes

        if "data_count" in kwargs:
            data_count = kwargs["data_count"]
            if self.scheduler_name == "LinearLR":
                self.scheduler_args.learning_rates = [1e-7, self.lr, 1e-7]
                self.scheduler_args.milestones = [data_count, data_count * self.epochs]

        return self


class Dict(dict):
    def __getattr__(self, item):
        return self.get(item, None)

    def __setattr__(self, key, value):
        self[key] = value
