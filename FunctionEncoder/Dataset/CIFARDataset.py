# from types import NoneType

import torch
import torchvision
from abc import abstractmethod
from typing import Tuple, Union

from torchvision.transforms import transforms

from FunctionEncoder import BaseDataset


class CIFARDataset(BaseDataset):
    def __init__(self,
                 logit_scale=5,
                 split="train",
                 heldout_classes=["apple", "bear", "castle", "dolphin", "crab", "hamster", "motorcycle", "plain", "snail", "willow_tree"],
                 heldout_classes_only=False,
                 device: str = "auto",
                 n_examples:int=100,
                 n_queries:int=100,
                 ):
        super(CIFARDataset, self).__init__(input_size=(3, 32, 32),
                                           output_size=(2,),
                                           data_type="categorical",
                                           device=device,
                                           dtype=torch.float32,
                                           n_examples=n_examples,
                                           n_queries=n_queries,
                                           )
        assert split.lower() in ["train", "test"], "split must be 'train' or 'test'"
        train = True if split.lower() == "train" else False
        cifar_dataset = torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transforms.ToTensor())
        self.classes = cifar_dataset.classes
        class_to_idx = cifar_dataset.class_to_idx

        # sort the images. This makes it easier to sample later for few-shot classification.
        sorted_datasets = []
        for class_name in self.classes:
            idx = class_to_idx[class_name]
            indices = torch.tensor(cifar_dataset.targets) == idx
            dataset = cifar_dataset.data[indices]
            dataset = torch.tensor(dataset, dtype=torch.float32)
            sorted_datasets.append(dataset)
        data_tensor = torch.stack(sorted_datasets, dim=0)
        self.data_tensor = data_tensor.to(self.device)
        self.data_tensor = self.data_tensor / 255.0
        self.data_tensor = self.data_tensor.permute(0, 1, 4, 3, 2) # FBWHC -> FBCHW

        # figure out which indicies we can sample from
        self.heldout_indicies = torch.tensor([self.classes.index(class_name) for class_name in heldout_classes], device=self.device, dtype=torch.int64)
        self.training_indicies = torch.tensor([i for i in range(len(self.classes)) if i not in self.heldout_indicies], device=self.device, dtype=torch.int64)

        # logit scale is used to specify how sharp the emprical distribution is. Higher values make the distribution sharper.
        self.logit_scale = logit_scale
        self.heldout_classes_only = heldout_classes_only

    @abstractmethod
    def __getitem__(self, index) -> Tuple[  torch.tensor,
                                                torch.tensor,
                                                torch.tensor,
                                                torch.tensor,
                                                dict]:
        with torch.no_grad():
            # convert the index from the acceptable classes to a global index
            if self.heldout_classes_only:
                index = self.heldout_indicies[index]
            else:
                index = self.training_indicies[index]


            # next, sample positive examples, ie images that belong to the class
            positive_example_xs, positive_example_class_indicies = self.sample_positive_examples(index, self.n_examples//2)
            positive_query_xs, positive_class_indicies = self.sample_positive_examples(index, self.n_queries//2)

            # next, sample negative examples, ie random images from other classes
            negative_example_xs, negative_example_class_indicies= self.sample_negative_examples(index, self.n_examples//2)
            negative_query_xs, negative_class_indicies = self.sample_negative_examples(index, self.n_queries//2)

            # concatenate the positive and negative examples
            example_xs = torch.cat([positive_example_xs, negative_example_xs], dim=0)
            query_xs = torch.cat([positive_query_xs, negative_query_xs], dim=0)

            # generate the ground truth labels
            example_ys = self.logit_scale * torch.ones((self.n_examples, 2), device=self.device)
            query_ys = self.logit_scale * torch.ones((self.n_queries, 2), device=self.device)
            example_ys[ :self.n_examples//2, 1] *= -1
            example_ys[ self.n_examples//2:, 0] *= -1
            query_ys[ :self.n_queries//2, 1] *= -1
            query_ys[ self.n_queries//2:, 0] *= -1

            # fetch relevant info for plotting
            info = {"class_idx": index, "class_labels": self.classes[index],
                    "positive_example_class_indicies": positive_example_class_indicies,
                    "positive_class_indicies": positive_class_indicies,
                    "negative_example_class_indicies": negative_example_class_indicies,
                    "negative_class_indicies": negative_class_indicies
                    }

            # return the data
            return example_xs, example_ys, query_xs, query_ys, info

    def __len__(self):
        if self.heldout_classes_only:
            return len(self.heldout_indicies)
        else:
            return len(self.training_indicies)

    # samples images randomly from the classes given
    def sample_positive_examples(self, class_index, count):
        example_indicies = torch.randperm(len(self.data_tensor[1]), device=self.device)[:count]
        example_indicies = example_indicies.reshape(*example_indicies.shape, 1, 1, 1).expand(-1, self.data_tensor.shape[-3], self.data_tensor.shape[-2], self.data_tensor.shape[-1])
        examples = self.data_tensor[class_index].gather(dim=0, index=example_indicies)
        return examples,  class_index

    # samples images randomly that ARENT from the classes given
    def sample_negative_examples(self, class_index, count):
        # get random indicies of classes
        acceptable_indicies = self.training_indicies[self.training_indicies != class_index]
        perm = torch.randint(0, len(acceptable_indicies), (count,), device=self.device)

        # get random indicies of images
        example_indicies =torch.randperm(len(self.data_tensor[1]), device=self.device)[:count]

        # get the images
        examples = self.data_tensor[perm, example_indicies]
        return examples, perm



