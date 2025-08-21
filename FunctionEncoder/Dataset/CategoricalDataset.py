from typing import Tuple, Union

import torch

from FunctionEncoder.Dataset.BaseDataset import BaseDataset


class CategoricalDataset(BaseDataset):

    def __init__(self,
                 input_range=(0,1),
                 n_categories=3,
                 logit_scale=5,
                 device:str="auto",
                 n_examples=100,
                 n_queries=1000,
                 ):
        super().__init__(input_size=(1,),
                         output_size=(n_categories,),
                         data_type="categorical",
                         device=device,
                         n_examples=n_examples,
                         n_queries=n_queries,
                         )
        self.n_categories = n_categories
        self.input_range = torch.tensor(input_range, device=self.device)
        self.logit_scale = logit_scale
    def __len__(self):
        return 1000

    def states_to_logits(self, xs:torch.tensor, categories:torch.tensor, boundaries:torch.tensor, n_examples,) -> torch.tensor:
        indexes = torch.searchsorted(boundaries, xs)# this is the index in the boundary list, need to convert it to index in the category list
        chosen_categories = categories[indexes]
        logits = torch.zeros(n_examples, self.n_categories, device=self.device)
        logits = logits.scatter(1, chosen_categories, 1)
        logits *= self.logit_scale
        return logits

    def __getitem__(self, idx) -> Tuple[  torch.tensor,
                                torch.tensor, 
                                torch.tensor, 
                                torch.tensor, 
                                dict]:
        with torch.no_grad():

            # generate n_functions sets of coefficients
            boundaries = torch.rand((self.n_categories-1), device=self.device)  * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
            boundaries = torch.sort(boundaries, dim=0).values

            # generate labels, each segment becomes a category
            categories = torch.randperm(self.n_categories, device=self.device)

            # now generate input data
            example_xs = torch.rand( self.n_examples, 1, device=self.device) * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
            query_xs = torch.rand( self.n_queries, 1, device=self.device) * (self.input_range[1] - self.input_range[0]) + self.input_range[0]

            # generate labels with high logits for the correct category and low logits for the others
            example_logits = self.states_to_logits(example_xs, categories, boundaries,  self.n_examples)
            logits = self.states_to_logits(query_xs, categories, boundaries, self.n_queries)

            # create info dict
            info = {"boundaries": boundaries, "categories": categories}


        # the output for the first function should be chosen_categories[0][indexes[0]]
        return example_xs, example_logits, query_xs, logits, info