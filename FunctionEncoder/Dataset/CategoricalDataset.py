from typing import Tuple, Union

import torch

from FunctionEncoder.Dataset.BaseDataset import BaseDataset


class CategoricalDataset(BaseDataset):

    def __init__(self,
                 input_range=(0,1),
                 n_categories=3,
                 logit_scale=5,
                 device:str="auto",
                 n_functions=None,
                 n_examples=None,
                 n_queries=None,

                 # deprecated arguments
                 n_functions_per_sample: int = None,
                 n_examples_per_sample: int = None,
                 n_points_per_sample: int = None,

                 ):
        if n_functions is None and n_functions_per_sample is None:
            n_functions = 10
        if n_examples is None and n_examples_per_sample is None:
            n_examples = 100
        if n_queries is None and n_points_per_sample is None:
            n_queries = 1000


        super().__init__(input_size=(1,),
                         output_size=(n_categories,),
                         data_type="categorical",
                         device=device,
                         n_functions=n_functions,
                         n_examples=n_examples,
                         n_queries=n_queries,

                         # deprecated arguments
                         total_n_functions=None,
                         total_n_samples_per_function=None,
                         n_functions_per_sample=n_functions_per_sample,
                         n_examples_per_sample=n_examples_per_sample,
                         n_points_per_sample=n_points_per_sample,
                         )
        self.n_categories = n_categories
        self.input_range = torch.tensor(input_range, device=self.device)
        self.logit_scale = logit_scale

    def states_to_logits(self, xs:torch.tensor, categories:torch.tensor, boundaries:torch.tensor, n_functions, n_examples,) -> torch.tensor:
        indexes = torch.stack([torch.searchsorted(b, x) for b, x in zip(boundaries, xs)]) # this is the index in the boundary list, need to convert it to index in the category list
        chosen_categories = torch.stack([c[i] for c, i in zip(categories, indexes)])
        logits = torch.zeros(n_functions, n_examples, self.n_categories, device=self.device)
        logits = logits.scatter(2, chosen_categories, 1)
        logits *= self.logit_scale
        return logits

    def sample(self) -> Tuple[  torch.tensor, 
                                torch.tensor, 
                                torch.tensor, 
                                torch.tensor, 
                                dict]:
        with torch.no_grad():

            # generate n_functions sets of coefficients
            boundaries = torch.rand((self.n_functions, self.n_categories-1), device=self.device)  * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
            boundaries = torch.sort(boundaries, dim=1).values

            # generate labels, each segment becomes a category
            categories = torch.stack([torch.randperm(self.n_categories, device=self.device) for _ in range(self.n_functions)])

            # now generate input data
            example_xs = torch.rand(self.n_functions, self.n_examples, 1, device=self.device) * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
            query_xs = torch.rand(self.n_functions, self.n_queries, 1, device=self.device) * (self.input_range[1] - self.input_range[0]) + self.input_range[0]

            # generate labels with high logits for the correct category and low logits for the others
            example_logits = self.states_to_logits(example_xs, categories, boundaries, self.n_functions, self.n_examples)
            logits = self.states_to_logits(query_xs, categories, boundaries, self.n_functions, self.n_queries)

            # create info dict
            info = {"boundaries": boundaries, "categories": categories}


        # the output for the first function should be chosen_categories[0][indexes[0]]
        return example_xs, example_logits, query_xs, logits, info