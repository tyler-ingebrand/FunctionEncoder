from typing import Tuple, Union

import torch

from FunctionEncoder.Dataset.BaseDataset import BaseDataset


class CategoricalDataset(BaseDataset):

    def __init__(self,
                 input_range=(0,1),
                 n_categories=3,
                 n_functions_per_sample:int = 10,
                 n_examples_per_sample:int = 1_000,
                 n_points_per_sample:int = 10_000,
                 logit_scale=5,
                 ):
        super().__init__(input_size=(1,),
                         output_size=(n_categories,),
                         total_n_functions=float('inf'),
                         total_n_samples_per_function=float('inf'),
                         data_type="categorical",
                         n_functions_per_sample=n_functions_per_sample,
                         n_examples_per_sample=n_examples_per_sample,
                         n_points_per_sample=n_points_per_sample,
                         )
        self.n_categories = n_categories
        self.input_range = input_range
        self.logit_scale = logit_scale

    def states_to_logits(self, xs:torch.tensor, categories:torch.tensor, boundaries:torch.tensor, n_functions, n_examples,) -> torch.tensor:
        indexes = torch.stack([torch.searchsorted(b, x) for b, x in zip(boundaries, xs)]) # this is the index in the boundary list, need to convert it to index in the category list
        chosen_categories = torch.stack([c[i] for c, i in zip(categories, indexes)])
        logits = torch.zeros(n_functions, n_examples, self.n_categories)
        logits = logits.scatter(2, chosen_categories, 1)
        logits *= self.logit_scale
        return logits

    def sample(self, device:Union[str, torch.device] ="auto") -> Tuple[ torch.tensor, 
                                                                torch.tensor, 
                                                                torch.tensor, 
                                                                torch.tensor, 
                                                                dict]:
        with torch.no_grad():
            device = device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
            n_functions = self.n_functions_per_sample
            n_examples = self.n_examples_per_sample
            n_points = self.n_points_per_sample

            # generate n_functions sets of coefficients
            boundaries = torch.rand((n_functions, self.n_categories-1))  * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
            boundaries = torch.sort(boundaries, dim=1).values

            # generate labels, each segment becomes a category
            categories = torch.stack([torch.randperm(self.n_categories) for _ in range(n_functions)])

            # now generate input data
            example_xs = torch.rand(n_functions, n_examples, 1) * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
            xs = torch.rand(n_functions, n_points, 1) * (self.input_range[1] - self.input_range[0]) + self.input_range[0]

            # generate labels with high logits for the correct category and low logits for the others
            example_logits = self.states_to_logits(example_xs, categories, boundaries, n_functions, n_examples)
            logits = self.states_to_logits(xs, categories, boundaries, n_functions, n_points)

            # convert device
            example_xs, example_ys, xs, ys = example_xs.to(device), example_logits.to(device), xs.to(device), logits.to(device)
            info = {"boundaries": boundaries, "categories": categories}


        # the output for the first function should be chosen_categories[0][indexes[0]]
        return example_xs, example_ys, xs, ys, info