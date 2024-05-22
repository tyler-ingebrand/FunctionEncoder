#!/bin/bash

python Examples/QuadraticExample.py --train_method least_squares
python Examples/QuadraticExample.py --train_method inner_product

python Examples/DonutExample.py --train_method least_squares
python Examples/DonutExample.py --train_method inner_product

python Examples/GaussianExample.py --train_method least_squares
python Examples/GaussianExample.py --train_method inner_product
