#!/bin/bash

python Examples/QuadraticExample.py --train_method least_squares --epochs 2
python Examples/QuadraticExample.py --train_method inner_product --epochs 2
python Examples/QuadraticExample.py --train_method least_squares --epochs 2 --residuals
python Examples/QuadraticExample.py --train_method inner_product --epochs 2 --residuals

python Examples/DonutExample.py --train_method least_squares --epochs 2
python Examples/DonutExample.py --train_method inner_product --epochs 2
python Examples/DonutExample.py --train_method least_squares --epochs 2 --residuals
python Examples/DonutExample.py --train_method inner_product --epochs 2 --residuals

python Examples/GaussianExample.py --train_method least_squares --epochs 2
python Examples/GaussianExample.py --train_method inner_product --epochs 2
python Examples/GaussianExample.py --train_method least_squares --epochs 2 --residuals
python Examples/GaussianExample.py --train_method inner_product --epochs 2 --residuals
 
python Examples/EuclideanExample.py --epochs 2 --n_basis 2
python Examples/EuclideanExample.py --epochs 2 --n_basis 3

python Examples/CategoricalExample.py --train_method inner_product --epochs 2
python Examples/CategoricalExample.py --train_method least_squares --epochs 2