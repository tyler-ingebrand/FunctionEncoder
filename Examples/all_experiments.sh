#!/bin/bash
# define a num epochs
epochs=100
python Examples/QuadraticExample.py --train_method least_squares --grad $epochs
python Examples/QuadraticExample.py --train_method inner_product --grad $epochs
python Examples/QuadraticExample.py --train_method least_squares --grad $epochs --residuals
python Examples/QuadraticExample.py --train_method inner_product --grad $epochs --residuals
python Examples/QuadraticExample.py --train_method least_squares --grad $epochs --parallel
python Examples/QuadraticExample.py --train_method inner_product --grad $epochs --parallel
python Examples/QuadraticExample.py --train_method least_squares --grad $epochs --residuals --parallel
python Examples/QuadraticExample.py --train_method inner_product --grad $epochs --residuals --parallel

python Examples/DonutExample.py --train_method least_squares --grad $epochs
python Examples/DonutExample.py --train_method inner_product --grad $epochs
python Examples/DonutExample.py --train_method least_squares --grad $epochs --residuals
python Examples/DonutExample.py --train_method inner_product --grad $epochs --residuals

python Examples/GaussianExample.py --train_method least_squares --grad $epochs
python Examples/GaussianExample.py --train_method inner_product --grad $epochs
python Examples/GaussianExample.py --train_method least_squares --grad $epochs --residuals
python Examples/GaussianExample.py --train_method inner_product --grad $epochs --residuals

python Examples/EuclideanExample.py --grad $epochs --n_basis 2
python Examples/EuclideanExample.py --grad $epochs --n_basis 3

python Examples/CategoricalExample.py --train_method inner_product --grad $epochs


python Examples/VanDerPolExample.py  --grad $epochs

python Examples/CIFARExample.py --grad $epochs