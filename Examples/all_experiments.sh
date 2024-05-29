#!/bin/bash

python Examples/QuadraticExample.py --train_method least_squares --epochs 200
python Examples/QuadraticExample.py --train_method inner_product --epochs 200
python Examples/QuadraticExample.py --train_method least_squares --epochs 200 --residuals
python Examples/QuadraticExample.py --train_method inner_product --epochs 200 --residuals

python Examples/DonutExample.py --train_method least_squares --epochs 1000
python Examples/DonutExample.py --train_method inner_product --epochs 1000
python Examples/DonutExample.py --train_method least_squares --epochs 1000 --residuals
python Examples/DonutExample.py --train_method inner_product --epochs 1000 --residuals

python Examples/GaussianExample.py --train_method least_squares --epochs 1000
python Examples/GaussianExample.py --train_method inner_product --epochs 1000
python Examples/GaussianExample.py --train_method least_squares --epochs 1000 --residuals
python Examples/GaussianExample.py --train_method inner_product --epochs 1000 --residuals
 
python Examples/EuclideanExample.py --epochs 1000 --n_basis 2
python Examples/EuclideanExample.py --epochs 1000 --n_basis 3