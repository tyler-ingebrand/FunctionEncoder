#!/bin/bash
# define a num epochs
epochs=1000
#python Examples/QuadraticExample.py --train_method least_squares --epochs $epochs
#python Examples/QuadraticExample.py --train_method inner_product --epochs $epochs
#python Examples/QuadraticExample.py --train_method least_squares --epochs $epochs --residuals
#python Examples/QuadraticExample.py --train_method inner_product --epochs $epochs --residuals
#python Examples/QuadraticExample.py --train_method least_squares --epochs $epochs --parallel
#python Examples/QuadraticExample.py --train_method inner_product --epochs $epochs --parallel
#python Examples/QuadraticExample.py --train_method least_squares --epochs $epochs --residuals --parallel
#python Examples/QuadraticExample.py --train_method inner_product --epochs $epochs --residuals --parallel

#python Examples/DonutExample.py --train_method least_squares --epochs $epochs
#python Examples/DonutExample.py --train_method inner_product --epochs $epochs
#python Examples/DonutExample.py --train_method least_squares --epochs $epochs --residuals
#python Examples/DonutExample.py --train_method inner_product --epochs $epochs --residuals

#python Examples/GaussianExample.py --train_method least_squares --epochs $epochs
#python Examples/GaussianExample.py --train_method inner_product --epochs $epochs
#python Examples/GaussianExample.py --train_method least_squares --epochs $epochs --residuals
#python Examples/GaussianExample.py --train_method inner_product --epochs $epochs --residuals

#python Examples/EuclideanExample.py --epochs $epochs --n_basis 2
#python Examples/EuclideanExample.py --epochs $epochs --n_basis 3

#python Examples/CategoricalExample.py --train_method inner_product --epochs $epochs
#python Examples/CategoricalExample.py --train_method least_squares --epochs $epochs

python Examples/CIFARExample.py --epochs $epochs