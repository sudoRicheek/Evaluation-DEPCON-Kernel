#! /bin/bash

# The implemented simulation datasets.
# Uncomment any line and run this script
# to reproduce the desired simulation run

# # 100 runs on standard linear dataset
# python3 main.py -V 10 -K 6 -p 0.2 -r 100 -S 360 --PLOT      

# # 100 runs on linear dataset with additive 
# # noise sampled with the same mean for a 
# # particular DAG
# python3 main.py -V 10 -K 6 -p 0.2 -r 100 -S 360 --SAME_MEAN --PLOT 

# # 100 runs on linear dataset with some edges 
# # removed and some edges replaced with conn-
# # ections having zero linear correlation
# python3 main.py -V 10 -K 6 -p 0.2 -r 100 -S 360 --LINCORR_ZERO --PLOT 

# # 100 runs on linear dataset LINCORR_ZERO 
# # case combined with SAME_MEAN case
# python3 main.py -V 10 -K 6 -p 0.2 -r 100 -S 360 --LINCORR_ZERO --SAME_MEAN --PLOT 

# # 100 runs on non-linear dataset with
# # random functions generated from a 
# # gaussian process as the edge connections 
# python3 main.py -V 10 -K 6 -p 0.2 -r 100 -S 360 --NON_LINEAR --PLOT

# # 100 runs on linear dataset with DAGs
# # having the same Transitive Closure
# python3 main.py -V 10 -K 6 -p 0.2 -r 100 -S 360 --SAME_TC --PLOT             

# -x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-
# -x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-

# Generate Plots from data of completed 
# runs/quickly reproducible runs

# # Adjusted Rand Indices for standard linear case and 
# # LINCORR_ZERO case for all the implemented methods.
# # Append a --SAVE flag to the command below to save the 
# # generated plots.
# python3 reproducible/reproducible_plots.py -t 5

# # Plot a histogram of the \# cluster prediction
# # for the standard linear case. 
# # Append a --SAVE flag to the command below to save the 
# # generated plots.
# python3 reproducible/reproducible_plots.py -t 4

# # Cluster search over a single linear dataset
# # using Calinski-Harabasz score as the unsup-
# # ervised metric.
# # Append a --SAVE flag to the command below to save the 
# # generated plots.
# python3 reproducible/reproducible_plots.py -t 3

# # Plot the Linear and Kernel (with DEP-CON Kernel)
# # PCA for a random linear dataset
# python3 reproducible/reproducible_plots.py -t 1