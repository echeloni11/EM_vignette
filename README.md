# EM Algorithm Clear Explained

This vignette repository contains four files.

## EM_vignette.pdf

This contains an in-depth explanation of the derivaton of EM algorithm, with two examples modeled from biological problem scRNA-seq. 


## BoundOptimization.mp4 and visualize.py

The mp4 file is an animated visualization of the process of bound optimization. It is generated by visualize.py. \\
If you want to manipulate visualize.py, you need to install manim
```
pip install manim
```
and run 
```
manim -pql visualize.py BoundOptimization
```

## examples.ipynb

This notebook file contains simple codes that implement EM algorithm for the examples mentioned in the pdf. One of them uses a real world dataset PBMC 3k dataset.

To run this notebook, your environment would need some common packages including numpy, matplotlib, pandas and scikit-learn.

