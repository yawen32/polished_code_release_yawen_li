# Polished Code Release
The code in fast_gradient_algorithm.py implements my own fast gradient descent algorithm to solve the l2 regularized logistic regression problems.

The file Demo_simple_dataset.py allows a user to launch the method on a simple simulated dataset, visualize the training process, and print the performance

The file Demo_real_dataset.py allows a user to launch the method on a a real-world dataset, visualize the training process, and print the performance.
This dataset can be found here:
```
https://statweb.stanford.edu/~tibs/ElemStatLearn/
```
The file Experimental_Comparison.py allows a user to run an experimental comparison between my own fast gradient descent algorithm and scikit-learn's on simulated dataset.

Also, there are some python packages need to be installed before users can run the files. 
```
import pandas
import numpy
import sklearn
```
Users can install these packages in their terminals by running the commands(for mac only):
```
easy_install pip
pip install numpy,pandas,scikit-learn
```
