# FiGLearn

A tool that allows to infer a graph and a filter from observed data. The framework is very flexible and can thus be adapted to your particular problem. Some examples are shown in the examples folder. 

Note that the simulations as presented in our paper were done with an ancient version of the code. It can be found in the `legacy` branch. The application to the temperature data is presented as an example notebook.

## Code Structure

In the file `src/optimize.py` we implement the `FiGLearn` object, that you can use for inferring graphs, filters or missing values. In `src/helpers.py` we provide helper functions, in particular to convert between different representations of the graph. In `src/NNet.py` we define the neural network architecture for the graph filter. In `src/generators.py` we provide code with which some example graphs and filtered signals can be generated. 

## Tutorials

A comprehensive example of how to use the `FiGLearn` object can be found in the `examples/example_sbm.ipynb` jupyter notebook. There, we show how you can jointly or separately learn a filter and the graph, and how you can use a custom neural network architecture to include some domain knowledge about the nature of your filter. In the `examples/temperature.ipynb` we show a real-world example of graph and filter inference. We show how to use this to infer missing data if only very little information is available.

