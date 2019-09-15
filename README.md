# gaia_tools

- [Click here](https://nbviewer.jupyter.org/github/lrbuechner/gaia_tools/blob/master/CNN.ipynb) to see the main project if CNN.ipynb doesn't load.

# Summary

This is an ongoing project to use statistical clustering algorithms and neural networks to detect new open clusters in the Milky Way. With over 1.3 billion stars cataloged, the European Space Agency's satellite, Gaia, has collected the most complete kinematic map of the local galaxy to date. Using clustering and classification algorithms such as DBSCAN and neural networks respectively, it is possible to extract individual star clusters from the galactic disk. 

In the gaia_tools module are a series of tools that allow for custom generation of training data for network training as well as some visualization tools to aid this process. After network training, I impliment DBSCAN on a volume of space to extract known clusters using astrometric data then test the accuracy of the model on known clusters and cluster candidates. I find 2 candidates with .90+ probability of being true clusters. 

Now the goal is to fine tune the model and develop a way to systematically inspect these candidates and their intereactions with their stellar neighborhood. All of this work is in anticipation of Gaia DR3 which will replace DR2 as the most complete/accurate astrometric catalouge and provide a unique opprotunity to discover new open clusters. 

# Extra
- [Visualization Demo](https://nbviewer.jupyter.org/github/lrbuechner/gaia_tools/blob/master/Visualization%20Demo.ipynb)
