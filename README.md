# PALMS image partitioning
PALMS is originally a Matlab/C++ toolbox for image partitioning 
using the piecewise affine-linear Mumford-Shah model (also known as "affine-linear Potts model"), 
see https://github.com/lu-kie/PALMS_ImagePartitioning for the original source code and references 
below for the creators. 

This is a wrapper for Python, which can also handle pointcloud data, i.e., solve the model on "stripes" through a pointcloud.

## Installation
To use the wrapper, you need to install OpenMP and Armadillo.

For MacOS:
```
brew install libomp armadillo
```

For Linux: 
```
sudo apt-get install libarmadillo-dev
sudo apt-get install libomp-dev
```

## References
- L. Kiefer, M. Storath, A. Weinmann.
    "An efficient algorithm for the piecewise affine-linear Mumford-Shah model based on a Taylor jet splitting."
    IEEE Transactions on Image Processing, 2020.
- L. Kiefer, M. Storath, A. Weinmann.
    "PALMS image partitioning – a new parallel algorithm for the piecewise affine-linear Mumford-Shah model."
     Image Processing On Line, 2020.
