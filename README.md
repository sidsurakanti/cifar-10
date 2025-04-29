## Overview

CIFAR-10 and CIFAR-100 Classification using a custom CNN architecture. The model reached 89.2% and 65.3% accuracy respectively on the test sets.

<table>
  <thead>
    <tr>
      <td>CIFAR-10</td>
      <td>CIFAR-100</td>
    </tr>
  </thead>
  <tr>
    <td><img src="/cifar10_89.png" width="400px"></td>
    <td><img src="/cifar100_65.png" width="400px"></td>
  </tr>
</table>

## What I learned

- Convolution Operations
- Convolutional Neural Networks
- Batch norm
- Dealing with overfitting
- More about input normalization
- More about pytorch
  - Torch autograd engine
  - Data augumentation
- More about hyperparameter tuning

## Stack

![Python](https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-%23ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![Matplotlib](https://img.shields.io/badge/matplotlib-2067b8?style=for-the-badge&logo=matplotlib&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## Getting started

Play around by running the model locally.

### Prerequisites

- Python 3.10+
- pip
- PyTorch with CUDA

```bash
# check versions
python --version
pip --version
```

### Installation

1. Clone the repo

```bash
git clone https://github.com/sidsurakanti/cifar-10.git
cd /path/to/project/
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

## Roadmap

- [x] Working CIFAR-10 Classifer
- [x] Get 80% accuracy
- [x] Working CIFAR-100 Classifer
- [ ] Play around with ResNets
- [ ] Implement a CNN from scratch

## Contributing

Pull requests are welcome! Feel free to open an issue or suggestion.
