# Geolocalization Project

## Overview

This project focuses on visual place recognition and geolocalization using deep learning techniques. It implements and experiments with various models, loss functions, and training strategies to improve the accuracy of place recognition across different datasets.

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
4. [Experiments](#experiments)
5. [Datasets](#datasets)
6. [Models](#models)
7. [Evaluation](#evaluation)

## Installation

To set up the project environment, follow these steps:

1. Clone the repository:

2. Create a virtual environment (optional but recommended):
   ```
   conda create --name geolocalization --file requirements.txt
   conda activate geolocalization
   ```

## Project Structure

The project is organized as follows:

- `experiments.ipynb`: Main Jupyter notebook for running experiments
- `models/`: Directory containing model architectures
- `datasets/`: Custom dataset classes for training and testing
- `utils/`: Utility functions, including custom learning rate schedulers
- `train.py`: Training script
- `evaluation.py`: Evaluation script
- `weights/`: Directory to store trained model weights

## Usage

To run experiments and train models, use the `experiments.ipynb` notebook. This notebook contains various experimental setups and configurations for training and evaluating models.

## Experiments

The project includes several experiments with different configurations. Some of them are:

1. Average Pooling + Contrastive Loss
2. GEM Pooling + Contrastive Loss
3. Batch Hard Miner + Contrastive Loss
4. DistanceWeightedMiner + SupConLoss
5. PairMargin Miner + FastAP Loss
6. MultiSimilarity Loss

Each experiment can be run by setting the `TRAIN` variable to `True` and executing the corresponding cell in the notebook.

## Datasets

The project uses the following datasets:

- GSVCitiesDataset: For training
- TestDataset: For validation and testing
- SF_XS Dataset: For testing on San Francisco data
- Tokyo Dataset: For testing on Tokyo data

Dataset loading and preprocessing are handled in the `datasets/` directory.

## Models

The main model architecture is defined in the `models/helper.py` file. It uses a truncated ResNet18 as the backbone with various pooling options (average, GEM).

## Evaluation

Model evaluation is performed using the `eval_model` function from `evaluation.py`. It calculates metrics such as Recall@1 and Recall@5 for the trained models on different test sets.

