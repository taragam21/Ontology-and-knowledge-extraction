# Ontology-and-knowledge-extraction

# Animal Classification Project

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact Information](#contact-information)

## Overview
This project uses Machine Learning and Ontology-based methods for classifying images of different animal classes. Using Python, SPARQL, OWLready2, TensorFlow, and Scikit-Learn, we train models on features extracted from the images, and use them to predict the class of new images.

## Prerequisites
- Python 3.x
- TensorFlow 2.x
- Scikit-Learn
- SPARQLWrapper
- OWLready2

## Installation and Setup
1. Clone the GitHub repository: `git clone https://github.com/username/repository.git`
2. Navigate to the project directory: `cd repository`
3. Install the required packages: `pip install -r requirements.txt`

## Usage
1. Train the model: Run `python projet.py` to train the model using the data in the `animals` directory.
2. Test the model: Use `python test.py` to test the trained model with new images in the `validation_images` directory.

## Code Structure
- `projet.py`: This file contains the main logic of the project. It handles creating the ontology, training the machine learning models, and making predictions on the test data.
- `test.py`: This script is used to test the trained models with new images and display the results along with the images.
- `animals`: This directory contains the training data, i.e., images of different animal classes.
- `validation_images`: This directory contains the testing data, i.e., images to test the trained models.

## Results
The models were able to classify the animal classes with high accuracy. They were trained on features extracted from the images using the InceptionV3 model pretrained on ImageNet.

