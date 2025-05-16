# CIFAR-10 Image Classification Project

This repository contains the code for a CIFAR-10 image classification project using PyTorch. The project aims to evaluate the impact of data and label perturbations on model performance, including baseline, random label shuffle, label noise, and input perturbations.

## Project Structure

```
CIFAR-10-Classification/
├── data/                  # CIFAR-10 dataset (downloaded automatically)
├── results/               # Generated results (accuracy and loss plots)
├── src/                   # Source code for training and evaluation
│   └── main.py           # Main training and evaluation script
├── README.md              # Project overview and instructions
└── requirements.txt       # Required Python packages
```

## Requirements

Ensure you have the following packages installed:

```bash
pip install -r requirements.txt
```

## Usage

To train the model and generate results, simply run:

```bash
python src/main.py
```

The results will be saved in the `results/` directory as:
- `accuracy_trends.png` - Training and validation accuracy over epochs
- `loss_trends.png` - Training and validation loss over epochs

## Data

The CIFAR-10 dataset will be downloaded automatically to the `data/` directory.

## Model

A simple convolutional neural network (CNN) is used for classification, with the following architecture:
- 3 convolutional layers
- 2 fully connected layers
- ReLU activation and max pooling

## Results

The generated figures include:
- Training and Validation Accuracy Trends
- Training and Validation Loss Trends

## License

This project is licensed under the MIT License.

## Contact

For questions or collaboration, please reach out at [your.email@domain.com].
