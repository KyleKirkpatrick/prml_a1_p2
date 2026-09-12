# Fashion-MNIST Logistic Regression

This project applies multinomial logistic regression to the Fashion-MNIST image dataset. It loads the IDX-formatted dataset through KaggleHub, flattens and normalises the 28 x 28 grayscale images, trains a scikit-learn model, and evaluates predictions with accuracy, a classification report, and a confusion matrix.

The project also compares L1 and L2 regularisation using cross-validation on training data. The final fitted model is saved with joblib under `models/`, using a filename based on the model type, solver, and iteration limit.

## Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- Kaggle access for downloading Fashion-MNIST through KaggleHub

## Setup

From the repository directory, run:

```sh
uv sync
```

This creates or updates the project virtual environment and installs the dependencies listed in `pyproject.toml`.

## Run the Model

Run the training and evaluation script with:

```sh
uv run python main.py
```

The script downloads Fashion-MNIST if it is not already cached, trains the model, prints evaluation results, and saves the fitted model to `models/`.

## Generate Report Figures

After a model has been saved, generate the report figures with:

```sh
uv run python resource/generate_report_figures.py
```

The figures are written to `report/figures/` and include training examples, a confusion matrix, and correct and incorrect prediction examples.

## Project Files

- `main.py`: loads data, trains and evaluates the model, and saves it.
- `data.py`: downloads and parses the Fashion-MNIST IDX files.
- `resource/generate_report_figures.py`: creates report figures from the saved model.
- `report/report.md`: project report.
- `report/`: assignment instructions, rubric, results, and generated figures.
