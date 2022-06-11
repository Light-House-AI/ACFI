# Arabic Calligraphy Font Identification

This font identification system is able to identify the font for a given Arabic text snippet.
It proves very useful in a wide range of applications, such as the fields of graphic design
like in illustrator arts and optical character recognition for digitizing hardcover documents.

Not limited to that, but it also raised interest in the analysis of historical texts authentic manuscript
verification where we would like to know the origin of a manuscript. For example, if the manuscript is written
in Andalusi font, it is most probably dates to the Andalusian Era. If it were written in Demasqi font,
it is most probably dates to the Abbasi Era.

## Getting started

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
# run cli program to identify fonts in image batch
python src/inference/predict.py --test_directory=<FULL PATH> --output_directory=<FULL PATH> --verbose=<OPTIONAL>
```

The results are two files:

- `output/results.txt`: contains the results of the evaluation.
- `output/time.txt`: contains the inference time for each result. The time is in seconds.

### Train new model

```bash
# run cli program to train and save model with name model.sav
python src/models/train_model.py --training_directory=<FULL PATH> --output_directory=<FULL PATH> --verbose=<OPTIONAL>
```

## Evaluation criteria

The system is evaluated on the results accuracy and the inference time. With an emphasis on the accuracy.

## System modules

- [x] Pre-processing Module.
- [x] Feature Extraction Module.
- [x] Model Selection and Training Module.
- [x] Performance Analysis Module.

**Note** The project is limited only to classical machine learning methods such as _Bayesian Classifiers_, _KNN_, _Linear/Logistic Regression_, Neural Networks (with two hidden layers as a maximum), _Support Vector Machines_, _Principal Component Analysis_, etc.

## Project structure

```bash
├── data
│   ├── processed      <- The processed data.
│   └── raw            <- The original dataset.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creators initials, and a short `-` delimited description, e.g.
│                         `01-data-exploration`.
│
├── src                <- Source code to build and evaluate models.
│   │
│   ├── data
│   │   ├── preprocess_data.py  <- Pre-processing module.
│   │   └── pipeline.py         <- Data preprocessing pipeline with cli interface.
│   │
│   ├── features
│   │   └──  build_features.py   <- Feature extraction module.
│   │
│   ├── models
│   │   └── train_model.py        <- Model training module.
│   │
│   ├── evaluation
│   │   ├── choose_model.py     <- Script to choose the best model.
│   │   └── evaluate_model.py   <- Script to evaluate the best model.
│   │
│   ├── inference
│   │   └── predict.py    <- Script to predict the results.
│   │
│   └── visualization
│       └── visualize.py        <- Script to visualize the results.
│
├── models             <- Trained and serialized models
│
├── cli                <- CLI code to interact with the models.
│
└── assets             <- Assets for the README file.

```

## Dataset

We used the [ACdb](https://drive.google.com/file/d/1dC7pwzT_RHL9B42H8-Nzf5Sant-86NV6/view) Arabic Calligraphy Database containing 9 categories of computer printed Arabic text snippets.

## Research Papers

**Note that** you don't need to open all research papers. From the following papers, they gave us a hint of which features to extract from an image to have the ability to identify fonts. Each research paper was implemented in separate python notebook in `notebooks/`.

1. [A Statistical Global Feature Extraction Method for Optical Font Recognition](https://link.springer.com/content/pdf/10.1007%2F978-3-642-20039-7_26.pdf)
2. [A New Computational Method for Arabic Calligraphy Style Representation and Classification](https://www.mdpi.com/2076-3417/11/11/4852/htm)
3. [An efficient multiple-classifier system for Arabic calligraphy style recognition](https://ieeexplore.ieee.org/document/8807829)
4. [Arabic Artistic Script Style Identification UsingTexture Descriptors](https://ieeexplore.ieee.org/abstract/document/9151569)
