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

```
python main.py --data_dir=data/ --model_dir=models/ --output_dir=output/
```

The results are two files:

- `output/results.txt`: contains the results of the evaluation.
- `output/time.txt`: contains the inference time for each result. The time is in seconds and rounded to two decimal places.

## Evaluation criteria

The system is evaluated on the results accuracy and the inference time. With an emphasis on the accuracy.

## System modules

- [ ] Pre-processing Module.
- [ ] Feature Extraction Module.
- [ ] Model Selection and Training Module.
- [ ] Performance Analysis Module.

**Note** The project is limited only to classical machine learning methods such as _Bayesian Classifiers_, _KNN_, _Linear/Logistic Regression_, Neural Networks (with two hidden layers as a maximum), _Support Vector Machines_, _Principal Component Analysis_, etc.

## System pipeline

**TODO**

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
│   │   └── split_data.py       <- Splitting module.
│   │
│   ├── features
│   │   └──  build_features.py   <- Feature extraction module.
│   │
│   ├── models
│   │   └── train_svm.py        <- Model training module.
│   │
│   ├── evaluation
│   │   ├── choose_model.py     <- Script to choose the best model.
│   │   └── evaluate_model.py   <- Script to evaluate the best model.
│   │
│   ├── inference
│   │   └── predict_model.py    <- Script to predict the results.
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

**Note that** you don't need to open all research papers. From the following papers, they gave us a hint of which features to extract from an image to have the ability to identify fonts.

1. [Local Binary Patterns for Arabic Optical Font Recognition](https://www.researchgate.net/publication/286595227_Local_Binary_Patterns_for_Arabic_Optical_Font_Recognition)
2. [Font Recognition Based on Global Texture Analysis](http://www.cbsr.ia.ac.cn/publications/yzhu/Font%20Recognition%20Based%20on%20Global%20Texture%20Analysis.pdf)
3. [Large-Scale Visual Font Recognition](https://openaccess.thecvf.com/content_cvpr_2014/papers/Chen_Large-Scale_Visual_Font_2014_CVPR_paper.pdf)
4. [A Statistical Global Feature Extraction Method for Optical Font Recognition](https://link.springer.com/content/pdf/10.1007%2F978-3-642-20039-7_26.pdf)
