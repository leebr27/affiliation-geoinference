# Affiliation Geoinference

Geoinference of author affiliations using NLP-based text classification.

## Overview

This project implements a machine learning pipeline for inferring geographic locations (cities) from author affiliation strings commonly found in academic publications. The system uses Natural Language Processing (NLP) techniques to classify affiliation text into geographic locations, trained on the MapAffil 2018 dataset.

## Publication

This work was published in **Nature Scientific Reports**:

[https://doi.org/10.1038/s41598-024-73318-7](https://doi.org/10.1038/s41598-024-73318-7)

## Repository Structure

### Numbered Jupyter Notebooks (Core Pipeline)

| Notebook | Description |
|----------|-------------|
| `1.create_mapaffil.ipynb` | Converts the original MapAffil 2018 TSV file to CSV format for easier processing |
| `2.create_unique_sorted_mapaffil.ipynb` | Reduces the MapAffil dataset to unique affiliation strings sorted by frequency of occurrence |
| `3.data_preparation.ipynb` | Performs data cleaning and preparation, including spaCy NER processing, filtering, and creation of training/validation sets |
| `4.save_model.ipynb` | Creates and saves the TF-IDF vectorizer and LinearSVC classifier trained on 1 million affiliations |
| `5.model_evaluation.ipynb` | Evaluates the trained model against multiple test and validation sets with accuracy, F1, recall, and precision metrics |

### Unnumbered Jupyter Notebooks (Experimentation)

| Notebook | Description |
|----------|-------------|
| `vectorizer_selection.ipynb` | Compares different text vectorization approaches: TF-IDF, Word2Vec, and BERT embeddings |
| `classifier_selection.ipynb` | Compares different classifiers: LinearSVC, Random Forest, Logistic Regression, and Multinomial Naive Bayes |
| `LinearSVC.ipynb` | Standalone LinearSVC implementation with TF-IDF vectorization |
| `LSTM+W2V.ipynb` | LSTM neural network with Word2Vec embeddings for text classification |
| `Bi-LSTM+W2V.ipynb` | Bidirectional LSTM neural network with Word2Vec embeddings |
| `GRU+W2V.ipynb` | GRU neural network with Word2Vec embeddings |

### Saved Model Files

| File | Description |
|------|-------------|
| `geoinference_vectorizer_1mil.joblib.lzma` | Trained TF-IDF vectorizer (fitted on 1 million affiliations) |
| `geoinference_linearsvc_1mil.joblib.lzma` | Trained LinearSVC classifier (fitted on 1 million affiliations) |

### Data Files

The `data/` folder contains the following files:

| File | Description |
|------|-------------|
| `clean_spacy_mapaffil.parquet` | Cleaned training dataset with spaCy-extracted entities |
| `ambiguous_mapaffil_validation.parquet` | Validation set of affiliations where MapAffil could not assign a specific city |
| `post_2018_validation.parquet` | Validation set of PubMed affiliations from papers published after December 2018 |
| `mapaffil2018-unique_sorted-affiliation-orgs.txt.gz` | spaCy-extracted organization (ORG) entities |
| `mapaffil2018-unique_sorted-affiliation-gpes.txt.gz` | spaCy-extracted geopolitical entity (GPE) entities |
| `authorships.csv.gz` | Author affiliations with publication year information |

## Data Processing Pipeline

1. **Data Import**: Load the MapAffil 2018 dataset containing over 52 million authorships
2. **Deduplication**: Reduce to approximately 20 million unique affiliation strings
3. **NER Processing**: Extract organization (ORG) and geopolitical entity (GPE) named entities using spaCy
4. **Data Cleaning**: Remove affiliations that:
   - Have no detected ORGs or GPEs
   - Have no assigned country
   - Contain special prefixes (FROMPMC, FROMNIH, FROMPAT)
   - Contain newlines or semicolons
   - Exceed 200 characters
5. **Validation Set Creation**: Separate ambiguous affiliations and post-2018 publications for validation

## Model Architecture

The final model uses:
- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency) with English stop words removed
- **Classifier**: LinearSVC (Linear Support Vector Classification)
- **Training Data**: 1 million affiliation strings from the cleaned MapAffil dataset

## Requirements

### Python Dependencies

- pandas
- numpy
- scikit-learn
- joblib
- spacy
- gensim
- tensorflow/keras (for deep learning experiments)
- transformers (for BERT experiments)
- fastparquet
- torch

## Usage

### Loading the Trained Model

```python
import joblib

# Load the vectorizer and classifier
vectorizer = joblib.load('geoinference_vectorizer_1mil.joblib.lzma')
classifier = joblib.load('geoinference_linearsvc_1mil.joblib.lzma')

# Make predictions
affiliation_text = "Department of Computer Science, Stanford University, Stanford, CA"
features = vectorizer.transform([affiliation_text])
predicted_city = classifier.predict(features)
```

### Reproducing Results

To reproduce the findings reported in the manuscript:

1. **Notebooks 4 and 5** can be run directly using the pre-processed data files in the `data/` folder
2. The unnumbered notebooks can also be run without additional data downloads

### Running the Full Pipeline (Optional)

If you wish to run the complete data preparation pipeline (Notebooks 1, 2, and 3):

1. Download `mapaffil2018.tsv.gz` from [Illinois Data Bank](https://databank.illinois.edu/datasets/IDB-2556310)
2. Save the file to the `data/` folder
3. Execute notebooks 1, 2, and 3 in order

**Note**: The MapAffil dataset is NOT included in this repository due to its size. The final training and validation sets are already provided in the `data/` folder.

## Evaluation

The model is evaluated using:
- **Accuracy**: Overall prediction accuracy
- **F1 Score**: Weighted F1 score
- **Recall**: Weighted recall
- **Precision**: Weighted precision

Evaluation is performed on:
1. Random samples from the cleaned MapAffil dataset
2. Ambiguous affiliations that MapAffil could not resolve to a specific city
3. Post-2018 PubMed affiliations (temporal validation)

## Acknowledgments

- MapAffil 2018 dataset from the Illinois Data Bank
- spaCy for named entity recognition
- scikit-learn for machine learning implementations
