# Emotion Analysis with Deep Learning 🎭

Ido Sar Shalom, Dana Gibor

Assignment 1 as part of Bar-Ilan's 83374 "NLP using DL techniques" 🌠.

Implemented in TensorFlow/Keras 🔥.

## Description 🤖

In this project we implement deep learning models for classifying text into **6 emotion categories**: Sadness, Joy, Love, Anger, Fear, and Surprise. We compare two recurrent architectures - **[Bidirectional GRU](https://arxiv.org/abs/1406.1078)** with Word2Vec embeddings and **[Bidirectional LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)** with GloVe embeddings - achieving **~93-94% accuracy** on social media text.

## The Repository 🧭

We provide here a short explanation about the structure of this repository:

* `data/train.csv` and `data/validation.csv` contain the raw datasets from the Emotion dataset.
* `data/gru` and `data/lstm` contain the trained models and tokenizers after running the training notebooks.
* `00_eda.ipynb` contains Exploratory Data Analysis with comprehensive visualizations, class distribution analysis, and text statistics.
* `01_preprocessing.ipynb` contains the text preprocessing pipeline including tokenization, padding, and stopword removal. This notebook runs on any data split.
* `02_train_gru.ipynb` contains the **Bidirectional GRU** architecture training with Word2Vec embeddings.
* `03_train_lstm.ipynb` contains the **Bidirectional LSTM** architecture training with GloVe embeddings.
* `04_inference.ipynb` contains the inference pipeline, model evaluation, and side-by-side performance comparison on test data.
* `requirements.txt` contains the Python package dependencies.

## Libraries to Install 📚

**Before trying to run anything please make sure to install all the packages below.**

| Library | Command to Run | Minimal Version |
| :--- | :--- | :--- |
| NumPy | `pip install numpy` | 2.2.5 |
| pandas | `pip install pandas` | 2.3.3 |
| matplotlib | `pip install matplotlib` | 3.10.6 |
| seaborn | `pip install seaborn` | 0.13.2 |
| NLTK | `pip install nltk` | 3.9.2 |
| scikit-learn | `pip install scikit-learn` | 1.7.2 |
| TensorFlow | `pip install tensorflow` | 2.20.0 |
| Keras | `pip install keras` | 3.12.0 |
| Gensim | `pip install gensim` | 4.4.0 |
| WordCloud | `pip install wordcloud` | 1.9.4 |

## Running The Project 🏃

### Inference 🔎

To predict emotions on a new test dataset (`test.csv`):

1. **Place Data**: Put the `test.csv` file in the `data/` directory.
2. **Preprocess**: Open `01_preprocessing.ipynb`.
   * Set `split = 'test'` in the configuration cell.
   * Run all cells. This will create `data/test_preprocessed.pkl`.
3. **Predict & Evaluate**: Run `04_inference.ipynb`. This notebook will:
   - Load the preprocessed test data.
   - Load the trained GRU and BiLSTM models.
   - Generate predictions.
   - Compare performance and display confusion matrices.

### Training 🏋️

In order to train the models and reproduce the results:

1. **Exploratory Data Analysis**: Open and run `00_eda.ipynb` to analyze the dataset structure and statistics.
2. **Preprocessing**: Open `01_preprocessing.ipynb`. 
   * Run first with `split = 'train'` to generate `data/train_preprocessed.pkl`.
   * Run again with `split = 'validation'` to generate `data/validation_preprocessed.pkl`.
3. **Train GRU**: Run `02_train_gru.ipynb`. This will download Word2Vec embeddings and train the GRU model.
4. **Train LSTM**: Run `03_train_lstm.ipynb`. This will download GloVe embeddings and train the LSTM model.
5. **Evaluation**: Run `04_inference.ipynb` to compare the models and view the results.
