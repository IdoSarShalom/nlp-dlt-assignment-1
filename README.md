# Emotion Analysis with Deep Learning

Ido

Final project as a part of NLP Deep Learning course 🌠.

Implemented in TensorFlow/Keras 🔥.

## Description 😊

In this project we implement deep learning models for classifying text into **6 emotion categories**: Sadness, Joy, Love, Anger, Fear, and Surprise. We compare two recurrent architectures - **Bidirectional GRU** with Word2Vec embeddings and **Bidirectional LSTM** with GloVe embeddings - achieving **~93-94% accuracy** on social media text.

## The Repository 🧭

We provide here a short explanation about the structure of this repository:

* `data/train.csv` and `data/validation.csv` contain the raw datasets from the Emotion dataset.
* `data/gru` and `data/lstm` contain the trained models and tokenizers after running the training notebooks.
* `00_eda.ipynb` contains Exploratory Data Analysis with comprehensive visualizations, class distribution analysis, and text statistics.
* `01_preprocessing.ipynb` contains the text preprocessing pipeline including tokenization, padding, and stopword removal. This notebook is run for both training and validation splits.
* `02_train_gru.ipynb` contains the **Bidirectional GRU** architecture training with Word2Vec embeddings.
* `03_train_lstm.ipynb` contains the **Bidirectional LSTM** architecture training with GloVe embeddings.
* `04_model_comparison.ipynb` contains the model evaluation, side-by-side performance comparison, and confusion matrix generation.
* `setup_environment.sh` is an automated script to set up the conda environment and install dependencies.
* `requirements.txt` contains the Python package dependencies.

## Running The Project 🏃

### Installation 📦

**Before trying to run anything, please make sure to install all the packages below.**

You can use the provided setup script:
```bash
bash setup_environment.sh
```

Or install manually:
```bash
conda create -n nlp-emotions python=3.10 -y
conda activate nlp-emotions
bash setup_environment.sh
```

### Training 🏋️

In order to train the models and reproduce the results:

1. **Exploratory Data Analysis**: Open and run `00_eda.ipynb` to analyze the dataset structure and statistics.
2. **Preprocessing**: Open `01_preprocessing.ipynb`. 
   * Run first with `split = 'train'` to generate `data/train_preprocessed.pkl`.
   * Run again with `split = 'validation'` to generate `data/validation_preprocessed.pkl`.
3. **Train GRU**: Run `02_train_gru.ipynb`. This will download Word2Vec embeddings and train the GRU model.
4. **Train LSTM**: Run `03_train_lstm.ipynb`. This will download GloVe embeddings and train the LSTM model.
5. **Evaluation**: Run `04_model_comparison.ipynb` to compare the models and view the results.

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
