# NLP Deep Learning Assignment 1
## Emotion Analysis with GRU and LSTM Models

This project implements emotion classification using GRU (Gated Recurrent Unit) and LSTM (Long Short-Term Memory) neural networks on text data.

---

## 📁 Project Structure

```
nlp-dlt-assignment-1/
├── 01_preprocessing.ipynb         # 1️⃣ Text preprocessing notebook
├── 02_train_gru.ipynb             # 2️⃣ GRU model training (Word2Vec embeddings)
├── 03_train_lstm.ipynb            # 3️⃣ BiLSTM model training (GloVe embeddings)
├── 04_model_comparison.ipynb      # 4️⃣ Model comparison notebook
│
├── data/
│   ├── gru/                       # GRU model artifacts (generated)
│   │   ├── gru_model.keras
│   │   ├── gru_tokenizer.pkl
│   │   └── gru_metadata.pkl
│   ├── lstm/                      # LSTM model artifacts (generated)
│   │   ├── lstm_model.keras
│   │   ├── lstm_tokenizer.pkl
│   │   └── lstm_metadata.pkl
│   ├── train.csv                  # Training dataset
│   ├── validation.csv             # Validation dataset
│   ├── test.csv                   # Test dataset (optional)
│   ├── train_preprocessed.pkl     # Preprocessed training data (generated)
│   ├── validation_preprocessed.pkl # Preprocessed validation data (generated)
│   └── test_preprocessed.pkl      # Preprocessed test data (generated)
│
├── references/                    # Reference notebooks
│   └── emotions-analysis-gru-94.ipynb  # Kaggle reference notebook
│
├── EMBEDDINGS_GUIDE.md            # 📖 Pre-trained embeddings guide
├── CHANGELOG.md                   # 📝 Project changes log
├── NOTEBOOKS_SUMMARY.md           # 📋 Notebooks summary
└── README.md                      # This file
```

---

## 🚀 Environment Setup

### Prerequisites
- Anaconda or Miniconda installed on your system
- Python 3.10+

### Step 1: Create Conda Environment

Create a new conda environment with all required dependencies:

```bash
conda create -n nlp-emotions python=3.10 -y
```

### Step 2: Activate the Environment

```bash
conda activate nlp-emotions
```

### Step 3: Install Required Packages

Install all necessary packages for data processing and modeling:

```bash
# Install basic packages via conda
conda install -y pandas numpy nltk jupyter ipykernel matplotlib seaborn

# Install TensorFlow, scikit-learn, and gensim via pip (better compatibility)
pip install tensorflow scikit-learn gensim
```

**Or use conda for everything (if preferred):**
```bash
conda install -y pandas numpy nltk jupyter ipykernel matplotlib seaborn tensorflow scikit-learn
pip install gensim  # gensim via pip is recommended
```

### Step 4: Register Jupyter Kernel

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name nlp-emotions --display-name "Python (nlp-emotions)"
```

---

## 📦 Installed Packages

The environment includes:

| Package | Version | Purpose |
|---------|---------|---------|
| **Data Processing** | | |
| pandas | 2.3.3 | Data manipulation and analysis |
| numpy | 2.2.5 | Numerical computing and arrays |
| nltk | 3.9.2 | Natural language processing and text preprocessing |
| **Deep Learning** | | |
| tensorflow | 2.20.0 | Deep learning framework for neural networks |
| keras | 3.12.0 | High-level neural network API (integrated with TensorFlow) |
| h5py | 3.15.1 | HDF5 file format support (for model saving) |
| **Machine Learning** | | |
| scikit-learn | 1.7.2 | Machine learning utilities and evaluation metrics |
| scipy | 1.15.3 | Scientific computing and optimization |
| **NLP & Embeddings** | | |
| gensim | 4.4.0 | Pre-trained word embeddings (Word2Vec, GloVe) |
| **Visualization** | | |
| matplotlib | 3.10.6 | Data visualization and plotting |
| seaborn | 0.13.2 | Statistical data visualization |
| **Development** | | |
| jupyter | 1.1.1 | Interactive notebook environment |
| ipykernel | Latest | Jupyter kernel for Python environments |
| tensorboard | 2.20.0 | TensorFlow visualization toolkit |

---

## 🎯 Running the Notebooks

### Option 1: Jupyter Notebook (Classic Interface)

```bash
# Navigate to project directory
cd /home/ido/GitRepos/nlp-dlt-assignment-1

# Activate environment
conda activate nlp-emotions

# Launch Jupyter Notebook
jupyter notebook
```

Then select the notebook you want to run from the browser interface.

### Option 2: Jupyter Lab (Modern Interface)

```bash
# Navigate to project directory
cd /home/ido/GitRepos/nlp-dlt-assignment-1

# Activate environment
conda activate nlp-emotions

# Launch Jupyter Lab
jupyter lab
```

### Option 3: VS Code / Cursor IDE

1. Open the notebook in your IDE
2. Click on the kernel selector (top right)
3. Select **"Python (nlp-emotions)"** from the list
4. Run the cells

---

## 📝 Detailed Notebook Guide

### 1️⃣ preprocessing.ipynb

The preprocessing notebook handles all text cleaning steps, **tokenization**, and sequence preparation.

**Configuration:**
```python
# Cell 4: Set the dataset split
split = 'train'  # or 'validation' or 'test'
```

This automatically sets:
- Input: `./data/{split}.csv`
- Output: `./data/{split}_X.npy` (tokenized sequences)
- Output: `./data/{split}_y.npy` (labels)
- Output: `./data/tokenizer.pkl` (training only - shared by both models)
- Output: `./data/tokenizer_metadata.pkl` (training only)

**Preprocessing Steps:**
1. ✅ Load CSV data
2. ✅ Check for null values
3. ✅ Remove duplicates (training data only)
4. ✅ Remove URLs
5. ✅ Remove special characters and punctuation
6. ✅ Remove extra whitespaces
7. ✅ Remove numeric values
8. ✅ Convert to lowercase
9. ✅ Remove stopwords (NLTK)
10. ✅ Remove non-alphanumeric characters
11. ✅ **Tokenize text to sequences** (using shared tokenizer)
12. ✅ **Pad sequences to uniform length**
13. ✅ Save sequences as NumPy arrays

**Why This Approach?**
- ⚡ **Faster training** - tokenization happens once, not twice
- 🎯 **Consistency** - both models use the same tokenizer and vocabulary
- 🔄 **Fair comparison** - identical input representation for both models
- 💾 **Efficient storage** - NumPy arrays are compact and fast to load

---

### 2️⃣ train_gru.ipynb

Trains a Bidirectional GRU model for emotion classification using **Word2Vec** pre-trained embeddings.

**Model Architecture:**
```
Embedding Layer (Word2Vec 300-dim, frozen)
    ↓
Bidirectional GRU (128 units)
    ↓
Batch Normalization
    ↓
Dropout (0.5)
    ↓
Dense (64, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (6, Softmax)
```

**Embeddings:**
- 🔵 **Word2Vec** (Google News, 300 dimensions)
- Captures local context patterns
- Complements GRU's fast sequential processing
- Auto-downloads on first run (~1.5 GB)

**Features:**
- Pre-trained word embeddings for better generalization
- Early stopping (patience=3)
- Automatic best weights restoration
- Training/validation curves visualization
- Confusion matrix
- Per-class classification report
- Saves model, tokenizer, and metadata

---

### 3️⃣ train_lstm.ipynb

Trains a Bidirectional LSTM model for emotion classification using **GloVe** pre-trained embeddings.

**Model Architecture:**
```
Embedding Layer (GloVe 200-dim, frozen)
    ↓
Bidirectional LSTM (128 units)
    ↓
Batch Normalization
    ↓
Dropout (0.5)
    ↓
Dense (64, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (6, Softmax)
```

**Embeddings:**
- 🟢 **GloVe** (Twitter, 200 dimensions)
- Captures global co-occurrence statistics
- Trained on social media text (perfect for emotions dataset!)
- Complements LSTM's long-term dependency learning
- Auto-downloads on first run (~1.4 GB)

**Features:**
- Pre-trained word embeddings optimized for Twitter-like text
- Early stopping (patience=3)
- Automatic best weights restoration
- Training/validation curves visualization
- Confusion matrix and classification report
- Saves model, tokenizer, and metadata

---

### 4️⃣ model_comparison.ipynb

Comprehensive comparison of GRU and BiLSTM models on test data.

**Key Feature:** Both models are evaluated on **identical test sequences**, ensuring a completely fair comparison!

**Comparisons Included:**
1. **Overall Metrics:**
   - Test accuracy
   - Test loss
   - Bar chart visualization

2. **Confusion Matrices:**
   - Side-by-side heatmaps
   - Per-emotion classification accuracy

3. **Classification Reports:**
   - Precision, Recall, F1-Score per class
   - Macro and weighted averages

4. **Per-Class Analysis:**
   - F1-score comparison bar chart
   - Best/worst performing emotions

5. **Final Verdict:**
   - Winner declaration
   - Performance margin calculation

**Why Fair Comparison?**
- ✅ Same tokenizer for both models
- ✅ Same vocabulary and word indices
- ✅ Same sequence lengths
- ✅ Identical test data representation

---

## 🎯 Pre-trained Embeddings Strategy

This project uses **different pre-trained word embeddings** for each model to maximize performance:

### Why Different Embeddings?

| Model | Embedding | Dimensions | Training Data | Rationale |
|-------|-----------|------------|---------------|-----------|
| **GRU** | Word2Vec | 300 | Google News (3B words) | Local context patterns match GRU's fast sequential processing |
| **LSTM** | GloVe | 200 | Twitter (2B tweets) | Global statistics complement LSTM's long-term dependencies |

### Benefits

✅ **Transfer Learning**: Leverage billions of words of pre-training  
✅ **Better Generalization**: Especially important for emotion classification  
✅ **Faster Convergence**: Models reach good performance in fewer epochs  
✅ **Domain Alignment**: GloVe trained on Twitter (perfect for social media emotions!)  

### First Run

On first execution, embeddings are automatically downloaded:
- **Word2Vec**: ~1.5 GB (cached for future use)
- **GloVe**: ~1.4 GB (cached for future use)

### Advanced Options

Want to experiment? See `EMBEDDINGS_GUIDE.md` for:
- Fine-tuning embeddings (trainable=True)
- Using different embedding models
- Training your own embeddings
- Performance comparison tips

---

## 🔧 Troubleshooting

### NLTK Data Download Issues

If you encounter NLTK data errors, download required data manually:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Kernel Not Found

If the kernel doesn't appear in Jupyter:

```bash
conda activate nlp-emotions
python -m ipykernel install --user --name nlp-emotions --display-name "Python (nlp-emotions)" --force
```

### Environment Activation Issues

If `conda activate` doesn't work, try:

```bash
source activate nlp-emotions
```

### Gensim/Embeddings Download Issues

If embeddings fail to download:

```bash
# Manually install gensim in the conda environment
conda activate nlp-emotions
pip install gensim

# Test the installation
python -c "import gensim.downloader as api; print('Gensim OK')"
```

If downloads are slow or fail, you can manually download embeddings:
- Word2Vec: https://github.com/RaRe-Technologies/gensim-data
- GloVe: https://nlp.stanford.edu/projects/glove/

### Memory Issues

If you run out of memory loading embeddings:
- Close other applications
- Use smaller embedding models (see `EMBEDDINGS_GUIDE.md`)
- Set `trainable=True` and use smaller custom embeddings

---

## 📊 Dataset Information

The emotions dataset contains English Twitter messages annotated with six emotions:

| Label | Emotion | Code |
|-------|---------|------|
| 0 | Sadness | 😢 |
| 1 | Joy | 😊 |
| 2 | Love | ❤️ |
| 3 | Anger | 😠 |
| 4 | Fear | 😨 |
| 5 | Surprise | 😲 |

### Dataset Files:

- `train.csv`: Training data (~16,000 samples)
- `validation.csv`: Validation data (~2,000 samples)

---

## 🔄 Complete Workflow

### Step 1: Preprocess Data 📝

Run `preprocessing.ipynb` for each dataset split (train/validation/test):

#### For Training Data:
1. Open `preprocessing.ipynb`
2. In Cell 4, set: `split = 'train'`
3. Run all cells
4. Output: 
   - `./data/train_X.npy` - tokenized sequences
   - `./data/train_y.npy` - labels
   - `./data/tokenizer.pkl` - **shared tokenizer**
   - `./data/tokenizer_metadata.pkl` - vocab_size, maxlen, etc.

#### For Validation Data:
1. In Cell 4, set: `split = 'validation'`
2. Run all cells
3. Output:
   - `./data/validation_X.npy`
   - `./data/validation_y.npy`
   - (Uses existing tokenizer from training)

#### For Test Data (when available):
1. In Cell 4, set: `split = 'test'`
2. Run all cells
3. Output:
   - `./data/test_X.npy`
   - `./data/test_y.npy`
   - (Uses existing tokenizer from training)

**Important:** The `split` variable automatically handles:
- Input path: `./data/{split}.csv`
- Output paths: `./data/{split}_X.npy` and `./data/{split}_y.npy`
- Duplicate removal (only for training data)
- **Tokenizer:** Created for training, loaded for validation/test

---

### Step 2: Train GRU Model 🔥

Run `train_gru.ipynb`:

**What it does:**
1. **Loads pre-tokenized sequences** from preprocessing
2. Loads tokenizer metadata (vocab_size, maxlen)
3. Builds and trains a Bidirectional GRU model
4. Visualizes training progress (accuracy/loss curves)
5. Evaluates on validation set
6. Displays confusion matrix and classification report
7. Saves model and metadata to `./data/`

**Output Files:**
- `gru_model.keras` - Trained model
- `gru_metadata.pkl` - Training metadata (accuracy, best_epoch, etc.)

**Note:** No separate tokenizer needed - uses shared tokenizer from preprocessing!

---

### Step 3: Train BiLSTM Model 🔥

Run `train_lstm.ipynb`:

**What it does:**
1. **Loads pre-tokenized sequences** from preprocessing
2. Loads **same tokenizer metadata** as GRU model
3. Builds and trains a Bidirectional LSTM model
4. Visualizes training progress (accuracy/loss curves)
5. Evaluates on validation set
6. Displays confusion matrix and classification report
7. Saves model and metadata to `./data/`

**Output Files:**
- `lstm_model.keras` - Trained model
- `lstm_metadata.pkl` - Training metadata (accuracy, best_epoch, etc.)

**Fair Comparison:** Uses identical tokenization as GRU model!

---

### Step 4: Compare Models 🏆

Run `model_comparison.ipynb`:

**Prerequisites:**
- Pre-tokenized test data (`test_X.npy`, `test_y.npy`)
- Trained GRU model (`gru_model.keras`)
- Trained BiLSTM model (`lstm_model.keras`)
- Shared tokenizer and metadata

**What it does:**
1. **Loads pre-tokenized test sequences** (same for both models)
2. Loads both trained models (GRU and BiLSTM)
3. Evaluates both models on **identical test sequences**
4. Compares performance:
   - Side-by-side accuracy and loss bar charts
   - Side-by-side confusion matrices
   - Detailed classification reports
   - Per-class F1-score comparison
5. Declares the winner! 🏆

**Example Output:**
```
============================================================
🏆 FINAL VERDICT
============================================================
Winner: BiLSTM Model
  Accuracy: 0.9402
  Margin: +0.0004 (0.04%)

📊 Summary:
  GRU Model:    0.9398 accuracy
  BiLSTM Model: 0.9402 accuracy
============================================================
```

**Fair Comparison Guaranteed:** Both models evaluated on identical tokenized sequences!

---

### Quick Start - Full Pipeline

```bash
# Activate environment
conda activate nlp-emotions

# Navigate to project
cd /home/ido/GitRepos/nlp-dlt-assignment-1

# 1. Preprocess training data (creates shared tokenizer)
#    - Open preprocessing.ipynb
#    - Set split='train' and run all cells
#    - Creates: train_X.npy, train_y.npy, tokenizer.pkl, metadata

# 2. Preprocess validation data (uses shared tokenizer)
#    - Set split='validation' and run all cells
#    - Creates: validation_X.npy, validation_y.npy

# 3. Preprocess test data (uses shared tokenizer)
#    - Set split='test' and run all cells
#    - Creates: test_X.npy, test_y.npy

# 4. Train GRU model
#    - Open and run train_gru.ipynb
#    - Loads pre-tokenized sequences

# 5. Train BiLSTM model  
#    - Open and run train_lstm.ipynb
#    - Loads same pre-tokenized sequences

# 6. Compare models on test set
#    - Open and run model_comparison.ipynb
#    - Fair comparison on identical sequences!
```

---

## 🤝 Additional Commands

### View Environment Packages

```bash
conda activate nlp-emotions
conda list
```

### Update Packages

```bash
conda activate nlp-emotions
conda update --all
```

### Deactivate Environment

```bash
conda deactivate
```

### Remove Environment (if needed)

```bash
conda env remove -n nlp-emotions
```

---

## 📝 Notes

- Always activate the `nlp-emotions` environment before running notebooks
- All notebooks use relative paths (`./data/`) for portability
- The `split` variable in preprocessing.ipynb controls which dataset to process
- Duplicate removal only happens for training data (preserves distribution in validation/test)
- Models are saved in Keras format (`.keras`) for TensorFlow/Keras compatibility
- Tokenizers and metadata are saved as pickle files for easy loading
- The environment is specifically configured for NLP text processing tasks

## 🎓 Model Training Tips

### Hyperparameters

Both training notebooks use these hyperparameters (can be modified):

```python
# Tokenization
num_words = 60000  # Vocabulary size

# Model architecture
embedding_dim = 100  # Embedding dimension
rnn_units = 128  # GRU/LSTM units
dense_units = 64  # Dense layer units
dropout_rate = 0.5  # Dropout rate

# Training
epochs = 15  # Maximum epochs
batch_size = 32  # Batch size
patience = 3  # Early stopping patience
```

### Improving Performance

To potentially improve model performance:

1. **Increase vocabulary size:** `num_words = 80000`
2. **Deeper embeddings:** `embedding_dim = 200`
3. **More RNN units:** `rnn_units = 256`
4. **Add more layers:** Stack multiple Bidirectional layers
5. **Adjust dropout:** Lower to 0.3 if underfitting
6. **More epochs:** Increase if early stopping triggers too soon
7. **Different optimizers:** Try `adamw` or `rmsprop`

### Expected Performance

Based on the reference notebooks:
- **Target Accuracy:** ~94% on test set
- **Training Time:** 10-15 minutes per model (CPU)
- **Training Time:** 2-3 minutes per model (GPU)

---

## 📚 References

- NLTK Documentation: https://www.nltk.org/
- Pandas Documentation: https://pandas.pydata.org/
- Jupyter Documentation: https://jupyter.org/

---

**Happy Modeling! 🎉**

