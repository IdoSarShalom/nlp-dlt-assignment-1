# 📚 Notebooks Summary & Accuracy Check

## ✅ All Notebooks Verified and Accurate

---

## 1️⃣ **preprocessing.ipynb**

### Purpose
Preprocesses raw CSV data (train/validation/test) through text cleaning steps.

### Configuration
```python
split = 'train'  # Options: 'train', 'validation', 'test'
```

### Input
- `./data/{split}.csv` - Raw CSV file with 'text' and 'label' columns

### Processing Steps
1. ✅ Load CSV data
2. ✅ Remove duplicates (train only)
3. ✅ Rename columns to 'Text' and 'Label'
4. ✅ Remove URLs
5. ✅ Remove special characters & punctuation
6. ✅ Remove extra whitespaces
7. ✅ Remove numeric values
8. ✅ Convert to lowercase
9. ✅ Remove stopwords (NLTK English)
10. ✅ Remove non-alphanumeric characters
11. ✅ Final cleanup

### Output
- `./data/{split}_preprocessed.pkl` - DataFrame with columns: ['Text', 'Label']
  - Text: cleaned text strings
  - Label: emotion labels as strings (not yet encoded)

### Key Features
- **Conditional duplicate removal**: Only removes duplicates from training data
- **Relative paths**: Uses `./data/` for portability
- **Split-aware**: Automatically sets input/output based on `split` variable

---

## 2️⃣ **train_gru.ipynb**

### Purpose
Trains a Bidirectional GRU model for emotion classification.

### Inputs
- `./data/train_preprocessed.pkl` - Training data
- `./data/validation_preprocessed.pkl` - Validation data

### Model Architecture
```
Input (Text)
    ↓
Embedding(input_dim=input_size, output_dim=100)
    ↓
Bidirectional GRU(128 units)
    ↓
BatchNormalization()
    ↓
Dropout(0.5)
    ↓
Dense(64, activation='relu')
    ↓
Dropout(0.5)
    ↓
Dense(6, activation='softmax')  # 6 emotions
```

### Training Process
1. ✅ Load preprocessed DataFrames
2. ✅ Map text labels to numbers (0-5)
3. ✅ Create and fit GRU tokenizer on training data
4. ✅ Tokenize text to sequences
5. ✅ Pad sequences to maxlen
6. ✅ Build BiGRU model
7. ✅ Train with EarlyStopping(patience=3)
8. ✅ Evaluate on validation set
9. ✅ Generate confusion matrix & classification report

### Outputs
- `./data/gru_model.keras` - Trained model
- `./data/gru_tokenizer.pkl` - GRU-specific tokenizer
- `./data/gru_metadata.pkl` - Contains:
  - `maxlen`: Maximum sequence length
  - `input_size`: Vocabulary size
  - `label_mapping`: Dict mapping emotion names to numbers
  - `val_accuracy`: Best validation accuracy
  - `val_loss`: Best validation loss
  - `best_epoch`: Epoch with best validation accuracy

### Hyperparameters
- Tokenizer: 60,000 max words
- Batch size: 32
- Epochs: 15 (with early stopping)
- Optimizer: Adam
- Loss: sparse_categorical_crossentropy

---

## 3️⃣ **train_lstm.ipynb**

### Purpose
Trains a Bidirectional LSTM model for emotion classification.

### Inputs
- `./data/train_preprocessed.pkl` - Training data
- `./data/validation_preprocessed.pkl` - Validation data

### Model Architecture
```
Input (Text)
    ↓
Embedding(input_dim=input_size, output_dim=100)
    ↓
Bidirectional LSTM(128 units)  ← KEY DIFFERENCE: LSTM instead of GRU
    ↓
BatchNormalization()
    ↓
Dropout(0.5)
    ↓
Dense(64, activation='relu')
    ↓
Dropout(0.5)
    ↓
Dense(6, activation='softmax')
```

### Training Process
1. ✅ Load preprocessed DataFrames
2. ✅ Map text labels to numbers (0-5)
3. ✅ Create and fit LSTM tokenizer on training data
4. ✅ Tokenize text to sequences
5. ✅ Pad sequences to maxlen
6. ✅ Build BiLSTM model
7. ✅ Train with EarlyStopping(patience=3)
8. ✅ Evaluate on validation set
9. ✅ Generate confusion matrix & classification report

### Outputs
- `./data/lstm_model.keras` - Trained model
- `./data/lstm_tokenizer.pkl` - LSTM-specific tokenizer
- `./data/lstm_metadata.pkl` - Contains:
  - `maxlen`: Maximum sequence length
  - `input_size`: Vocabulary size
  - `label_mapping`: Dict mapping emotion names to numbers
  - `val_accuracy`: Best validation accuracy
  - `val_loss`: Best validation loss
  - `best_epoch`: Epoch with best validation accuracy

### Hyperparameters
- **Identical to GRU** except for the recurrent layer type (LSTM vs GRU)

---

## 4️⃣ **model_comparison.ipynb**

### Purpose
Compares GRU and BiLSTM models on test data.

### Prerequisites
- ✅ Test data preprocessed: `./data/test_preprocessed.pkl`
- ✅ GRU model trained: `gru_model.keras`, `gru_tokenizer.pkl`, `gru_metadata.pkl`
- ✅ LSTM model trained: `lstm_model.keras`, `lstm_tokenizer.pkl`, `lstm_metadata.pkl`

### Process
1. ✅ Load test DataFrame
2. ✅ Map text labels to numbers
3. ✅ Load both models and their tokenizers
4. ✅ **Tokenize test data with GRU tokenizer** → `X_test_gru_padded`
5. ✅ **Tokenize test data with LSTM tokenizer** → `X_test_lstm_padded`
6. ✅ Evaluate GRU model on `X_test_gru_padded`
7. ✅ Evaluate LSTM model on `X_test_lstm_padded`
8. ✅ Compare performance metrics
9. ✅ Generate side-by-side confusion matrices
10. ✅ Generate classification reports
11. ✅ Per-class F1-score comparison
12. ✅ Declare winner

### Visualizations
- ✅ Test accuracy bar chart
- ✅ Test loss bar chart
- ✅ Side-by-side confusion matrices (GRU in blue, LSTM in green)
- ✅ Per-class F1-score comparison

### Key Comparisons
1. **Overall Metrics**
   - Test Accuracy
   - Test Loss

2. **Per-Class Performance**
   - Precision, Recall, F1-Score for each emotion
   - Identifies which model performs better on which emotions

3. **Final Verdict**
   - Declares winner based on test accuracy
   - Shows performance margin

### Important Note
**Each model is evaluated with its own tokenizer!**
- GRU uses `gru_tokenizer` → `X_test_gru_padded`
- LSTM uses `lstm_tokenizer` → `X_test_lstm_padded`

This ensures each model is tested in the same conditions it was trained in.

---

## 🎯 Emotion Label Mapping

All notebooks use consistent label encoding:

| Code | Emotion  |
|------|----------|
| 0    | sadness  |
| 1    | joy      |
| 2    | love     |
| 3    | anger    |
| 4    | fear     |
| 5    | surprise |

---

## 📊 Complete Workflow

```
1. preprocessing.ipynb (split='train')
   └─> train_preprocessed.pkl

2. preprocessing.ipynb (split='validation')
   └─> validation_preprocessed.pkl

3. preprocessing.ipynb (split='test')
   └─> test_preprocessed.pkl

4. train_gru.ipynb
   ├─ Reads: train_preprocessed.pkl, validation_preprocessed.pkl
   └─ Creates: gru_model.keras, gru_tokenizer.pkl, gru_metadata.pkl

5. train_lstm.ipynb
   ├─ Reads: train_preprocessed.pkl, validation_preprocessed.pkl
   └─ Creates: lstm_model.keras, lstm_tokenizer.pkl, lstm_metadata.pkl

6. model_comparison.ipynb
   ├─ Reads: test_preprocessed.pkl
   ├─ Reads: All GRU and LSTM assets
   └─ Generates: Comprehensive comparison report
```

---

## ✅ Verification Checklist

### preprocessing.ipynb
- [x] Uses `split` variable for configuration
- [x] Removes duplicates only for training data
- [x] Uses relative paths (`./data/`)
- [x] Saves DataFrame to pickle
- [x] Proper output messages

### train_gru.ipynb
- [x] Loads preprocessed pickle files
- [x] Maps labels to 0-5
- [x] Creates GRU-specific tokenizer
- [x] Tokenizes and pads sequences
- [x] Builds Bidirectional GRU model
- [x] Uses EarlyStopping
- [x] Saves model, tokenizer, and metadata
- [x] Generates visualizations

### train_lstm.ipynb
- [x] Loads preprocessed pickle files
- [x] Maps labels to 0-5
- [x] Creates LSTM-specific tokenizer
- [x] Tokenizes and pads sequences
- [x] Builds Bidirectional LSTM model
- [x] Uses EarlyStopping
- [x] Saves model, tokenizer, and metadata
- [x] Generates visualizations

### model_comparison.ipynb
- [x] Loads test preprocessed pickle
- [x] Maps labels to 0-5
- [x] Loads both models and tokenizers
- [x] Tokenizes test data with GRU tokenizer → `X_test_gru_padded`
- [x] Tokenizes test data with LSTM tokenizer → `X_test_lstm_padded`
- [x] Evaluates GRU on `X_test_gru_padded`
- [x] Evaluates LSTM on `X_test_lstm_padded`
- [x] Generates all comparisons and visualizations
- [x] Declares winner

---

## 🎓 Key Design Decisions

1. **Separate Tokenizers**: Each model has its own tokenizer to ensure independence and fair evaluation
2. **Pickle Format**: Used for DataFrames (efficient, preserves types)
3. **Conditional Preprocessing**: Duplicates removed only from training data
4. **Early Stopping**: Prevents overfitting, restores best weights
5. **Consistent Architecture**: Same structure for GRU and LSTM (except recurrent layer type)
6. **Comprehensive Comparison**: Multiple metrics and visualizations for thorough analysis

---

## 📝 Notes

- All notebooks use the same preprocessing steps
- Each model creates its own vocabulary during training
- Test data is tokenized separately for each model
- This approach ensures each model is evaluated in its optimal conditions
- The comparison is fair because both models use the same preprocessed text

---

**All notebooks are now accurate and ready to use!** ✅

