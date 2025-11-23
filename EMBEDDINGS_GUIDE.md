# 🎯 Pre-trained Embeddings Strategy

## Overview

This project uses **different pre-trained word embeddings** for each model to maximize performance:

- **GRU Model** → **Word2Vec** embeddings
- **LSTM Model** → **GloVe** embeddings

## Why Different Embeddings?

### Word2Vec for GRU
- **Training Method**: Predictive (CBOW/Skip-gram)
- **Context Focus**: Local context patterns
- **Best For**: Short texts, local word associations
- **Why GRU?**: GRUs are faster and focus on recent context, making them a perfect match for Word2Vec's local patterns
- **Model Used**: `word2vec-google-news-300` (3 billion words, 300 dimensions)

### GloVe for LSTM
- **Training Method**: Global co-occurrence matrix factorization
- **Context Focus**: Global statistical relationships
- **Best For**: Capturing overall semantic structure
- **Why LSTM?**: LSTMs excel at long-term dependencies, complementing GloVe's global context understanding
- **Model Used**: `glove-twitter-200` (2 billion tweets, 200 dimensions)

## Dataset Alignment

Our **Emotions dataset** consists of short Twitter-like messages, making both embeddings highly relevant:
- **Word2Vec (Google News)**: Captures general English word relationships
- **GloVe (Twitter)**: Specifically trained on social media text (perfect match!)

## Implementation Details

### Embedding Layers
Both models use **frozen embeddings** (trainable=False):
- ✅ **Pros**: Faster training, prevents overfitting, leverages pre-trained knowledge
- 🔄 **Alternative**: Set `trainable=True` to fine-tune embeddings on your specific dataset

### Coverage Statistics
After running the notebooks, you'll see embedding coverage:
```
Words found in embeddings: ~85-90%
Words not found: ~10-15% (initialized as zeros)
```

Missing words are typically:
- Domain-specific terms
- Slang or informal language
- Misspellings
- Very rare words

## Installation Requirements

```bash
pip install gensim
```

The pre-trained embeddings are downloaded automatically on first run using `gensim.downloader`.

## Performance Expectations

Using pre-trained embeddings typically provides:
- 🚀 **Faster convergence** (fewer epochs needed)
- 📈 **Better generalization** (especially with small datasets)
- 🎯 **Improved accuracy** (leveraging billions of words of training data)

## Comparison: Trainable vs Pre-trained

| Aspect | Trainable Embeddings | Pre-trained Embeddings |
|--------|---------------------|----------------------|
| Training Time | Longer | Faster |
| Data Required | More (millions) | Less (thousands) |
| Generalization | Task-specific | Broad knowledge |
| Initial Performance | Lower | Higher |
| Final Performance | Good (with enough data) | Better (with limited data) |

## Experimentation

Want to try different configurations? Modify these settings:

### Option 1: Fine-tune embeddings
```python
model.add(Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    input_length=maxlen,
    trainable=True  # Allow fine-tuning
))
```

### Option 2: Try different embeddings
GloVe alternatives:
- `glove-wiki-gigaword-100` (Wikipedia + Gigaword)
- `glove-wiki-gigaword-300` (larger dimension)

Word2Vec alternatives:
- `word2vec-google-news-300` (current)
- Train your own Word2Vec on domain-specific data

### Option 3: Use same embeddings for both
Test if model architecture matters more than embedding choice:
- Both models with Word2Vec
- Both models with GloVe

## References

- **Word2Vec Paper**: Mikolov et al. (2013) - "Efficient Estimation of Word Representations in Vector Space"
- **GloVe Paper**: Pennington et al. (2014) - "GloVe: Global Vectors for Word Representation"
- **Gensim Documentation**: https://radimrehurek.com/gensim/

## Results Tracking

After training with pre-trained embeddings, compare with baseline:

| Model | Embedding | Val Accuracy | Test Accuracy | Notes |
|-------|-----------|--------------|---------------|-------|
| GRU (baseline) | Trainable 100d | 90.25% | TBD | Original |
| GRU | Word2Vec 300d | TBD | TBD | Pre-trained |
| LSTM (baseline) | Trainable 100d | TBD | TBD | Original |
| LSTM | GloVe 200d | TBD | TBD | Pre-trained |

Fill in the results after retraining with the new embeddings!

