# 🔬 Hyperparameter Tuning Guide

## Overview

Both training notebooks (`02_train_gru.ipynb` and `03_train_lstm.ipynb`) now include comprehensive hyperparameter tuning experiments that systematically test different configurations using the **validation dataset**.

---

## 📊 Experiments Included

### For Both Models (GRU & LSTM)

Each notebook tests **6 key hyperparameters**:

| # | Hyperparameter | Values Tested | Purpose |
|---|----------------|---------------|---------|
| 1 | **Learning Rate** | 0.0001, 0.0005, 0.001, 0.005, 0.01 | Optimization speed and convergence |
| 2 | **Batch Size** | 16, 32, 64, 128 | Training stability and speed |
| 3 | **RNN Units** | 64, 96, 128, 192, 256 | Model capacity and complexity |
| 4 | **Dropout Rate** | 0.2, 0.3, 0.4, 0.5, 0.6 | Regularization strength |
| 5 | **Optimizer** | Adam, RMSprop, SGD | Optimization algorithm |
| 6 | **Embedding Mode** | Frozen, Fine-tuned | Transfer learning approach |

---

## 🎯 What Gets Evaluated

For each hyperparameter configuration:
- ✅ **Validation Accuracy** - Primary metric
- ✅ **Validation Loss** - Secondary metric  
- ✅ **Best Epoch** - Convergence speed (via early stopping)

All experiments use:
- 10 epochs maximum
- Early stopping (patience=3)
- Validation dataset for evaluation
- Same random seed for reproducibility

---

## 📈 Visualizations Provided

### Individual Experiment Plots
Each hyperparameter experiment shows:
- Line/bar plot of validation accuracy vs parameter value
- Highlighted best configuration (red star/marker)
- Grid lines for easy reading

### Comprehensive Summary
- **Horizontal bar chart**: Best accuracy for each hyperparameter
- **Multi-panel comparison**: All experiments side-by-side
- **Summary table**: Best values with accuracies and losses

---

## 🔍 How to Use

### 1. Run Initial Training
First, run the initial model training cells (before hyperparameter tuning section) to:
- Download pre-trained embeddings
- Create baseline model
- Understand baseline performance

### 2. Run Hyperparameter Experiments
Execute the hyperparameter tuning cells:
- **GRU**: Individual experiments (cells 25-39)
- **LSTM**: Combined experiment (cells 24-28)

**Note**: Experiments take time (~30-60 minutes total per model)

### 3. Analyze Results
Review the comprehensive summary:
- Compare accuracies across hyperparameters
- Identify best configuration for each parameter
- Note any surprising findings

### 4. Apply Recommendations
Use the recommended hyperparameters to:
- Retrain your final model
- Update the model architecture
- Improve test performance

---

## 💡 Key Insights to Look For

### Learning Rate
- **Too low**: Slow convergence, may not reach optimal
- **Too high**: Unstable training, divergence
- **Sweet spot**: Usually 0.0005 - 0.001

### Batch Size
- **Smaller** (16-32): More noise, better generalization
- **Larger** (64-128): Faster training, more stable
- **Trade-off**: Speed vs generalization

### RNN Units
- **Fewer** (64-96): Faster, less overfitting risk
- **More** (192-256): More capacity, better complex patterns
- **Watch for**: Diminishing returns, overfitting

### Dropout Rate
- **Lower** (0.2-0.3): Less regularization, higher training accuracy
- **Higher** (0.5-0.6): More regularization, better generalization
- **Balance**: Training vs validation performance

### Optimizer
- **Adam**: Usually best, adaptive learning rate
- **RMSprop**: Good for RNNs, stable
- **SGD**: Slower but can achieve better final performance

### Embedding Mode
- **Frozen**: Faster, prevents overfitting, leverages pre-training
- **Fine-tuned**: Adapts to your data, may overfit on small datasets
- **Recommendation**: Start frozen, try fine-tuning if accuracy plateaus

---

## 📋 Expected Output

### Summary Table Example
```
Hyperparameter      Best Value    Val Accuracy   Val Loss
Learning Rate       0.001         0.9125         0.3124
Batch Size          32            0.9087         0.3256
RNN Units           128           0.9156         0.3089
Dropout Rate        0.5           0.9098         0.3187
Optimizer           adam          0.9145         0.3102
Embedding Mode      Frozen        0.9087         0.3256
```

### Recommendations Example
```
💡 RECOMMENDATIONS FOR OPTIMAL MODEL
Learning Rate:       0.001
Batch Size:          32
RNN Units:           128
Dropout Rate:        0.5
Optimizer:           adam
Embedding Mode:      Frozen

🎯 Expected Validation Accuracy: 0.9156
📉 Expected Validation Loss:     0.3089
```

---

## 🚀 Best Practices

### 1. One Parameter at a Time
The experiments test each hyperparameter **independently**, holding others constant at defaults. This isolates the effect of each parameter.

### 2. Grid Search for Final Model
After finding best individual values, consider testing combinations:
```python
# Example: Test best 2-3 values for top parameters
best_lrs = [0.0005, 0.001]
best_units = [128, 192]
# Run grid search with combinations
```

### 3. Validation vs Test
- Use **validation set** for hyperparameter tuning (what we do)
- Save **test set** for final evaluation only
- This prevents overfitting to test data

### 4. Time Management
If experiments take too long:
- Reduce `epochs` from 10 to 5
- Test fewer values (e.g., 3 learning rates instead of 5)
- Run in batches (one experiment at a time)

### 5. Document Findings
Keep track of:
- Best hyperparameters found
- Any unusual patterns
- Differences between GRU and LSTM optimal settings

---

## 🔄 Iterative Refinement

After initial tuning:
1. **Retrain** with best hyperparameters
2. **Compare** to baseline model
3. **Fine-tune** further if needed:
   - Test narrower ranges around best values
   - Try combinations of best parameters
   - Test on test set for final validation

---

## 🆚 Comparing GRU vs LSTM

After running both notebooks, compare:

| Aspect | Compare |
|--------|---------|
| **Best Hyperparameters** | Are they similar or different? |
| **Optimal Learning Rate** | Does LSTM need different LR than GRU? |
| **Model Capacity** | Which needs more units? |
| **Regularization** | Which needs higher dropout? |
| **Training Speed** | Which converges faster? |
| **Final Accuracy** | Which achieves better performance? |

---

## 📚 Additional Resources

- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Grid Search**: Systematic testing of parameter combinations
- **Random Search**: More efficient for large parameter spaces
- **Bayesian Optimization**: Advanced method for hyperparameter tuning

---

## ⚠️ Important Notes

1. **Computational Cost**: Full hyperparameter tuning trains ~30-40 models
2. **Time Estimate**: Allow 30-60 minutes per model on CPU
3. **GPU Recommended**: Speeds up experiments significantly
4. **Reproducibility**: Results may vary slightly due to random initialization
5. **Validation Leakage**: Only use validation for tuning, not test set

---

## 🎓 Learning Outcomes

By completing the hyperparameter tuning experiments, you will:
- ✅ Understand impact of each hyperparameter
- ✅ Learn optimal configuration for emotion classification
- ✅ Gain experience with systematic experimentation
- ✅ Develop intuition for deep learning hyperparameters
- ✅ Compare GRU vs LSTM optimization requirements

---

## 📊 Results Interpretation

### High Variance in Results?
- Check if early stopping is triggering too early
- Consider increasing patience or epochs
- May indicate unstable training (reduce learning rate)

### Similar Accuracy Across Parameters?
- Model may be at capacity (data-limited)
- Pre-trained embeddings provide strong baseline
- Consider other improvements (data augmentation, ensemble)

### One Parameter Dominates?
- Focus on that parameter for further tuning
- May indicate bottleneck in current configuration
- Test combinations with that parameter varied

---

Good luck with your hyperparameter tuning experiments! 🚀

