import json

# Read the notebook
with open('./model_comparison.ipynb', 'r') as f:
    nb = json.load(f)

# Fix cell 16 (index 15) - Evaluate GRU Model
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        source = ''.join(cell['source'])
        
        # Fix GRU evaluation
        if 'gru_model.evaluate(X_test, y_test' in source:
            cell['source'] = [line.replace('gru_model.evaluate(X_test, y_test', 'gru_model.evaluate(X_test_gru_padded, y_test')
                             .replace('gru_model.predict(X_test, verbose', 'gru_model.predict(X_test_gru_padded, verbose')
                             for line in cell['source']]
            print(f"Fixed GRU evaluation in cell {i}")
        
        # Fix LSTM evaluation  
        if 'lstm_model.evaluate(X_test, y_test' in source:
            cell['source'] = [line.replace('lstm_model.evaluate(X_test, y_test', 'lstm_model.evaluate(X_test_lstm_padded, y_test')
                             .replace('lstm_model.predict(X_test, verbose', 'lstm_model.predict(X_test_lstm_padded, verbose')
                             for line in cell['source']]
            print(f"Fixed LSTM evaluation in cell {i}")

# Write back
with open('./model_comparison.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("✅ Fixed model_comparison.ipynb")

