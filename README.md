# Indian Tax Regime - Small Language Model (SLM)

A character-level GPT model trained from scratch to calculate Indian income tax under the New Tax Regime.

## 🎯 What This Does

This project trains a small language model on your Mac that can:
- Calculate income tax under India's New Tax Regime
- Handle salaried and non-salaried income
- Apply standard deductions and rebates
- Provide marginal relief calculations

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Mac M1/M2/M3 (uses MPS acceleration)

Install dependencies:
```bash
pip install torch numpy
```

## 🚀 Quick Start

### Step 1: Generate Training Data
```bash
python make_dataset.py
```
Creates 12,000 tax calculation examples with diverse income ranges.

### Step 2: Prepare Data for Training
```bash
python prepare.py
```
Converts text data to binary format for efficient training.

### Step 3: Train the Model
```bash
python train.py
```
Training takes ~20 hours on Mac M1/M2/M3.

**Expected training progress:**
- Initial loss: ~4.0
- Final loss: <0.5
- Total iterations: 20,000
- Checkpoints saved every 250 steps

### Step 4: Test the Model
```bash
python generate_taxregime.py
```
Runs 3 test cases to verify the model.

### Step 5: Interactive Mode
```bash
python sample.py
```
Enter income and salaried status interactively.

## 📁 Project Structure

```
TAX-SLM-FROM-SCRATCH/
├── make_dataset.py          # Generate training data
├── prepare.py               # Convert to training format
├── model.py                 # GPT architecture
├── train.py                 # Training script
├── generate_taxregime.py    # Test generation
├── sample.py                # Interactive mode
├── config/
│   └── train_taxregime.py   # Training parameters
├── data/
│   └── taxregime/
│       ├── input.txt        # Generated text data
│       ├── train.bin        # Training data
│       ├── val.bin          # Validation data
│       └── meta.pkl         # Vocabulary metadata
└── out-taxregime/
    └── ckpt.pt              # Trained model checkpoint
```

## 🔧 Configuration

Edit `config/train_taxregime.py` to adjust:
- Model size (n_layer, n_head, n_embd)
- Training duration (max_iters)
- Batch size and learning rate
- Device (mps/cpu/cuda)

**Current settings (optimized for Mac):**
- 6 layers, 6 heads, 384 embedding dimensions
- ~10.6M parameters
- Block size: 256 tokens
- Learning rate: 6e-4

## 💡 Usage Examples

### Example 1: Salaried Employee
```
Income: ₹12,10,000
Salaried: YES

Output:
Taxable income is Rs 11,35,000.
Slab tax is about Rs 97,500.
Marginal relief applies near Rs 12,00,000.
Estimated tax payable is Rs 62,500.
```

### Example 2: Non-Salaried
```
Income: ₹8,50,000
Salaried: NO

Output:
Taxable income is Rs 8,50,000.
Slab tax is about Rs 27,500.
Rebate does not apply at this income level.
Estimated tax payable is Rs 27,500.
```

### Example 3: High Income
```
Income: ₹25,00,000
Salaried: YES

Output:
Taxable income is Rs 24,25,000.
Slab tax is about Rs 5,27,500.
Rebate does not apply at this income level.
Estimated tax payable is Rs 5,27,500.
```

## 📊 Tax Regime Details

**Income Tax Slabs (New Regime):**
- ₹0 - ₹4,00,000: 0%
- ₹4,00,000 - ₹8,00,000: 5%
- ₹8,00,000 - ₹12,00,000: 10%
- ₹12,00,000 - ₹16,00,000: 15%
- ₹16,00,000 - ₹20,00,000: 20%
- ₹20,00,000 - ₹24,00,000: 25%
- Above ₹24,00,000: 30%

**Deductions:**
- Standard deduction for salaried: ₹75,000
- Rebate limit: ₹12,00,000
- Marginal relief: ₹12,00,000 - ₹12,75,000

## 🔍 Monitoring Training

Watch for these indicators:
1. **Loss decreasing steadily**: Train loss should drop from ~4.0 to <0.5
2. **Val loss tracking train loss**: No significant divergence (overfitting)
3. **Checkpoints saving**: Every 250 iterations
4. **Time per iteration**: ~100-200ms on M1/M2/M3

## ⚠️ Troubleshooting

### Garbled Output
- Delete `out-taxregime/` folder
- Re-run `python prepare.py`
- Re-run `python train.py`

### Out of Memory
- Reduce `batch_size` in `config/train_taxregime.py`
- Reduce `gradient_accumulation_steps`

### Slow Training
- Ensure MPS is enabled (Mac M1/M2/M3)
- Check that `device = 'mps'` in config
- Close other applications

### Poor Quality Results
- Train for more iterations (increase `max_iters`)
- Increase model size (n_layer, n_embd)
- Generate more training data (increase examples in make_dataset.py)

## 🎓 Model Architecture

- **Type**: Character-level GPT
- **Vocabulary**: ~50 unique characters
- **Context window**: 256 characters
- **Parameters**: ~10.6M
- **Optimizer**: AdamW with cosine learning rate decay
- **Training time**: 2-4 hours on Mac M1/M2/M3

## 📝 Notes

- This is a **simplified estimate** - actual tax may vary
- Does not include cess (4%) or surcharge
- Does not handle:
  - HRA exemptions
  - Section 80C deductions
  - Capital gains
  - Other complex scenarios
- Educational project for learning LLM training

## 🤝 Contributing

To extend this model:
1. Add more examples in `make_dataset.py`
2. Include additional tax scenarios
3. Adjust model architecture in `config/train_taxregime.py`
4. Retrain from scratch

## 📄 License

Educational project - use freely for learning purposes.

## 🙏 Credits

- Based on nanoGPT by Andrej Karpathy
- Tax calculations follow India's New Tax Regime (2024-25)

---

**Happy Training! 🚀**

For questions or issues, ensure you're running commands in sequence:
1. `make_dataset.py`
2. `prepare.py`
3. `train.py`
4. `generate_taxregime.py` or `sample.py`
