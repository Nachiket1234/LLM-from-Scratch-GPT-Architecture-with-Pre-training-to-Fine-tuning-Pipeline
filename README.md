# LLM from Scratch: GPT Architecture with Pre-training to Fine-tuning Pipeline

A complete implementation of GPT-based language model from scratch using PyTorch, demonstrating the full pipeline from pre-training for text generation to fine-tuning for classification tasks.

## 🎯 Project Overview

This project implements a complete transformer-based language model (GPT architecture) from the ground up, showcasing:
- **Pre-training** on text data for coherent text generation
- **Fine-tuning** with transfer learning for spam classification
- **96%+ accuracy** on classification tasks

## ✨ Features

### 1. **Complete GPT Architecture**
- Multi-head self-attention mechanism
- Transformer blocks with residual connections
- Positional embeddings
- Layer normalization and GELU activation
- Feed-forward networks

### 2. **Pre-training Pipeline**
- BPE tokenization using GPT-2 tokenizer (tiktoken)
- Custom dataset with sliding window approach
- Training/validation split with loss tracking
- Text generation capabilities

### 3. **Fine-tuning for Classification**
- Transfer learning from pre-trained model
- Frozen base layers with unfrozen classification head
- Spam classification on custom dataset (200 samples)
- 96%+ test accuracy

### 4. **Advanced Text Generation**
- Temperature scaling for controlled randomness
- Top-k sampling
- Configurable generation parameters

## 🛠️ Technologies Used

- **Python** - Core programming language
- **PyTorch** - Deep learning framework
- **tiktoken** - GPT-2 BPE tokenization
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization
- **pandas** - Data manipulation

## 📊 Model Architecture

```
GPT Model Configuration:
- Vocabulary Size: 50,257 (GPT-2 tokenizer)
- Context Length: 128 tokens
- Embedding Dimension: 256
- Number of Heads: 4
- Number of Layers: 4
- Dropout Rate: 0.1
- Total Parameters: ~4.5M
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip install torch numpy matplotlib pandas tiktoken
```

### Installation

```bash
git clone https://github.com/Nachiket1234/LLM-from-Scratch-GPT-Architecture-with-Pre-training-to-Fine-tuning-Pipeline.git
cd LLM-from-Scratch-GPT-Architecture-with-Pre-training-to-Fine-tuning-Pipeline
```

### Usage

Open the Jupyter notebook:
```bash
jupyter notebook LLM_Implementation.ipynb
```

The notebook is organized into sections:
1. **Data Preparation & Tokenization** - Text preprocessing and BPE tokenization
2. **Model Architecture Components** - Building blocks (attention, transformers, etc.)
3. **Training & Text Generation** - Pre-training on text corpus
4. **Fine-tuning for Text Classification** - Transfer learning for spam detection
5. **Conclusion** - Results and future work

## 📈 Results

### Pre-training (Text Generation)
- Successfully generates coherent text after training on literary corpus
- Learns language patterns and structures

### Fine-tuning (Spam Classification)
- **Training Accuracy:** 96%+
- **Validation Accuracy:** 96%+
- **Test Accuracy:** 96%+
- Dataset: 200 samples (100 spam, 100 ham)

## 🔍 Key Components

### Multi-Head Self-Attention
Implements scaled dot-product attention with causal masking for autoregressive generation.

### Transformer Block
- Pre-layer normalization
- Residual connections
- Multi-head attention + Feed-forward network

### Training Pipeline
- Custom dataset with sliding window tokenization
- Batch processing with DataLoader
- Loss tracking and evaluation metrics
- Model checkpointing

### Transfer Learning
- Freeze pre-trained layers
- Fine-tune classification head and last transformer block
- Adapter-based training for downstream tasks

## 📚 Dataset

- **Pre-training:** `the-verdict.txt` - Literary text corpus
- **Fine-tuning:** Custom spam dataset with 200 labeled messages
  - 100 spam messages
  - 100 ham (non-spam) messages
  - Split: 70% train, 10% validation, 20% test

## 🎓 Key Insights

- Transformer architecture is highly modular and scalable
- Pre-trained language models can be effectively fine-tuned for downstream tasks
- Attention mechanisms enable the model to capture long-range dependencies
- Transfer learning significantly reduces training time and data requirements

## 🔮 Future Work

- Scale to larger models (GPT-2 medium/large configurations)
- Experiment with instruction fine-tuning
- Implement advanced decoding strategies (beam search, nucleus sampling)
- Add model quantization for deployment
- Support for multi-task learning

## 📝 License

This project is open-source and available for educational purposes.

## 👤 Author

**Nachiket N Doddamani**
- LinkedIn: [linkedin.com/in/nachiket-doddamani](https://www.linkedin.com/in/nachiket-doddamani)
- GitHub: [@Nachiket1234](https://github.com/Nachiket1234)
- Email: nachiketdoddamani@gmail.com

## 🙏 Acknowledgments

- Inspired by the GPT architecture and modern transformer-based language models
- Built using best practices from deep learning research
- Educational implementation for learning purposes

---

⭐ If you find this project helpful, please consider giving it a star!
