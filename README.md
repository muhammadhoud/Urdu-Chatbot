# Urdu Chatbot - Custom Transformer Implementation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_DEPLOYED_URL)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> *"A production-ready Urdu NLP system built from scratch, demonstrating deep learning fundamentals under resource constraints."*

---

## 📋 Project Overview

A fully functional Urdu language chatbot featuring a custom-built Transformer architecture. This project demonstrates end-to-end deep learning implementation—from architectural design to production deployment—while handling the unique challenges of low-resource language processing.

### The Challenge

This project tackled a deliberately constrained scenario:

- **20,000 samples** of isolated Urdu sentences
- **No conversational structure** (no Q&A pairs or dialogue flow)
- **Minimal contextual relationships** between samples
- **Limited data quality** requiring extensive preprocessing

**Objective:** Build a working system that demonstrates architectural understanding and adaptive problem-solving under real-world constraints.

---

## ✨ Key Features

### 1. Custom Transformer Architecture
Built entirely from scratch without pre-trained models or high-level frameworks. Every component—from positional encoding to multi-head attention—implemented manually to demonstrate deep architectural understanding.

### 2. Adaptive Solution Design
Developed an autoencoder approach to learn underlying Urdu text patterns from unstructured data, then leveraged these patterns for text generation—proving architectural flexibility when traditional methods aren't viable.

### 3. Comprehensive Urdu Language Support
- **Right-to-Left (RTL)** text rendering
- **Unicode normalization** for complex character variants
- **Nastaliq font** integration for authentic typography
- **Character-level tokenization** (125-token vocabulary)

### 4. Production-Ready Deployment
- Interactive Streamlit web interface
- Real-time text generation
- Robust error handling
- CPU-optimized inference
- Cloud deployment ready

---

## 🏗️ Technical Architecture

### Model Specifications
```
Encoder-Decoder Transformer
├── 2 Encoder Layers (512-dim feedforward)
├── 2 Decoder Layers (512-dim feedforward)
├── 2 Attention Heads per layer
├── Dynamic Positional Encoding (up to 5000 positions)
├── 256-dimensional embeddings
└── 125-character vocabulary
```

**Design Rationale:**
- **Compact architecture** - Optimized for limited training data
- **Character-level tokenization** - Handles Urdu's morphological richness
- **CPU-friendly** - Deployable without GPU infrastructure

### Core Technical Components

#### 1. Urdu Text Normalization Pipeline
```python
# Addresses Urdu's multiple character representations
- Unicode NFKC normalization
- Diacritic removal and standardization
- Alef variant normalization (آ أ إ → ا)
- Yeh variant standardization (ى → ی)
```

#### 2. Multi-Metric Evaluation Framework
- **BLEU Score** - Character-level n-gram precision
- **ROUGE-L** - Word-level longest common subsequence
- **chrF Score** - Character-based F-score
- **Perplexity** - Model confidence measurement
- **Human Evaluation** - Fluency, relevance, and adequacy assessment

#### 3. Training Strategy
```python
# Optimized for sparse, fragmented data
✓ BLEU-based model checkpointing
✓ 80/10/10 train/validation/test split
✓ Dropout (0.1) for regularization
✓ Early convergence detection
✓ Comprehensive metric tracking
✓ Aggressive data augmentation via normalization
```

#### 4. Deployment Optimization
- Model size reduction for CPU inference
- Streamlit caching (`@st.cache_resource`)
- Cloud-compatible architecture
- Efficient generation algorithms

---

## 🔧 Implementation Challenges & Solutions

### Challenge 1: Non-Conversational Dataset
**Problem:** Dataset contained isolated sentences without Q&A structure or dialogue flow.

**Solution:** Implemented an autoencoder architecture that learns to reconstruct sentences, then uses learned representations for generation—demonstrating architectural flexibility beyond traditional chatbot designs.

### Challenge 2: Limited Training Data
**Problem:** 20,000 sparse, contextually disconnected samples made pattern extraction challenging.

**Solution:** 
- Aggressive text normalization to maximize usable patterns
- Character-level tokenization for data efficiency
- Deliberately compact architecture (2 layers) to prevent overfitting
- Extensive regularization techniques

### Challenge 3: Urdu Linguistic Complexity
**Problem:** Multiple character variants, optional diacritics, RTL direction, complex Unicode composition.

**Solution:** Custom normalization pipeline to standardize text representation before tokenization.

### Challenge 4: Evaluation Methodology
**Problem:** Standard metrics assume quality conversational context.

**Solution:** 
- Multi-paradigm evaluation (automatic + human assessment)
- Comparative visualization analysis
- Transparent documentation of limitations

### Challenge 5: Resource-Constrained Deployment
**Problem:** Most Transformer deployments require GPU infrastructure.

**Solution:** CPU-optimized inference with small model size, efficient generation, and smart caching.

---

## 📊 Results & Performance

### Achievements
✅ Successfully learns Urdu character patterns from limited data  
✅ Generates grammatically plausible output within constraints  
✅ Correct RTL text handling and rendering  
✅ Production deployment and real-time inference  
✅ Comprehensive evaluation framework  
✅ Demonstrates deep understanding of Transformer architecture  

### Performance Metrics
| Metric	 | Score 			 | Description 			|
|----------------|-------------------------------|--------------------------------------|
| **BLEU** 	 | See `training_history.json` | Character-level accuracy 		|
| **ROUGE-L** 	 | See `training_history.json` | Word-level overlap (Urdu-specific) 	|
| **chrF** 	 | See `training_history.json` | Character F-score 			|
| **Perplexity** | See `training_history.json` | Model confidence 			|

---

## 🚀 Live Demo

**Try the deployed application:** [https://urdu-chatbot-nrzkicdtrgvlvhvbd9ixbm.streamlit.app/]

Experience real-time RTL Urdu text generation powered by a custom Transformer architecture.

---

## 💻 Installation & Usage

### Quick Start
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/urdu-chatbot.git
cd urdu-chatbot

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

### Docker Deployment
```bash
docker build -t urdu-chatbot .
docker run -p 8501:8501 urdu-chatbot
```

---

## 📁 Project Structure

```
urdu-chatbot/
├── app.py                          # Streamlit interface
├── model_architecture.py           # Transformer implementation
├── requirements.txt                # Dependencies
├── README.md                       # Documentation
├── model/
│   ├── final_transformer_model.pth # Trained weights
│   └── urdu_vocabulary.json        # Character vocabulary
└── .streamlit/
    └── config.toml                 # UI configuration

notebooks/ (not included in deployment)
├── preprocessing.ipynb             # Data pipeline
├── training.ipynb                  # Training loop
└── evaluation.ipynb                # Metrics & analysis
```

---

## 🎓 Skills Demonstrated

### Technical Competencies
1. **Deep Learning Architecture** - Custom Transformer implementation from scratch
2. **Low-Resource NLP** - Urdu language processing with limited data
3. **Evaluation Methodology** - Multi-metric assessment framework
4. **Production Engineering** - Deployment-ready implementation
5. **Adaptive Problem Solving** - Effective solutions under constraints

### Key Learnings
This project demonstrates that meaningful results are achievable even with:
- Limited data quantity
- Constrained data quality
- Non-standard problem structure

Through:
- Strong architectural fundamentals
- Creative problem-solving
- Rigorous evaluation methodology
- Production-ready engineering practices

---

## 🔮 Future Enhancements

Given access to quality conversational datasets:
- [ ] Fine-tuning on dialogue corpora
- [ ] Beam search decoding implementation
- [ ] Multi-turn context awareness
- [ ] Scaled architecture (6-12 layers)
- [ ] Attention mechanism visualization
- [ ] Retrieval-augmented generation (RAG)
- [ ] Subword tokenization (BPE/WordPiece)

---

## 🙏 Acknowledgments

- **Sir Usama & Sir Ali Raza** - Project design and mentorship
- **Vaswani et al. (2017)** - "Attention is All You Need" paper
- **Urdu NLP Community** - Linguistic resources and documentation

---

## 📄 License

MIT License - Open for learning, modification, and extension.

---

## 👤 Author

**Muhammad Houd**
- GitHub: [@muhammadhoud](https://github.com/muhammadhoud)
- LinkedIn: [Muhammad Houd](https://www.linkedin.com/in/muhammadhoud/)
- Email: 6240houd@gmail.com

---

## 📬 Contact

Questions about implementation? Found an issue? Interested in discussing low-resource NLP strategies?

**Open an issue** or **reach out directly** - I'm always happy to discuss Transformer architectures, Urdu NLP, or problem-solving under constraints.

---

<div align="center">

**⭐ Star this repository if you found it valuable**

*"Technical mastery is proven not by perfect conditions, but by delivering working solutions when constraints are real."*

</div>
