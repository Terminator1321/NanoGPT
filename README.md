# NanoGPT

A lightweight transformer-based language model with interactive 3D visualization of neural activations. Built with PyTorch and powered by a Flask web interface.

## Features

- **Transformer Architecture**: 6-layer transformer model with multi-head attention and feedforward networks
- **4 Model Versions**: Progressive training on different datasets (base, chat, short responses, feedback-tuned)
- **BPE Tokenization**: Byte-Pair Encoding with 4000 vocabulary size
- **Interactive Chat**: Real-time conversational interface via web browser
- **3D Visualization**: Live visualization of neuron activations across all transformer layers using Three.js
- **Feedback Training**: Fine-tune models on user corrections through the web interface
- **Token Inspector**: View token IDs and probabilities in real-time
- **Timeline Visualization**: Step-by-step processing pipeline visualization

## Project Structure

```
NanoGPT/
├── NanoGPT.py                    # Core transformer model
├── app.py                        # Flask web server
├── visualizer.py                 # Activation recording & timeline tracking
├── Test.py                       # CLI testing interface
├── Train.ipynb                   # Training notebook
├── bpe_tokenizer.json            # Pre-trained BPE tokenizer
├── feedback_data.jsonl           # User corrections log
│
├── DataHandler/
│   ├── DataExtractor.py          # PDF/text extraction utilities
│   ├── BPETokenizer.py           # Custom BPE tokenizer implementation
│   ├── Vocab.py                  # Vocabulary management
│   └── data.ipynb                # Data preprocessing notebook
│
├── finishedVersionModels/
│   ├── NanoGPT_v1.pt             # Base model
│   ├── NanoGPT_v2.pt             # Chat-tuned model
│   ├── NanoGPT_v3.pt             # Short response model (default)
│   └── NanoGPT_v4_feedback.pt    # Feedback-trained model
│
├── templates/
│   └── index.html                # Web UI template
│
└── static/
    ├── style.css                 # Web UI styles
    └── visualizer/
        ├── index.html            # 3D visualization page
        ├── main.js               # Three.js visualization logic
        └── style.css             # Visualization styles
```

## Model Architecture

The model follows a standard transformer pipeline:
- Input text → BPE Tokenization (vocab_size=4000)
- Token Embedding (384 dimensions)
- Positional Encoding (context window of 256 tokens)
- 6 Transformer Blocks with Self-Attention (6 heads) and Feed-Forward Networks
- Final Layer Normalization
- Output prediction over vocabulary

**Model Hyperparameters:**
- Embedding Dimension: 384
- Attention Heads: 6
- Transformer Layers: 6
- Block Size (context): 256 tokens
- Dropout: 0.2

## Installation

### Requirements
- Python 3.8+
- PyTorch (GPU recommended for faster inference)
- Flask & Flask-CORS
- Tokenizers (Hugging Face)
- Three.js (CDN, already included in HTML)

### Setup

1. **Clone and Navigate** to the NanoGPT directory

2. **Install Dependencies** using the requirement.txt file

3. **Verify Model Files** exist in `finishedVersionModels/`:
   - NanoGPT_v1.pt
   - NanoGPT_v2.pt
   - NanoGPT_v3.pt
   - NanoGPT_v4_feedback.pt

## Usage

### Web Interface

Start the Flask server to launch the web application. Once running, access:
- **Chat Interface** on the main page
- **3D Visualizer** for interactive neuron activation viewing

**Features:**
- Type a message and click "RUN" to generate responses
- View token breakdowns and probabilities
- Watch activations light up in real-time across layers
- Switch between model versions
- Correct responses and fine-tune the model

### CLI Testing

For command-line interaction, run the Test.py script.

**Available Commands:**
- `/1`, `/2`, `/3`, `/4` → Switch between model versions (base, chat, short response, feedback-trained)
- `/a` → Full response mode
- `/b` → Streaming mode
- `/fix <correction>` → Save manual correction for fine-tuning
- `/train_feedback` → Train on saved corrections
- `/save_feedback_model` → Save updated model
- `/bye` → Exit program

### Programmatic Usage

You can load and use the model directly in your Python code by importing the NanoGPT class, loading a pre-trained model, tokenizing input text, and generating predictions. The generate function supports configurable parameters like temperature and top-k sampling for different response styles.

## Model Versions 📚

| Version | Training Focus | Use Case |
|---------|---|---|
| **v1** | General text (PDFs, web data) | Knowledge base queries |
| **v2** | Dialogue/chat patterns | Conversation |
| **v3** | Short, concise responses | Quick answers (default) |
| **v4** | User feedback fine-tuned | Personalized responses |

## Web API

### Endpoints

**POST `/chat`**
- Send a message and get a response
- Body: `{"message": "your text"}`
- Returns: `{"response": "ai response"}`

**POST `/switch_model`**
- Switch active model version
- Body: `{"model": "v1"|"v2"|"v3"|"v4"}`

**POST `/tokenize`**
- Tokenize text with BPE
- Body: `{"text": "your text"}`
- Returns: `{"ids": [...], "tokens": [...], "count": n}`

**GET `/activations`**
- Get raw activation data for all layers

**GET `/activations_summary`**
- Get layer-wise activation statistics for visualization

**GET `/timeline`**
- Get processing pipeline timeline events

**GET `/chat_history`**
- Get conversation history

**POST `/fix`**
- Save user correction for fine-tuning
- Body: `{"correction": "corrected text"}`

**POST `/train`**
- Train on accumulated feedback corrections

**POST `/save`**
- Save fine-tuned model as v4

## Training

### Data Preparation

Use the data processing notebook to prepare training data.

This handles:
- PDF text extraction from documents
- Text cleaning and Unicode normalization
- Train/validation data splitting
- BPE tokenizer training and vocabulary building

### Training from Scratch

Use the provided training notebook to train models from scratch.

**Training stages:**
1. **Pre-training**: Large text corpus (500MB+) for general knowledge
2. **Chat tuning**: Dialogue datasets for conversation patterns
3. **Short response tuning**: Concise answer datasets for brief responses
4. **Feedback fine-tuning**: User corrections for personalization

### Fine-tuning on Feedback

#### Via Web UI:
1. Chat with the model
2. Click "Fix" to provide corrections
3. Model learns from corrections automatically

#### Programmatically:
Use the FeedBackTraning class to save feedback, train on accumulated corrections, and save the updated model. The system tracks all corrections and can fine-tune the model in multiple epochs.

## Visualization

### 3D Activation Viewer

The interactive 3D visualizer shows:
- **Pipeline stages**: Tokenizer → Embedding → Positional Encoding → Transformer Layers → Output
- **Neuron activations**: Color-coded by magnitude (brightness = stronger activation)
- **Transformer internals**: Click any transformer block to expand and see:
  - Multi-head attention layers
  - Feed-forward networks
  - Residual connections
  - Layer normalizations
- **Processing timeline**: Step-by-step breakdown of how tokens flow through the model
- **Token probabilities**: Top-5 candidate tokens for generation

### Real-time Features

- **Attention flow animation**: Watch attention patterns activate in real-time
- **Activation heatmaps**: Magnitude distribution across layers
- **Token-level tracking**: See which tokens were generated at each step
- **Interactive exploration**: Hover and click to inspect internals

## Performance Notes

- **Inference speed**: ~200-500ms per response (GPU)
- **Model size**: ~50MB per checkpoint
- **Memory**: ~500MB GPU VRAM required
- **Context window**: 256 tokens
- **Max generation length**: 120 tokens per request

## Parameters & Configuration

### Model Parameters (NanoGPT.py):
- **vocab_size**: 4000 (BPE vocabulary size)
- **block_size**: 256 (context window/max sequence length)
- **n_embd**: 384 (embedding dimension)
- **n_head**: 6 (number of attention heads)
- **n_layer**: 6 (number of transformer layers)
- **dropout**: 0.2 (dropout rate for regularization)

### Generation Parameters (app.py):
- **num_blocks**: 6 (must match n_layer)
- **temperature**: 0.7 (controls randomness; lower = more deterministic)
- **top_k**: 30 (top-K sampling parameter for diversity)
- **max_new_tokens**: 120 (maximum tokens to generate per request)

## Feedback System

The feedback system allows continuous improvement:

1. **User provides correction** → Saved to feedback_data.jsonl
2. **Train on feedback** → Fine-tune model on accumulated corrections
3. **Evaluate improvement** → Test updated model (v4)
4. **Iterate** → More feedback leads to better responses

Feedback data is stored in JSONL format with user prompts and corresponding correct AI responses for training.

## Troubleshooting

**Model not loading:**
- Ensure .pt files exist in `finishedVersionModels/`
- Check CUDA availability: `torch.cuda.is_available()`

**Tokenizer errors:**
- Verify `bpe_tokenizer.json` exists
- Reinstall tokenizers: `pip install --upgrade tokenizers`

**Visualization not rendering:**
- Check browser console for errors
- Ensure Three.js CDN is accessible
- Try a different browser

**Out of memory:**
- Reduce `max_new_tokens`
- Use CPU device instead of GPU
- Reduce batch size

## License

This project is open source. Feel free to use, modify, and distribute.

