# GPT from Scratch (Character-Level Transformer in PyTorch)

A decoder-only Transformer (GPT) implemented from scratch in PyTorch and trained on the Tiny Shakespeare dataset to generate Shakespeare-style text one character at a time.

Based on Andrej Karpathy's lecture ["Let's build GPT: from scratch, in code, spelled out"](https://youtu.be/kCc8FmEb1nY). I followed the lecture, rebuilt every component, and extended it with best-checkpoint saving, early stopping, and loss logging (see *My additions*).

## What's implemented
- Character-level tokenizer and train/validation split (90/10)
- Token + positional embeddings
- Masked (causal) single-head self-attention with query/key/value projections
- Scaled dot-product attention (divide by sqrt(head_size))
- Multi-head attention with output projection
- Position-wise feed-forward network
- Residual connections and pre-LayerNorm Transformer blocks
- Dropout, AdamW optimizer, cross-entropy loss
- Autoregressive text generation by sampling

## Results

| Run | Parameters | Hardware | Context | Best val loss |
|---|---|---|---|---|
| Small | 0.82M | HP laptop (CPU) | 64 | ~1.68 |
| Large | 10.79M | Google Colab T4 GPU | 256 | 1.48 (step 3000) |

Random guessing over the 65-character vocabulary gives a loss of about 4.17 (ln 65). Both runs started there (4.22) and learned real structure.

### Overfitting observation
In the large run, validation loss reached its minimum at step 3000 (1.4797) and then rose (1.50, 1.51, 1.54) while training loss kept falling (1.06 to 0.91). The 10.8M-parameter model was memorizing the ~1 MB training text. This is why the improved script saves only the best checkpoint and stops early.

### Sample output (small laptop model)
```
LEONTES:
Cold me you were war in his paked, I'll girl at come remas.

TRANIO:
The very boy: thou, Commend make this friendsmal:
Abuse, all have poles.
```
The model learned play formatting (speaker names, colons, line breaks) and real character names, but at this scale the sentences do not carry meaning.

## My additions (beyond the lecture)
- **Best-checkpoint saving:** weights are saved only when validation loss improves.
- **Early stopping:** training stops after `patience` evaluations without improvement.
- **Loss logging and plot:** `loss_log.csv` and `loss_curve.png` are produced automatically.
- **Separate `generate.py`:** loads saved weights and generates text without retraining.

## Setup and usage

```bash
pip install torch matplotlib
```

Download the data:
```bash
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Train (saves the best weights to `gpt_shakespeare.pt`):
```bash
python gpt_improved.py
```

Generate text from saved weights:
```bash
python generate.py
```

To get the larger model, edit the hyperparameters at the top of the training script:
```python
batch_size = 64; block_size = 256; max_iters = 5000
learning_rate = 3e-4; n_embd = 384; n_head = 6; n_layer = 6; dropout = 0.2
```
Use a GPU (for example Google Colab) for this configuration. The same values must be set in `generate.py` to load those weights.

## Files
| File | Purpose |
|---|---|
| `gpt_improved.py` | Model, training loop, checkpointing, logging |
| `generate.py` | Load saved weights and generate text |
| `gpt.py` | Original tutorial-style version (no early stopping) |

## Limitations and next steps
- Character-level tokens force the model to spell words, causing invented words. A subword tokenizer (BPE) would help.
- The dataset is tiny, so larger models overfit quickly. More data or stronger regularization would help.
- Possible extensions: learning-rate schedule, batched multi-head attention, training on a different corpus.

## Credits
Andrej Karpathy, "Let's build GPT" and [nanoGPT](https://github.com/karpathy/nanoGPT). Paper: Vaswani et al., "Attention Is All You Need" (2017).
