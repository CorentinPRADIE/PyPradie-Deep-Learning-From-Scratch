# Roadmap from Linear Layers to Transformers in PyPradie

## **Stage 1: Immediate Enhancements**
1. **Implement GPU Support**:
   - Modify the **Tensor class** to handle GPU/CPU switching (CuPy for GPU).
   - Add `to`, `cuda`, and `cpu` methods for tensor transfers.
   - Test GPU support with existing operations and layers.

2. **Implement Adam Optimizer**:
   - Add the **Adam optimizer** in the `optim` module.
   - Implement momentum and variance tracking (m_t, v_t), bias correction.
   - Test Adam with tasks like MNIST to validate functionality.

3. **Batch Gradient Descent**:
   - Extend the **DataLoader** class to support **mini-batches**.
   - Modify the training loop for batch updates.
   - Test on simple models to ensure correct gradient updates.

## **Stage 2: Expand Core Functionality**
4. **Batch Normalization**:
   - Implement **Batch Normalization** in the `nn` module.
   - Incorporate batch normalization in the forward pass, ensure gradients propagate.
   - Test on deeper models for improved convergence.

5. **Additional Optimizers** (Optional):
   - Implement **RMSProp** and other optimizers for flexibility.

## **Stage 3: Implement Recurrent Models**
6. **Recurrent Neural Networks (RNNs)**:
   - Build a basic **RNN layer** with hidden states and BPTT.
   - Test RNN on sequence tasks like **Shakespeare text generation** or **time series**.

7. **Long Short-Term Memory (LSTM)**:
   - Implement **LSTM** with input, forget, and output gates.
   - Handle long-range dependencies using the cell state.
   - Test on tasks like **IMDB sentiment analysis** or **time series**.

8. **Gated Recurrent Unit (GRU)** (Optional):
   - Implement **GRU** as a simpler alternative to LSTM.
   - Test on similar sequence tasks to compare performance.

## **Stage 4: Prepare for Transformers**
9. **Multi-Head Attention**:
   - Implement **Multi-Head Attention** with Query, Key, Value matrices.
   - Test attention on toy tasks like sequence alignment or the **bABI** dataset.

10. **Positional Encoding**:
   - Implement **Positional Encoding** (sinusoidal or learned).
   - Integrate into the transformer input embeddings.

11. **Transformer Encoder Block**:
   - Build the full **Transformer Encoder**: attention, feed-forward layers, residual connections.
   - Test on tasks like **language modeling** (e.g., **Penn Treebank** or **Wikitext-2**).

## **Stage 5: Build Transformers**
12. **Transformer Decoder Block**:
   - Build the **Transformer Decoder** with masked attention.
   - Stack encoder-decoder layers for sequence-to-sequence tasks (e.g., translation).
   - Test on **Multi30k** dataset for machine translation.

13. **Autoregressive Generation**:
   - Implement **autoregressive token generation**.
   - Add techniques like **beam search** and **greedy decoding**.

14. **Pretraining and Fine-Tuning**:
   - Implement **pretraining** (masked language modeling or causal modeling).
   - Add mechanisms for **fine-tuning** on downstream tasks (e.g., text classification, QA).

15. **Advanced Transformer Architectures** (Optional):
   - Implement models like **BERT**, **GPT**, or **T5**.
   - Test on benchmarks like **SQuAD** or **GLUE**.

## **Stage 6: Testing and Benchmarking**
16. **Comparison with PyTorch**:
   - Compare PyPradie with PyTorch using the **notebook setup**.
   - Benchmark transformer models on small datasets.

## **Final Milestone: Deploy a Small-Scale LLM Prototype**
- Train a small-scale transformer-based **language model** on a **tiny dataset**.
- Demonstrate **text generation**, **sequence-to-sequence translation**, or **question answering** using PyPradie.
