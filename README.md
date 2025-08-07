# Code & Train GPT-5 - Step by Step

> **To train the model:**
> 
> ```bash
> python train_gpt5.py
> ```
> This will save the trained model.
>
> **To run inference:**
>
> ```bash
> python inference_gpt5.py
> ```
> This will load the saved model and run inference.
>
> Works on Google Colab.

---

GPT-5 is not open sourced, however, we can make an educated guess on how it's built:
- **GPT architecture + latest advancements in LLM pretraining**

If you need a reminder on GPT or base LLM architecture check:

---

🎓 **[🦙 LLaMA 4 From Scratch (first 2h 30min)](https://youtu.be/wcDV3l4CD14)**

> In the first **2h 30min**, I give a **clear and intuitive explanation** of both:

* 🧠 Attention Mechanism
* 🧩 Tokens & Tokenizers

Highly recommended if you're just starting out or want a solid refresher.

---

🎥 **[📘 GPT From Scratch by Andrej Karpathy](https://youtu.be/kCc8FmEb1nY)**

> A legendary tutorial from Karpathy that walks through building a GPT model from scratch. Great for understanding the fundamentals!
- If Andrej's course is too complex or difficult, then return to it after watching few more of the videos / courses below that you find more digestible.

---

## 🧠 Modern GPT architecture with latest advancements

⚙️ For attention mechanism there are 3 options:
1. Option 1: OpenAI invented its own attention mechanism, in this case it is almost certainly not too different from others, unless they compretely surpassed transformer architecture, which is unlikely to happen until 2026 or 2027. So their architecture is likely very similar to option 2 - GQA.

2. Option 2: Grouped-Query Attention (GQA) - memory and compute efficienct attention
- Explained in my course [Code & Train Qwen 3 From Scratch - Full Course](https://youtu.be/wM-KP_wNAeY) starting at 16:35

3. Option 3 (less likely): Multihead Latent Attention (MLA)
- Given that some of the latest LLMs are using Multihead Latent Attention (MLA) by DeepSeek, there is some chance GPT-5 also uses it instead of the GQA - you can learn how to code it in my [Code DeepSeek From Scratch Course](https://youtu.be/TfEG0TwueTs)
- Advantage of MLA over GQA is lower memory usage, however it's a bit more complex to build.
- I recommend learning all of these as it will help you understand neural networks, transformers and how to surpass transformers and invent the next architecture. 

💡 Rotary Positional Embeddings (RoPE) for better performance and context window extrapolation
- 📌 **[Rotary Positional Embeddings & Rotation Matrix + Python LLM Code](https://youtu.be/wiJ-OU-URYg)**
- 🧠 **[Get SMARTER Than 99% of AI Researchers](https://youtu.be/X0JryI85hL0)** - Beginning part
- 🛠️ **[RoPE In DeepSeek V3 – Code Step by Step](https://youtu.be/Rs9tLDSMUkM)**
- 🏋️ **[Excercises with ChatGPT Chat](https://chatgpt.com/share/68945a01-8d48-8002-8cf0-04b7f6db744b)**

🚀 Muon optimizer using Newton-Schulz orthogonalization for better weight updates, faster learning with less data
- This is the new best optimizer for 2D matrices, while AdamW is used for other parts of LLM. Highly likely Muon is used in GPT-5 as [OpenAI's researcher is tied to its invention](https://kellerjordan.github.io/posts/muon/).
- 🔁 [Backpropagation From Scratch](https://youtu.be/W8g1hvW4Wic) — Understand gradients deeply
- 🧠 [Orthonormal Matrix Intuition](https://youtu.be/FbYRZpBgFz4) — Key concept behind Muon’s update step
- Search for "Muon" on YouTube, you will find more tutorials.

> For all other things (below and above), watch my [Code & Train Qwen 3 From Scratch - Full Course](https://youtu.be/wM-KP_wNAeY) - I built and trained it on modern GPT architecture with latest advancements

📐 QK-Norm with RMSNorm for improved numerical / training stability

🔁 Hybrid optimization using Muon for matrices and AdamW for other parameters

🔄 SwiGLU activation and deep residual learning in the feedforward layers

🔢 Efficient dataset tokenization and caching with HuggingFace Datasets and Transformers

🧪 Validation metrics including loss, accuracy, and perplexity

🧵 Gradient accumulation + AMP (Automatic Mixed Precision) training for larger batch sizes

🎛️ Cosine learning rate scheduling with warmup

Find more tutorials / courses on AI research and engineering on [my YouTube](https://www.youtube.com/channel/UC7XJj9pv_11a11FUxCMz15g).

