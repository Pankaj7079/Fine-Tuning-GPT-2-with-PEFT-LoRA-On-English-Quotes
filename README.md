# Fine-Tuning GPT-2 with PEFT LoRA on English Quotes

A concise, end-to-end project that fine-tunes **GPT-2** using **PEFT (LoRA)** on an English quotes dataset. This approach drastically reduces trainable parameters and GPU memory while retaining strong generation quality.

---

## 🚀 Highlights

* **Parameter-Efficient Fine-Tuning** with LoRA (train only low-rank adapters)
* **Fast & Memory-Efficient** training on consumer GPUs
* Clean **training → save → reload → inference** workflow
* Built with **Hugging Face Transformers + PEFT**

---

## 📁 Project Structure

```
.
├── lora-gpt2/                 # Saved LoRA adapter + tokenizer
├── training/                  # Training scripts / notebooks
├── inference/                 # Inference examples
├── README.md
└── requirements.txt
```

---

## 🧠 Model & Dataset

* **Base Model:** `gpt2`
* **Dataset:** `Abirate/english_quotes`
* **Split:** 90% train / 10% validation (no overlap)
* **Task:** Causal Language Modeling

---

## ⚙️ LoRA Configuration

```python
LoRA Rank (r):        8
LoRA Alpha:          16
Target Modules:      c_attn (GPT-2 attention)
Dropout:             0.05
Bias:                none
Task Type:           CAUSAL_LM
```

**LoRA update rule:**

```
W = W_pretrained + (alpha / r) * A * B
```

Only `A` and `B` are trained; base GPT-2 weights remain frozen.

---

## 🏋️ Training Setup

```python
Epochs:                  5
Batch Size (device):     4
Gradient Accumulation:   2  (effective batch size = 8)
Learning Rate:           2e-4
FP16:                    Enabled
Evaluation Strategy:     Steps
```

**Observed:** steady decrease in training & validation loss.

---

## 💾 Saving the Model

```python
model.save_pretrained("lora-gpt2")
tokenizer.save_pretrained("lora-gpt2")
```

This saves **only LoRA adapters**, not full GPT-2 weights.

---

## 🔁 Load Model for Inference

```python
base_model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "lora-gpt2")
```

---

## ✨ Text Generation Example

```python
prompt = "The secret to happiness is"
output = text_gen(
    prompt,
    max_new_tokens=70,
    do_sample=True,
    temperature=0.7
)
```

**Sample Output:**

> *"The secret to happiness is to be not ashamed of it..."*

---

## 📦 Installation

```bash
conda create -n atlas python=3.10
conda activate atlas
pip install -r requirements.txt
```

---

## 🛠 Tech Stack

* Python 3.10
* Hugging Face Transformers
* PEFT (LoRA)
* PyTorch
* Datasets

---

## 🎯 Key Takeaways

* LoRA enables efficient fine-tuning of large LLMs
* Ideal for domain adaptation with limited compute
* Easy to save, share, and deploy adapters

---

## 📌 Future Improvements

* Merge LoRA weights into base model
* Train on larger quote corpora
* Add Streamlit / FastAPI demo

---

⭐ If this repo helped you, consider giving it a star!