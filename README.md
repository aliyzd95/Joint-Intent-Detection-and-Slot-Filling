# Joint Intent Detection and Slot Filling in Persian

## 📝 Overview

This project presents a deep learning-based approach to **Intent Detection** and **Slot Filling** in Persian language utterances, specifically **banking-related** user queries and commands. We utilize a modified ParsBERT-based architecture, enhanced with CNN, Dense, and BLSTM layers, to simultaneously extract the user's main intent and label each token in the input sequence.

The project consists of a single Jupyter notebook (`.ipynb`), and all the necessary datasets and models are loaded or prepared within it.

---


## 📚 Dataset Description

The dataset used in this project consists of **3,307 Persian utterances**, most of which are **banking-related** user queries and commands. Each sample contains:

- A sequence of **BIO slot labels** (e.g., `B-Bankname`, `I-Amount`, `O`)
- An **intent label** (e.g., `balance_query`, `transaction_card`, `bill_payment`)
- The original **sentence in Persian**

### 📊 Example Rows

| Slot Tags                              | Intent           | Sentence                                        |
|----------------------------------------|------------------|-------------------------------------------------|
| O O O O O                              | balance_query    | .بگو چند ریال پول دارم                          |
| O O B-Bankname O O O                   | balance_query    | توی کارت ملتم چقدر پول دارم؟                   |
| O O B-Billname I-Billname O O ...      | bill_payment     | .قصد دارم قبض آب ماه جاری رو پرداخت کنم        |
| O O O O B-Amount I-Amount I-Amount ... | transaction_card | .من می‌خوام مبلغ 200 هزار تومان رو کارت به کارت کنم |



## 📚 Dataset Augmentation

To improve training performance, we augmented the dataset using the ChatGPT API. Approximately 3,000 new annotated sentences were generated. The process included:

- Designing an optimized prompt for GPT-3.5-Turbo.
- Post-processing generated samples.
- Creating slot annotations for each sentence using a custom NER model.
- Manual validation of generated labels.

---

## 🧠 Model Architecture

We use **ParsBERT** as the base model to extract contextual word embeddings. The classifier layer of ParsBERT is not used—instead, we extract the embedding layer’s output of shape `(batch_size, 50, 768)` and feed it into a hybrid architecture.

### 🔹 Key Components

- **TransformerEncoderLayer**  
  A custom transformer encoder layer for further feature enrichment.

- **Intent Detection Module**  
  Includes multiple parallel CNN layers with different filter sizes (1, 5, 7), followed by pooling and dense layers.

- **Slot Filling Module**  
  A `Bidirectional LSTM` layer processes the BERT output, followed by a dense classifier for per-token labeling.

- **Feature Fusion**  
  Outputs from both branches (CNN and BLSTM) are concatenated for joint optimization.

---

## ⚙️ Implementation Notes

- **Input Representation**  
  The input sequences are tokenized using the ParsBERT tokenizer and padded to a fixed length of 50 tokens.

- **Encoding Labels**  
  Token labels for slot filling are aligned with tokenized words and padded appropriately.

- **Regularization Techniques**
  - Dropout layers are applied to reduce overfitting.
  - Early stopping is used during training to ensure generalization.

---

## 📈 Results

Despite hardware limitations, the model achieved promising accuracy due to:

- Efficient use of feature-rich embeddings from ParsBERT.
- Customized CNN and BLSTM layers for intent and slot tasks.
- Smart dataset augmentation with GPT-generated data.

---
