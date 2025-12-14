## Sentiment Analysis on IMDB Movie Reviews

This project implements an end-to-end sentiment analysis pipeline to classify movie reviews as **positive** or **negative** using the IMDB dataset. The goal was not to overcomplicate the model, but to build a clean, explainable, and reproducible NLP workflow that reflects how real-world baseline models are built.

The project focuses on practical text preprocessing, feature extraction using TF-IDF, and training a Logistic Regression classifier, achieving strong and balanced performance.

---

## Dataset

* **Source**: IMDB Movie Reviews Dataset
* **Size**: 50,000 labeled reviews
* **Classes**: Positive (25,000) / Negative (25,000)

The dataset is intentionally not included in the repository due to size. You can download it separately and place it in the `Dataset/` folder.

---

## Approach

### 1. Text Preprocessing

* Lowercasing
* HTML tag removal
* URL removal
* Punctuation removal
* Stopword filtering using Scikit-learn’s built-in stopword list

This keeps the preprocessing fast, robust, and dependency-light.

### 2. Feature Extraction

* **TF-IDF Vectorization**
* Unigrams + Bigrams (`ngram_range=(1, 2)`)
* Limited to 5,000 features to balance performance and efficiency

### 3. Model

* **Logistic Regression**
* Optimized for convergence with increased max iterations
* Trained on an 80/20 train-test split

---

## Results

* **Accuracy**: ~88.5%
* Strong and balanced precision, recall, and F1-score across both classes

The model performs well as a solid baseline sentiment classifier.

---

## Project Structure

```
Sentiment-Analysis-IMDB/
├── Dataset/
│   └── IMDB Dataset.csv
├── Notebooks/
│   └── sentiment_analysis.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run

1. Download the IMDB Dataset (CSV format)
2. Place the file inside the `Dataset/` directory
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:

   ```bash
   python sentiment_analysis.py
   ```

---

## Notes

This project is intentionally kept simple and interpretable. The emphasis is on understanding the full NLP pipeline rather than chasing marginal accuracy gains with complex models.

---

## Future Improvements

* Model persistence (saving/loading trained models)
* Interactive prediction interface (CLI or API)
* Experimentation with other classifiers or embeddings

---

Built as part of a hands-on learning path focused on practical machine learning and NLP fundamentals.
