
# 🎬 Sentiment Analysis with RNN (LSTM) on IMDb Dataset

This project focuses on performing **sentiment analysis** on the **IMDb Movie Reviews dataset** using a **Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM)**.  
The notebook includes **data preprocessing**, **model building**, **training**, and **performance evaluation** for classifying reviews as **positive** or **negative**.

---

## 📂 Project Structure
```

Sentiment\_Analysis\_with\_RNN\_(LSTM)\_on\_IMDb\_Dataset.ipynb   # Main Jupyter Notebook with code and analysis
requirements.txt                                            # Project dependencies
README.md                                                  # Project documentation

````

---

## 🚀 Features
- Preprocessing IMDb dataset (tokenization, padding, embeddings).
- Building an LSTM-based RNN model for text classification.
- Training and evaluating the model on large-scale text data.
- Visualizing accuracy and loss over training epochs.
- Predicting sentiment for custom movie reviews.

---

## 🛠 Tech Stack
- **Programming Language:** Python 🐍
- **Libraries Used:**
  - `tensorflow` / `keras` - Deep Learning framework
  - `numpy` - Numerical computations
  - `pandas` - Data handling
  - `matplotlib` & `seaborn` - Data visualization
  - `jupyter` - Notebook environment

---

## 📊 Workflow
1. **Import Libraries** – Load required Python libraries.  
2. **Load Dataset** – Use IMDb reviews dataset from `tensorflow.keras.datasets`.  
3. **Data Preprocessing** – Tokenization, padding, and sequence preparation.  
4. **Model Building** – Create an RNN with LSTM layers.  
5. **Model Training** – Train the network on training data.  
6. **Model Evaluation** – Evaluate on test dataset.  
7. **Prediction** – Test sentiment analysis on custom text input.  

---

## 📥 Installation
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/sentiment-analysis-lstm-imdb.git
cd sentiment-analysis-lstm-imdb
````

### **2. Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## 🧪 Usage

Open the Jupyter Notebook and run step-by-step:

```bash
jupyter notebook
```

Then open `Sentiment_Analysis_with_RNN_(LSTM)_on_IMDb_Dataset.ipynb` and execute the cells.

Example prediction:

```python
review = "The movie was fantastic! I really loved the storyline."
predict_sentiment(review)
# Output: Positive
```

```python
review = "It was the worst movie I have ever seen."
predict_sentiment(review)
# Output: Negative
```

---

## 📈 Results

* Achieved **\~85–90% accuracy** on IMDb sentiment classification.
* Clear training vs validation accuracy and loss plots.
* Model successfully predicts sentiment for new text inputs.

---

## 🤝 Contributing

Contributions are always welcome!
Steps to contribute:

1. Fork this repository
2. Create a new branch
3. Commit your changes
4. Create a pull request

---

## 📜 License

This project is licensed under the MIT License.
Feel free to use and modify as per your needs.

---

## 👤 Author

* **Krishna Karbhari**
* GitHub: [kishu01karb](https://github.com/kishu01karb)

---

## 📦 requirements.txt

```
tensorflow>=2.0
numpy
pandas
matplotlib
seaborn
jupyter
```

```

✅ Now you can copy this **entire block at once** and paste it into your README.md file.  

Do you also want me to include **badges (like TensorFlow, Python, Jupyter)** at the top for a more professional GitHub look?
```
