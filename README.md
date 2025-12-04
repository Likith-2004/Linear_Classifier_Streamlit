# 📊 Linear Classifier Sentiment Analysis

A robust and interactive web application designed to classify text sentiment in real-time. Built with Streamlit and powered by a Linear Machine Learning Classifier, this project demonstrates the end-to-end pipeline of Natural Language Processing (NLP)—from raw text to predictive insights.

## 🚀 Features

* **Real-time Analysis:** Instantly predict whether a given text is Positive or Negative.
* **Dual Interface:**
   * **Streamlit App:** A modern, reactive dashboard for easy interaction.
   * **Flask App:** A classic web interface using HTML templates.
* **Machine Learning Integration:** Utilizes a pre-trained linear model (Logistic Regression/SVM) for high-speed inference.
* **Custom Vectorization:** Implements `TF-IDF` or `CountVectorization` to process raw text inputs.

## 🛠️ Tech Stack

* **Language:** Python
* **Web Frameworks:** Streamlit, Flask
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Model Serialization:** Pickle

## 📂 Project Structure
```bash
Linear_Classifier_Streamlit/
├── 📄 Sentiment Analysis.ipynb  # Jupyter Notebook for data exploration & model training
├── 📄 streamlit_app.py          # Main Streamlit application file
├── 📄 app.py                    # Alternative Flask application file
├── 📄 sentiment_model.pkl       # Pre-trained linear classifier model
├── 📄 vectorizer.pkl            # Pre-trained text vectorizer
├── 📂 templates/                # HTML templates for the Flask app
└── 📄 README.md                 # Project documentation
```

## 💻 Installation

Follow these steps to set up the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/Likith-2004/Linear_Classifier_Streamlit.git
cd Linear_Classifier_Streamlit
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv linear

# Windows
linear\Scripts\activate

# macOS/Linux
source linear/bin/activate
```

### 3. Install Dependencies

Create a `requirements.txt` file or install the core libraries directly:
```bash
pip install streamlit flask scikit-learn pandas numpy
```

## 🎯 Usage

You can run the application using either Streamlit or Flask.

### Option A: Run with Streamlit (Recommended)

This launches the modern interactive dashboard.
```bash
streamlit run streamlit_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### Option B: Run with Flask

This launches the traditional web interface.
```bash
python app.py
```

The app will run at `http://127.0.0.1:5000`.

## 🧠 Model Workflow

The machine learning pipeline implemented in `Sentiment Analysis.ipynb` follows these steps:

1. **Data Preprocessing:** Cleaning and tokenizing text data.
2. **Vectorization:** Converting text into numerical vectors using the saved `vectorizer.pkl`.
3. **Training:** Fitting a Linear Classifier (e.g., Logistic Regression) on the processed data.
4. **Inference:** The `app.py` and `streamlit_app.py` scripts load the saved model (`sentiment_model.pkl`) to make predictions on new user input.

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the model accuracy or add new features:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/NewFeature`).
3. Commit your changes (`git commit -m 'Add some NewFeature'`).
4. Push to the branch (`git push origin feature/NewFeature`).
5. Open a Pull Request.

## 📬 Contact

[Likith GitHub Profile](https://github.com/Likith-2004)

If you find this project useful, please consider giving it a ⭐ star!
