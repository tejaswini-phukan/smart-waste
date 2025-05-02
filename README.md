# 🗑️ Smart Waste Segregation Advisor

A beginner-friendly AI/ML web application that classifies waste as **Biodegradable**, **Non-Biodegradable**,  **recyclable** or **E-waste** using **Machine Learning** and **AI** (Decision Tree Classifier). Built with **Python**, **Streamlit**, and **Pillow**, this tool also offers filename-based waste prediction via image upload.

---

## 🚀 Features

- ✅ Waste type classification using user input (text)
- 🖼️ Image upload support (filename-based classification using keywords)
- 🧠 Machine Learning backend (Decision Tree Classifier)
- 📊 Trained on a custom dataset (`dataset.csv`)
- ⚙️ Web interface using Streamlit
- 📦 Clean and modular code with `train_model.py` and `smart_waste_segregator_pro.py`

---

## 🧠 Technologies & Libraries Used

- **Python 3.x**
- **Scikit-learn** (Machine Learning)
- **Pandas** (Data handling)
- **Streamlit** (Web UI)
- **Pillow** (Image handling)
- **Joblib** (Model saving/loading)

---

## 📂 Project Structure
Smart-Waste-Segregation-Advisor/
│
├── dataset.csv # Training data for ML model
├── model.pkl # Trained Decision Tree model
├── train_model.py # Script to train and save the model
├── smart_waste_segregator_pro.py # Streamlit app
├── README.md # Project documentation
