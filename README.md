# ✈️ British Airways — Customer Booking Predictive Model & Lounge Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-85.20%25-brightgreen?style=for-the-badge)](#)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.77-blueviolet?style=for-the-badge)](#)

Completed as part of the **British Airways Data Science Job Simulation** on Forage. This repository contains end-to-end Machine Learning pipelines to predict customer booking completion and an algorithmic engine for Heathrow Terminal 3 lounge eligibility.

---

## 📌 Project Overview
* **Dataset Scale**: 50,000+ flight records across international routes.
* **Core Task**: Predict whether a customer will complete a booking conversion based on purchase lead times, ancillary selections, duration of stay, and route characteristics.
* **Model**: Optimized Random Forest Classifier with cross-validation.
* **Accuracy**: **85.20%** test accuracy | **0.77 ROC-AUC** score | **84.99%** 5-fold cross-validation accuracy.

---

## 📊 Key Insights & Feature Importances
1. **Purchase Lead Time**: Long lead times significantly alter conversion probability; early planners exhibit distinct cancellation patterns.
2. **Flight Duration & Length of Stay**: Transatlantic and long-haul flights have higher commitment rates once selected.
3. **Ancillary Purchases**: Requests for preferred seating and extra baggage serve as direct signals of high completion intent.

---

## 🚀 Quickstart

```bash
# Clone the repository
git clone https://github.com/mrprashantkumar2323-ctrl/British-Airways-Customer-Booking-Prediction.git
cd British-Airways-Customer-Booking-Prediction

# Install dependencies
pip install -r requirements.txt

# Run predictive ML pipeline
python predictive_model.py

# Run lounge eligibility engine
python lounge_eligibility.py
```

---

## 👨‍💻 Author
**Prashant Kumar**  
* [LinkedIn](https://www.linkedin.com/in/prashant-kumar-p453) • [GitHub](https://github.com/mrprashantkumar2323-ctrl) • [Email](mailto:mr.prashantkumar2323@gmail.com)
