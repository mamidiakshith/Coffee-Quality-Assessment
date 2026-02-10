# Automated Coffee Quality Assessment using Machine Learning

This repository presents the implementation, analysis, and deployment of the research work:

**Automated Coffee Quality Assessment: A Comparative Analysis Using Sensory Data**

The project proposes an automated and objective framework for evaluating coffee quality using supervised machine learning, significantly reducing the cost, time, and subjectivity associated with traditional manual coffee cupping.

---

## 📌 Abstract

Coffee quality assessment is traditionally performed by certified Q-graders through sensory evaluation, a process that is expensive, time-intensive, and prone to subjectivity. This research introduces a machine learning–based coffee grading system trained on expert-labeled sensory and physical attributes from the Coffee Quality Institute (CQI) dataset.  

The proposed system classifies Arabica coffee samples into four industry-aligned quality grades using Random Forest and Logistic Regression models. Experimental results demonstrate that the Random Forest model achieves superior performance, reaching accuracy levels of up to 96%, while maintaining interpretability through feature importance analysis.  

The solution enables scalable, rapid, and consistent quality evaluation, making it suitable for real-world adoption across coffee production and supply chains.

---

## 📊 Dataset Information

- **Source**: Coffee Quality Institute (CQI)
- **Samples**: 1,339 Arabica coffee samples
- **Geographic Coverage**: Multiple coffee-producing countries
- **Labeling**: Expert-graded by certified Q-graders

### Selected Features (14 total)

**Sensory Attributes (0–10 scale):**
- Aroma
- Flavor
- Acidity
- Body
- Balance
- Aftertaste

**Quality Indicators (0–10 scale):**
- Uniformity
- Clean Cup
- Sweetness
- Cupper Points

**Physical Characteristics:**
- Moisture Content
- Category One Defects
- Category Two Defects

---

## 🎯 Quality Grade Engineering

Quality scores are converted into categorical grades based on CQI standards:

| Score Range | Grade |
|------------|-------|
| ≥ 85 | Excellent |
| 80 – 84 | Very Good |
| 75 – 79 | Good |
| < 75 | Poor |

This categorization improves interpretability and aligns the model outputs with industry practices.

---

## 🧠 Machine Learning Models

### 1. Random Forest Classifier
- Captures complex non-linear relationships
- Robust to noise and feature interactions
- Provides feature importance for interpretability
- Achieved **94–96% accuracy**

### 2. Logistic Regression
- Used as a baseline linear classifier
- Requires feature scaling
- Performs well but struggles with closely ranked grades

---

## ⚙️ Methodology

1. Data cleaning and preprocessing  
2. Median imputation for missing values  
3. Feature selection and grade engineering  
4. Feature scaling using StandardScaler  
5. Train-test split (80:20)  
6. Hyperparameter tuning using Grid Search  
7. Model evaluation using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix  

---

## 📈 Results and Insights

- Random Forest significantly outperforms Logistic Regression
- Minimal misclassification observed in confusion matrix
- Most influential features:
  - Total Cup Points
  - Flavor
  - Aroma
  - Cupper Points
- Strong positive correlations among sensory attributes validate expert cupping logic

---

## 🌐 Live Deployment (Hugging Face)

The trained model has been deployed as an interactive web application on Hugging Face Spaces.

🔗 **Live Demo**  
https://huggingface.co/spaces/Sai1012/coffee

### Deployment Highlights
- Real-time quality prediction
- User-friendly interface
- Industry-aligned grade output
- Demonstrates real-world usability of the research system

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Hugging Face Spaces

---

## 📂 Repository Structure

```text
├── data/
│   └── coffee_quality.csv
├── notebooks/
│   └── analysis.ipynb
├── models/
│   └── random_forest_model.pkl
├── app/
│   └── app.py
├── Automated_Coffee_Quality_Assessment.pdf
├── README.md
└── requirements.txt


📄 Research Paper

The complete research paper is included in this repository:

Automated_Coffee_Quality_Assessment.pdf

If you use this work for academic or research purposes, please cite the paper.

👥 Authors

Shruthi Kodi

Anuroop Kothireddy

Kishore Mainelly

Sai Venkata Karthik Mallala

Akshith Mamidi

Sreeja Palla

Department of Computer Science and Engineering
VNR Vignana Jyothi Institute of Engineering and Technology
Hyderabad, India
