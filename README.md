# 🗑️ Smart Waste Management System

**AI-Powered Collection Optimization for Ghanaian Municipalities**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-green.svg)](https://xgboost.readthedocs.io/)

## 📋 Overview

This project develops a machine learning system to optimize waste collection in Ghanaian municipalities. Unlike generic solutions, it's specifically designed for African urban contexts with:

- **Mixed waste streams** with high organic content
- **Seasonal patterns** (rainy/dry seasons, Harmattan)
- **Cultural factors** (festivals like Homowo)
- **Resource constraints** requiring smart prioritization

## 🎯 Key Features

| Component | Description | Performance |
|:----------|:------------|:------------|
| **Fill Level Predictor** | Predicts bin fill level (low/medium/high/critical) | 86.6% accuracy |
| **Waste Type Classifier** | Classifies waste (organic, recyclable, hazardous, etc.) | 74.4% accuracy |
| **Priority Scorer** | Ranks bins for collection urgency | Real-time scoring |
| **Interactive Dashboard** | Streamlit app for monitoring and predictions | Live demo |

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/smart-waste-management-ml.git
cd smart-waste-management-ml
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard

```bash
streamlit run App/app.py
```

Open `http://localhost:8501` in your browser.

## 📁 Project Structure

```
smart-waste-management-ml/
├── App/
│   └── app.py                 # Streamlit dashboard
├── data/
│   ├── raw/                   # Raw datasets
│   └── processed/             # Feature-engineered data
├── models/                    # Trained model files
├── notebooks/
│   ├── EDA.ipynb                      # Exploratory Data Analysis
│   ├── Feature Engineering.ipynb
│   ├── Model Training.ipynb
│   ├── Priority Scoring Model.ipynb   # Priority scoring system
│   └── Waste Type Classification.ipynb
├── src/
│   ├── config.py             # Configuration settings
│   └── models/
│       ├── train.py          # Training pipeline
│       └── predict.py        # Prediction module
├── tests/
│   └── test_pipeline.py      # Unit tests
├── reports/
│   └── figures/              # Visualizations
└── requirements.txt
```

## 📊 Model Performance

### Fill Level Prediction

| Model | Accuracy | F1-Macro | CV Mean |
|:------|:---------|:---------|:--------|
| Decision Tree | 80.9% | 0.737 | 0.726 |
| Random Forest | 84.1% | 0.782 | 0.779 |
| **XGBoost** | **86.6%** | **0.810** | **0.803** |

**Top Features:**
1. `fill_rate_7day_avg` (29%)
2. `days_since_last_collection` (17%)
3. `location_type` (17%)

### Waste Type Classification

| Model | Accuracy | F1-Macro |
|:------|:---------|:---------|
| Decision Tree | 59.2% | 0.477 |
| **Random Forest** | **74.4%** | **0.570** |
| XGBoost | 76.3% | 0.561 |

## 🖥️ Dashboard Features

### 📊 Overview Tab
- Key metrics (total bins, critical count, avg fill level)
- Fill level distribution pie chart
- Waste type bar chart
- Location heatmap
- Collection trends over time

### 🔮 Predictions Tab
- Interactive form for bin characteristics
- Real-time fill level prediction
- Waste type classification
- Priority score calculation
- Recommended action

### 🚨 Priority Queue Tab
- Ranked list of bins by urgency
- Filter by priority level
- Export to CSV

### 📈 Model Performance Tab
- Accuracy and F1 scores
- Feature importance charts
- Cross-validation results

## 🔧 Training Your Own Models

```bash
python -m src.models.train --data-path data/processed/features.csv
```

Options:
- `--tune` - Enable hyperparameter tuning
- `--no-save` - Skip saving models
- `--version` - Model version string

## 🧪 Running Tests

```bash
pytest tests/test_pipeline.py -v
```

## 🌍 Domain Knowledge Encoded

The system incorporates realistic Ghanaian waste management patterns:

### Fill Rates by Location
| Location | Fill Rate/Day |
|:---------|:--------------|
| Market | 18% |
| Hospitality | 14% |
| Industrial | 12% |
| Commercial | 10% |
| Institutional | 8% |
| Residential | 5% |

### Seasonal Effects
- **Rainy Season** (Apr-Jul, Sep-Nov): -15% fill rate
- **Harmattan** (Dec-Feb): +10% fill rate (dust/debris)
- **Festivals** (Homowo, Christmas): +50% organic waste

## 🛠️ Technologies Used

- **Python 3.9+**
- **pandas, numpy** - Data manipulation
- **scikit-learn** - Machine learning
- **XGBoost** - Gradient boosting
- **Streamlit** - Interactive dashboard
- **Plotly** - Visualizations
- **joblib** - Model serialization

## 📈 Future Improvements

- [ ] Improve hazardous waste recall (currently 32%, target 85%)
- [ ] Add GPS-based route optimization
- [ ] Implement real-time IoT sensor integration
- [ ] Deploy REST API with FastAPI
- [ ] Mobile app for field workers

## 👤 Author

**David Quayefio**

- LinkedIn: [David Quayefio](https://www.linkedin.com/in/david-quayefio/)
- GitHub: [@david006-DS](https://github.com/david006-DS)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Domain knowledge informed by research on Ghana's sanitation sector
- Greater Accra district data approximated from public sources
- Weather patterns based on Ghana Meteorological Agency data

---

⭐ **If you found this project useful, please give it a star!**
