# SalaryIQ — Employee Salary Predictor

A production-ready Flask web application for predicting employee salaries using a Random Forest ML model trained on global tech compensation data.

---

## Project Structure

```
salary_app/
├── app.py                  # Flask backend (API + routing)
├── train_model.py          # One-time model training script
├── requirements.txt
├── Procfile                # Heroku / Render deployment
├── Best_model/
│   └── random_forest_model.pkl    # (generated after training)
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/main.js
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

Place your CSV dataset (e.g. `job_salary_prediction_dataset.csv`) in the project root, then run:

```bash
python train_model.py --data job_salary_prediction_dataset.csv
```

This saves `Best_model/random_forest_model.pkl`.

> **If you already have a trained `.pkl`** from the notebook, copy it to `Best_model/` and rename it to `random_forest_model.pkl` (or `random_forest_tuned_model.pkl`).

### 3. Run the app

```bash
python app.py
```

Visit `http://localhost:5000`

---

## API Reference

### `POST /predict`

**Request body (JSON):**

```json
{
  "job_title":        "Data Scientist",
  "industry":         "Technology",
  "location":         "USA",
  "remote_work":      "Yes",
  "education_level":  "Master",
  "company_size":     "Large",
  "experience_years": 5,
  "skills_count":     8,
  "certifications":   2
}
```

**Response:**

```json
{
  "salary": 135200.45,
  "formatted": "$135,200"
}
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

---

## Deployment

### Heroku

```bash
heroku create your-app-name
git push heroku main
```

### Render / Railway

Set start command: `gunicorn app:app`

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
```

---

## Supported Values

| Field | Options |
|---|---|
| job_title | AI Engineer, Backend Developer, Business Analyst, Cloud Engineer, Cybersecurity Analyst, Data Analyst, Data Scientist, DevOps Engineer, Frontend Developer, Machine Learning Engineer, Product Manager, Software Engineer |
| industry | Consulting, Education, Finance, Government, Healthcare, Manufacturing, Media, Retail, Technology, Telecom |
| location | Australia, Canada, Germany, India, Netherlands, Remote, Singapore, Sweden, UK, USA |
| remote_work | Hybrid, No, Yes |
| education_level | Bachelor, Diploma, High School, Master, PhD |
| company_size | Enterprise, Large, Medium, Small, Startup |
