from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# ─────────────────────────────────────────────
#  Feature & encoding config (mirrors notebook)
# ─────────────────────────────────────────────

FEATURE_COLUMNS = [
    'job_title_AI Engineer', 'job_title_Backend Developer',
    'job_title_Business Analyst', 'job_title_Cloud Engineer',
    'job_title_Cybersecurity Analyst', 'job_title_Data Analyst',
    'job_title_Data Scientist', 'job_title_DevOps Engineer',
    'job_title_Frontend Developer', 'job_title_Machine Learning Engineer',
    'job_title_Product Manager', 'job_title_Software Engineer',

    'industry_Consulting', 'industry_Education', 'industry_Finance',
    'industry_Government', 'industry_Healthcare', 'industry_Manufacturing',
    'industry_Media', 'industry_Retail', 'industry_Technology',
    'industry_Telecom',

    'location_Australia', 'location_Canada', 'location_Germany',
    'location_India', 'location_Netherlands', 'location_Remote',
    'location_Singapore', 'location_Sweden', 'location_UK', 'location_USA',

    'remote_work_Hybrid', 'remote_work_No', 'remote_work_Yes',

    'education_level', 'company_size',
    'experience_years', 'skills_count', 'certifications',
]

EDUCATION_MAP = {
    'Bachelor': 0, 'Diploma': 1, 'High School': 2, 'Master': 3, 'PhD': 4
}

COMPANY_SIZE_MAP = {
    'Enterprise': 0, 'Large': 1, 'Medium': 2, 'Small': 3, 'Startup': 4
}

JOB_TITLES = [
    'AI Engineer', 'Backend Developer', 'Business Analyst', 'Cloud Engineer',
    'Cybersecurity Analyst', 'Data Analyst', 'Data Scientist', 'DevOps Engineer',
    'Frontend Developer', 'Machine Learning Engineer', 'Product Manager',
    'Software Engineer',
]

INDUSTRIES = [
    'Consulting', 'Education', 'Finance', 'Government', 'Healthcare',
    'Manufacturing', 'Media', 'Retail', 'Technology', 'Telecom',
]

LOCATIONS = [
    'Australia', 'Canada', 'Germany', 'India', 'Netherlands',
    'Remote', 'Singapore', 'Sweden', 'UK', 'USA',
]

REMOTE_OPTIONS  = ['Hybrid', 'No', 'Yes']
EDUCATION_LEVELS = list(EDUCATION_MAP.keys())
COMPANY_SIZES    = list(COMPANY_SIZE_MAP.keys())

# ─────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────

MODEL_PATHS = [
    'Best_model/random_forest_tuned_model.pkl',
    'Best_model/random_forest_model.pkl',
]

model = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        model = joblib.load(path)
        print(f"[INFO] Loaded model from {path}")
        break

if model is None:
    print("[WARN] No pre-trained model found. Train one first or place a .pkl in Best_model/")


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template(
        'index.html',
        job_titles=JOB_TITLES,
        industries=INDUSTRIES,
        locations=LOCATIONS,
        remote_options=REMOTE_OPTIONS,
        education_levels=EDUCATION_LEVELS,
        company_sizes=COMPANY_SIZES,
    )


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train and save a model first.'}), 503

    try:
        payload = request.get_json(force=True)

        job_title       = payload['job_title']
        industry        = payload['industry']
        location        = payload['location']
        remote_work     = payload['remote_work']
        education_level = payload['education_level']
        company_size    = payload['company_size']
        experience_years = float(payload['experience_years'])
        skills_count     = int(payload['skills_count'])
        certifications   = int(payload['certifications'])

        # Build feature vector
        data = dict.fromkeys(FEATURE_COLUMNS, 0)

        data['education_level']  = EDUCATION_MAP[education_level]
        data['company_size']     = COMPANY_SIZE_MAP[company_size]
        data['experience_years'] = experience_years
        data['skills_count']     = skills_count
        data['certifications']   = certifications

        for prefix, value in [
            ('job_title', job_title),
            ('industry',  industry),
            ('location',  location),
            ('remote_work', remote_work),
        ]:
            col = f'{prefix}_{value}'
            if col in data:
                data[col] = 1

        input_df = pd.DataFrame([data])[FEATURE_COLUMNS]
        salary   = float(model.predict(input_df)[0])

        return jsonify({
            'salary': round(salary, 2),
            'formatted': f"${salary:,.0f}",
        })

    except KeyError as e:
        return jsonify({'error': f'Missing field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
