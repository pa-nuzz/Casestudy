import pandas as pd
import joblib

model = joblib.load('model.pkl')
le_domain = joblib.load('le_domain.pkl')
le_status = joblib.load('le_status.pkl')

feature_cols = ['Attendance', 'Assignment_Score', 'Domain_enc', 'Project_Count',
    'Certifications', 'Hackathon_Participation', 'Soft_Skills_Score', 'Interview_Score']

test_cases = [
    ("Very Bad Student (50/50 attendance/score)", [50, 50, 'Data Science', 0, 0, 0, 50, 50]),
    ("Bad Student (60/60)", [60, 60, 'Data Science', 1, 0, 0, 60, 60]),
    ("Average Student (75/75)", [75, 75, 'Data Science', 3, 2, 0, 75, 75]),
    ("Good Student (85/85, 5 projects)", [85, 85, 'Data Science', 5, 3, 1, 85, 85]),
    ("Excellent Student (95/95, 7 projects)", [95, 95, 'Data Science', 7, 4, 1, 95, 95]),
]

print("Prediction Tool Test Results:")
print("="*70)
for name, values in test_cases:
    df = pd.DataFrame({
        'Attendance': [values[0]],
        'Assignment_Score': [values[1]],
        'Domain': [values[2]],
        'Project_Count': [values[3]],
        'Certifications': [values[4]],
        'Hackathon_Participation': [values[5]],
        'Soft_Skills_Score': [values[6]],
        'Interview_Score': [values[7]]
    })
    df['Domain_enc'] = le_domain.transform(df['Domain'])
    X = df[feature_cols]
    pred = model.predict(X)
    proba = model.predict_proba(X)[0]
    result = le_status.inverse_transform(pred)[0]
    print(f"\n{name}:")
    print(f"  Prediction: {result}")
    print(f"  Probabilities: Job={proba[1]:.2%}, Intern={proba[0]:.2%}, No Job={proba[2]:.2%}")
print("\n" + "="*70)
