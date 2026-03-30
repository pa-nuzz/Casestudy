"""
Student Placement Model Training - Balanced Classes
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load data
print("Loading data...")
df = pd.read_csv('student_data.csv')
print(f"Loaded {len(df)} rows")
print(f"\nPlacement distribution:")
print(df['Placement_Status'].value_counts())
print(f"\nPercentages:")
print(df['Placement_Status'].value_counts(normalize=True) * 100)

# Encode categorical variables
print("\nEncoding variables...")
le_domain = LabelEncoder()
df['Domain_enc'] = le_domain.fit_transform(df['Domain'])
le_status = LabelEncoder()
df['Placement_Status_enc'] = le_status.fit_transform(df['Placement_Status'])

print(f"Domain classes: {le_domain.classes_}")
print(f"Status classes: {le_status.classes_}")

# Save encoders
joblib.dump(le_domain, 'le_domain.pkl')
joblib.dump(le_status, 'le_status.pkl')

# Features and target
feature_cols = [
    'Attendance', 'Assignment_Score', 'Domain_enc', 'Project_Count',
    'Certifications', 'Hackathon_Participation', 'Soft_Skills_Score', 'Interview_Score'
]
X = df[feature_cols]
y = df['Placement_Status_enc']

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calculate class weights for imbalance handling
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {c: w for c, w in zip(classes, class_weights)}
print(f"\nClass weights: {class_weight_dict}")

# Train Random Forest with balanced class weights
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"Test Accuracy: {acc:.3f} ({acc*100:.1f}%)")
print(f"{'='*50}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_status.classes_))

# Cross-validation
print("\nCross-validation (5-fold):")
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# Feature importance
print("\nFeature Importance:")
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance)

# Test edge cases
print("\n" + "="*50)
print("Testing edge cases:")
print("="*50)

# Bad student (low scores)
bad_student = pd.DataFrame({
    'Attendance': [55],
    'Assignment_Score': [55],
    'Domain': ['Data Science'],
    'Project_Count': [0],
    'Certifications': [0],
    'Hackathon_Participation': [0],
    'Soft_Skills_Score': [55],
    'Interview_Score': [55]
})
bad_student['Domain_enc'] = le_domain.transform(bad_student['Domain'])
bad_X = bad_student[feature_cols]
bad_pred = model.predict(bad_X)
bad_proba = model.predict_proba(bad_X)[0]
print(f"\nBad student (low scores): {le_status.inverse_transform(bad_pred)[0]}")
print(f"Probabilities: {dict(zip(le_status.classes_, bad_proba))}")

# Good student (high scores)
good_student = pd.DataFrame({
    'Attendance': [95],
    'Assignment_Score': [95],
    'Domain': ['Data Science'],
    'Project_Count': [7],
    'Certifications': [4],
    'Hackathon_Participation': [1],
    'Soft_Skills_Score': [95],
    'Interview_Score': [95]
})
good_student['Domain_enc'] = le_domain.transform(good_student['Domain'])
good_X = good_student[feature_cols]
good_pred = model.predict(good_X)
good_proba = model.predict_proba(good_X)[0]
print(f"\nGood student (high scores): {le_status.inverse_transform(good_pred)[0]}")
print(f"Probabilities: {dict(zip(le_status.classes_, good_proba))}")

# Save model and data
df.to_csv('data_clean.csv', index=False)
joblib.dump(model, 'model.pkl')

print("\n" + "="*50)
print("Model and data saved successfully!")
print("="*50)
