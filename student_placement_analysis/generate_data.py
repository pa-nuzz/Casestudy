import pandas as pd
import numpy as np

np.random.seed(42)
n = 10000

domains = ['Data Science', 'MERN', 'DevOps', 'Cybersecurity']

# Generate base features
attendance = np.random.randint(50, 101, n)
assignment_score = np.random.randint(50, 101, n)
project_count = np.random.randint(0, 8, n)
certifications = np.random.randint(0, 5, n)
hackathon = np.random.randint(0, 2, n)
soft_skills = np.random.randint(50, 101, n)
interview_score = np.random.randint(50, 101, n)
domain = np.random.choice(domains, n)

# Calculate composite score (0-100 scale)
placement_status = []
for i in range(n):
    # Normalize features to 0-100 scale
    att = attendance[i]
    assign = assignment_score[i]
    proj = project_count[i] * 12.5  # 0-7 -> 0-87.5
    cert = certifications[i] * 20    # 0-4 -> 0-80
    hack = hackathon[i] * 50         # 0-1 -> 0-50
    soft = soft_skills[i]
    interview = interview_score[i]
    
    # Calculate weighted composite score
    composite = (
        att * 0.15 +
        assign * 0.15 +
        proj * 0.15 +
        cert * 0.15 +
        hack * 0.10 +
        soft * 0.15 +
        interview * 0.15
    )
    
    # Add some randomness
    composite += np.random.normal(0, 8)
    composite = np.clip(composite, 0, 100)
    
    # Clear separation based on composite score with balanced classes
    if composite >= 75:
        status = 'Job'
    elif composite >= 55:
        status = np.random.choice(['Job', 'Intern'], p=[0.4, 0.6])
    elif composite >= 40:
        status = np.random.choice(['Intern', 'No Job'], p=[0.6, 0.4])
    else:
        status = 'No Job'
    
    placement_status.append(status)

# Create DataFrame
df = pd.DataFrame({
    'Attendance': attendance,
    'Assignment_Score': assignment_score,
    'Domain': domain,
    'Placement_Status': placement_status,
    'Project_Count': project_count,
    'Certifications': certifications,
    'Hackathon_Participation': hackathon,
    'Soft_Skills_Score': soft_skills,
    'Interview_Score': interview_score
})

df.to_csv('student_data.csv', index=False)
print(f'Generated {n} rows of student data with {len(df.columns)} features!')
print(f'Data saved to student_data.csv')
print(f'\nPlacement distribution:')
print(df['Placement_Status'].value_counts())
print(f'\nPercentages:')
print(df['Placement_Status'].value_counts(normalize=True) * 100)
