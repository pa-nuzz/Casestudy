import pandas as pd
import numpy as np

np.random.seed(42)

# Generate 10,000 rows with proper class distribution
n = 10000
domains = ['Data Science', 'MERN', 'DevOps', 'Cybersecurity', 'AI/ML']

# Generate features
attendance = np.random.randint(50, 101, n)
assignment_score = np.random.randint(50, 101, n)
project_count = np.random.randint(0, 8, n)
certifications = np.random.randint(0, 6, n)
hackathon = np.random.randint(0, 2, n)
soft_skills = np.random.randint(50, 101, n)
interview_score = np.random.randint(50, 101, n)
domain = np.random.choice(domains, n)

# Calculate composite score for placement logic
placement_status = []
for i in range(n):
    # Weighted composite score
    composite = (attendance[i] * 0.15 + 
                 assignment_score[i] * 0.20 + 
                 project_count[i] * 10 + 
                 certifications[i] * 8 + 
                 hackathon[i] * 12 +
                 soft_skills[i] * 0.10 + 
                 interview_score[i] * 0.20)
    
    # Add some randomness
    composite += np.random.normal(0, 15)
    
    # Fixed thresholds for 3-class distribution:
    # ~55% Job, ~25% Intern, ~20% No Job
    if composite > 80:
        status = 'Job'
    elif composite > 65:
        status = np.random.choice(['Job', 'Intern'], p=[0.6, 0.4])
    elif composite > 50:
        status = np.random.choice(['Intern', 'No Job'], p=[0.7, 0.3])
    elif composite > 35:
        status = np.random.choice(['No Job', 'Intern'], p=[0.8, 0.2])
    else:
        status = 'No Job'
    
    placement_status.append(status)

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

# Print distribution
dist = df['Placement_Status'].value_counts()
print(f"\nGenerated {n} rows")
print(f"Class Distribution:")
print(f"  Job: {dist.get('Job', 0)} ({dist.get('Job', 0)/n*100:.1f}%)")
print(f"  Intern: {dist.get('Intern', 0)} ({dist.get('Intern', 0)/n*100:.1f}%)")
print(f"  No Job: {dist.get('No Job', 0)} ({dist.get('No Job', 0)/n*100:.1f}%)")