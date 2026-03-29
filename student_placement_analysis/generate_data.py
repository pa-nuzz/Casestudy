import pandas as pd
import numpy as np

np.random.seed(42)
n = 200
domains = ['Data Science', 'MERN', 'Python', 'AI/ML', 'Web Development']

# Generate synthetic data
df = pd.DataFrame({
    'Attendance': np.random.randint(50, 101, n),
    'Assignment_Score': np.random.randint(50, 101, n),
    'Domain': np.random.choice(domains, n),
    'Placement_Status': np.random.choice(['Job', 'Intern', 'No Job'], n, p=[0.4,0.3,0.3])
})

df.to_csv('student_data.csv', index=False)
print('student_data.csv generated!')
