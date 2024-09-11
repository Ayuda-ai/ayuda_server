import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

courses = pd.read_csv(r'data/mgen_courses.csv')

courses.head()

courses.shape

# converting the string to array of strings
for skills in courses['skills_associated']:
    courses['skills_associated'] = courses['skills_associated'].apply(lambda x: x.split(', ') if isinstance(x, str) else x)

courses.head()

course_skills_columns = ['course_id', 'course_name', 'skills_associated']
course_skills_map = pd.DataFrame(columns=course_skills_columns)
print(course_skills_map)

list_of_new_rows = []
for i in range(1, len(courses)+1):
    row = courses.iloc[i-1]
    # Create a new row for each course
    new_row = {
        'course_id': row['course_id'],
        'course_name': row['course_name'],
        'skills_associated': row['skills_associated'],
    }
    list_of_new_rows.append(new_row)
course_skills_map = pd.concat([course_skills_map, pd.DataFrame(list_of_new_rows)], ignore_index=True)

print(course_skills_map['skills_associated'])


# #### Vectorization of Course Skills
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create dummy user dataframe
user = pd.DataFrame({
    'user_id': 1,
    'user_name': 'Kartikey',
    'skills': ['object-oriented, java, spring, python, aws, springboot']
})
print(type(course_skills_map['skills_associated']))

vectorizer = TfidfVectorizer()
def vectorize_course_skills():
    course_skills_map['skills_associated'] = course_skills_map['skills_associated'].apply(lambda x: ' '.join(x))
    course_skill_matrix = vectorizer.fit_transform(course_skills_map['skills_associated'])
    return course_skill_matrix

def vectorize_user_skill():
    user_skill_matrix = vectorizer.transform(user['skills'])
    return user_skill_matrix

course_skill_matrix = vectorize_course_skills()
user_skill_matrix = vectorize_user_skill()

similarity_scores = cosine_similarity(user_skill_matrix, course_skill_matrix)
print(similarity_scores)

# Flatten similarity scores and course names
flat_scores = similarity_scores.flatten()
course_names = course_skills_map['course_name'].values
course_ids = course_skills_map['course_id'].values
print(flat_scores)


# Create a DataFrame to store course names and similarity scores
result_df = pd.DataFrame({'Course ID': course_ids, 'Course Name': course_names, 'Similarity Score': flat_scores})
# Sort by similarity score in descending order
result_df = result_df.sort_values(by='Similarity Score', ascending=False)

print("Recommendations for", user['user_name'])

# Print course names along with similarity scores
for idx, row in result_df.iterrows():
    print(f"{row['Course ID']}: {row['Course Name']}: {row['Similarity Score']}")
