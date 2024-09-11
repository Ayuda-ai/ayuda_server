from vectorizer import get_vectorized_matrix
import pandas as pd

# user = pd.DataFrame({
#     'user_id': 1,
#     'user_name': 'Kartikey',
#     'skills': ['java', 'python', 'javascript']
# })

# user_skills_map = pd.DataFrame(columns=) # converting the list to pandas df
user_skills = ['java', 'python', 'javascript']
user_skills_vectorized = get_vectorized_matrix(user_skills)

print(user_skills_vectorized)