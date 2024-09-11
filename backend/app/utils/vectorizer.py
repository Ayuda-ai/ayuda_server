from sklearn.feature_extraction.text import TfidfVectorizer

'''
This function takes a data_column and converts it into a vectorized matrix using TfidfVectorizer
'''
def get_vectorized_user_matrix(user_list, vectorizer):
    formatted_skills = ', '.join(user_list[0].split(', '))
    vectorized_matrix = vectorizer.transform([formatted_skills])
    return vectorized_matrix

def get_vectorized_course_matrix(course_list):
    vectorizer = TfidfVectorizer(stop_words=None, token_pattern=r'\b\w+\b', min_df=1)
    course_list['skills_associated'] = course_list['skills_associated'].apply(lambda x: ' '.join(x))
    vectorized_matrix = vectorizer.fit_transform(course_list['skills_associated'])
    return vectorized_matrix, vectorizer