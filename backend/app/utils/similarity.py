from sklearn.metrics.pairwise import cosine_similarity

'''
This function takes 2 matrix and computes cosine similarity between them
'''
def get_cosine_similarity(user_skill_matrix, course_skill_matrix):
    similarity_scores = cosine_similarity(user_skill_matrix, course_skill_matrix)
    flat_scores = similarity_scores.flatten()
    return flat_scores



