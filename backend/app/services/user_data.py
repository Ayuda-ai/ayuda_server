import requests

def get_current_user_data(user_email):
    params = {
        'email': user_email,
    }
    user_data_response = requests.get('http://127.0.0.1:5000/api/v1/users/get-user', params=params)
    return user_data_response.json()
