from ..utils.user_utils import get_current_user_email
import requests

def get_current_user_data():
    current_user_email = get_current_user_email()
    params = {
        'email': current_user_email,
    }
    user_data_response = requests.get('http://127.0.0.1:5000/api/v1/users/get-user', params=params)
    return user_data_response.json()
