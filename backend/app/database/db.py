from flask_pymongo import PyMongo
import configparser, os

# function to get the config .ini file
def get_config():
    # Use configparser to load the .ini file
    config = configparser.ConfigParser()
    # config.read(os.path.abspath(os.path.join(".ini")))
    config.read(os.path.abspath("db.ini"))
    return config

def get_mongodb(app):
    # Set the MongoDB URI in the Flask app's configuration by retrieving the DB_URI under the DEV section
    db_uri = os.getenv("DB_URI")
    if db_uri:
        app.config["MONGO_URI"] = db_uri
    else:
        config = get_config()
        app.config["MONGO_URI"] = config['DEV']['DB_URI']
    mongo = PyMongo(app)
    return mongo
