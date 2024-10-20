# Ayuda_Server

This repository contains backend code for Ayuda.

## Local Dev Environment Setup

- Install Python 3.11.2 and MongoDB
- Install all the Python packages used in this project using `pip install requirements.txt`
- Create a `db.ini` file in the root directory (where run.py exists). Use `db.ini.txt` file for the reference to create the `db.ini` file
- Refer this [link](https://www.prisma.io/dataguide/mongodb/connection-uris#:~:text=A%20quick%20description%20of%20each,username%20%3A%20An%20optional%20username.) to know the format of a standard MongoDB URI that needs to be put in the `db.ini`
- BASE_URL is url where you need the backend Flask server to run.It can be `http://127.0.0.1:5000`
- Once done with the above setup, start the server by `python run.py` command
