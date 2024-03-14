from flask import Flask, request
from flask_restful import Api, Resource
import sqlite3
import threading

app = Flask(__name__)
api = Api(app)

import os

# Function to create a new database connection
def get_db():
    # Check if the database file exists
    if not os.path.exists('database.db'):
        # If the database file doesn't exist, create it
        conn = sqlite3.connect('database.db')
        # Create the users table
        conn.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    return sqlite3.connect('database.db')

class Login(Resource):
    def post(self):
        try:
            # Get username and password from request JSON
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')

            # Create a new database connection and cursor within the current thread
            with get_db() as conn:
                cur = conn.cursor()

                # Query the database to check if the user exists
                cur.execute("SELECT * FROM users WHERE username = ?", (username,))
                user = cur.fetchone()

                if user:
                    # Check if the provided password matches the password in the database
                    if password == user[2]:  # Assuming password is stored in the third column
                        return {'message': 'Login successful'}, 200
                    else:
                        return {'message': 'Invalid username or password'}, 401
                else:
                    return {'message': 'User does not exist. Please create a new user.'}, 404
        except Exception as e:
            return {'message': f'Error logging in: {str(e)}'}, 500

class CreateUser(Resource):
    def post(self):
        try:
            # Get username and password from request JSON
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')

            # Create a new database connection and cursor within the current thread
            with get_db() as conn:
                cur = conn.cursor()

                # Check if the username already exists in the database
                cur.execute("SELECT * FROM users WHERE username = ?", (username,))
                existing_user = cur.fetchone()

                if existing_user:
                    return {'message': 'Username already exists. Please choose a different username.'}, 400
                else:
                    # Insert the new user into the database
                    cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                    conn.commit()
                    return {'message': 'User created successfully'}, 201
        except Exception as e:
            return {'message': f'Error creating user: {str(e)}'}, 500

# Add the Login and CreateUser resources to the API
api.add_resource(Login, '/login')
api.add_resource(CreateUser, '/create_user')

if __name__ == '__main__':
    app.run(debug=True, port=5002)
