"""
auth.py
=======

This module provides user authentication and management for the Latxa-Instruct frontend.
It handles user registration, login, logout, and user data storage using JSON files and bcrypt for password hashing.
Custom exceptions are defined for common authentication errors.

License:
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at:

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Author:
    Oscar Sainz (oscar.sainz@ehu.eus)
"""

import json
import os
import bcrypt


class UserDoesNotExist(Exception):
    pass


class UserAlreadyExists(Exception):
    pass


class IncorrectPassword(Exception):
    pass


class IncorrectUsername(Exception):
    pass


class AuthManager:
    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.current_user = None

    def login(self, username: str, password: str) -> bool:
        username = username.strip()

        if not os.path.exists(os.path.join(self.dir_path, f"{username}.json")):
            raise UserDoesNotExist(f"User {username} does not exist")

        user = self.load_user(username)
        if bcrypt.checkpw(password, user.get("password")):
            self.current_user = username
            return True
        else:
            raise IncorrectPassword("Incorrect password")

    def register(
        self, username: str, password: str, email: str, hizk_maila: str, hezk_maila: str
    ) -> bool:
        # Register user in the database
        username = username.strip()

        if not username.isalnum():
            raise IncorrectUsername(
                "Erabiltzaile izen desegokia. Mesedez, karaktere alfanumerikoak bakarrik erabali."
            )

        if os.path.exists(os.path.join(self.dir_path, f"{username}.json")):
            raise UserAlreadyExists(f"User {username} already exists")

        hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())
        user = {
            "username": username,
            "password": hashed_password,
            "email": email,
            "hizk_maila": hizk_maila,
            "hezk_maila": hezk_maila,
        }

        with open(os.path.join(self.dir_path, f"{username}.json"), "w") as file:
            json.dump(user, file)

        self.current_user = username
        return True

    def load_user(self, username):
        with open(os.path.join(self.dir_path, f"{username}.json"), "r") as file:
            return json.load(file)

    def logout(self):
        self.current_user = None

    def get_current_user(self):
        return self.current_user

    def get_all_users(self):
        # "eneko.json".rstrip(".json") --> "enek"
        return [
            user.rstrip(".json")
            for user in os.listdir(self.dir_path)
            if user.endswith(".json")
        ]
