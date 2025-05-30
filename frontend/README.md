# Latxa-Instruct Frontend

This directory contains the main user-facing application for the Latxa-Instruct project. The frontend is built with [Gradio](https://www.gradio.app/) and provides an interactive web interface for evaluating and comparing AI chat models, collecting user feedback, and managing user authentication.

## Contents

- `arena_with_user.py`: Main Gradio app. Handles user registration, login, prompt submission, model response comparison, feedback collection, and leaderboard display. Integrates with backend model endpoints and manages user sessions.
- `api.py`: Abstraction layer for integrating multiple AI model APIs (OpenAI, Google, Cohere, Anthropic, vLLM) with a unified interface for chat-based interactions.
- `auth.py`: User authentication and management. Handles registration, login, logout, and user data storage using JSON files and bcrypt for password hashing.
- `scoring.py`: Functions for evaluating model responses, computing ELO ratings, generating leaderboards, and summarizing user contributions.
- `style.py`: Custom Gradio theme classes for the frontend, providing a clean and modern user interface.


## Usage

To launch the frontend, ensure all dependencies are installed and the backend is running with a valid `partial_config.jsonl` file. Then run:

```sh
python3 arena_with_user.py
```

The app will be available at the configured host and port (default: `http://localhost:7887`).

## Requirements

- Python 3.9+
- [Gradio](https://www.gradio.app/)
- [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub)
- Other dependencies as specified in your environment or `requirements.txt`

## Notes

- The frontend expects the backend to provide a `partial_config.jsonl` file with model endpoint information.
- User data is stored locally in the `data/users` directory as JSON files.
- Feedback and conversation data are saved locally and optionally uploaded to a Hugging Face dataset repository.
- The UI is designed for Basque-speaking users and supports evaluation of Basque-language chatbots.

For more details, see the docstrings and comments in each script.