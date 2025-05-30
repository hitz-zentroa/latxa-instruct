"""
arena_with_user.py
==================

This module implements the main Gradio-based frontend for the Latxa-Instruct project,
enabling interactive evaluation of multiple AI chat models by end users. It provides
the user interface for logging in, registering, submitting prompts, comparing model
responses, and submitting feedback. The script manages user authentication, leaderboard
display, and feedback collection, and integrates with various backend model endpoints
as defined in the configuration file.

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

import datetime
import json
import logging
import os
import random
import re
import string
import time
import uuid
from functools import partial

import gradio as gr
import huggingface_hub

from gradio_modal import Modal
import pandas as pd
import rich

import api
import style
from scoring import get_results_dfs, get_user_dfs
from auth import AuthManager

auth_token = os.environ.get("TOKEN") or True
hf_repo = "HiTZ/Feedback_LatxaTxat_Ebaluatoia_Mar19"
huggingface_hub.create_repo(
    repo_id=hf_repo,
    repo_type="dataset",
    token=auth_token,
    exist_ok=True,
    private=True,
)
# os.makedirs("data", exist_ok=True)
os.makedirs("data/messages", exist_ok=True)
os.makedirs(os.path.join("data", hf_repo), exist_ok=True)

# Load model endpoints
if not os.path.exists("backend/partial_config.jsonl"):
    raise FileNotFoundError("Configuration file not found")

with open("backend/partial_config.jsonl", "r") as f:
    configs = [json.loads(line) for line in f]

configs = {line["model_name"]: line for line in configs}


STOP_STRINGS = "<|eot_id|>"

GENERATION_CONFIGS = {
    "default": api.GenerationConfig(
        max_tokens=2048,
        temperature=0.9,
        top_p=0.95,
        frequency_penalty=0.0,
        # frequency_penalty=0.0,  # This is only for repetition testing purposes
        stop=STOP_STRINGS,
        repetition_penalty=1.0,
    ),
    "low_temp": api.GenerationConfig(
        max_tokens=2048,
        temperature=0.3,
        top_p=0.95,
        frequency_penalty=0.0,
        # frequency_penalty=0.0,  # This is only for repetition testing purposes
        stop=STOP_STRINGS,
        repetition_penalty=1.0,
    ),
    "low_temp-penalty": api.GenerationConfig(
        max_tokens=2048,
        temperature=0.3,
        top_p=0.95,
        frequency_penalty=0.0,
        # frequency_penalty=0.0,  # This is only for repetition testing purposes
        stop=STOP_STRINGS,
        repetition_penalty=1.1,
    ),
}
generation_config = GENERATION_CONFIGS["default"]

endpoints = {}

_MODEL_NAME = "exp_0_010" # 
endpoints[_MODEL_NAME] = api.vLLMAPI(
    url=f"http://{configs[_MODEL_NAME]['host']}:{configs[_MODEL_NAME]['port']}/v1",
    model_name=_MODEL_NAME,
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)
_MODEL_NAME = "exp_0_011" #
endpoints[_MODEL_NAME] = api.vLLMAPI(
    url=f"http://{configs[_MODEL_NAME]['host']}:{configs[_MODEL_NAME]['port']}/v1",
    model_name=_MODEL_NAME,
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)
_MODEL_NAME = "exp_0_101" # 
endpoints[_MODEL_NAME] = api.vLLMAPI(
    url=f"http://{configs[_MODEL_NAME]['host']}:{configs[_MODEL_NAME]['port']}/v1",
    model_name=_MODEL_NAME,
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)
_MODEL_NAME = "exp_0_110" # 
endpoints[_MODEL_NAME] = api.vLLMAPI(
    url=f"http://{configs[_MODEL_NAME]['host']}:{configs[_MODEL_NAME]['port']}/v1",
    model_name=_MODEL_NAME,
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)
_MODEL_NAME = "exp_0_111" # 
endpoints[_MODEL_NAME] = api.vLLMAPI(
    url=f"http://{configs[_MODEL_NAME]['host']}:{configs[_MODEL_NAME]['port']}/v1",
    model_name=_MODEL_NAME,
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)
_MODEL_NAME = "exp_1_001" # 
endpoints[_MODEL_NAME] = api.vLLMAPI(
    url=f"http://{configs[_MODEL_NAME]['host']}:{configs[_MODEL_NAME]['port']}/v1",
    model_name=_MODEL_NAME,
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)
_MODEL_NAME = "exp_1_010" #
endpoints[_MODEL_NAME] = api.vLLMAPI(
    url=f"http://{configs[_MODEL_NAME]['host']}:{configs[_MODEL_NAME]['port']}/v1",
    model_name=_MODEL_NAME,
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)
_MODEL_NAME = "exp_1_011" # 
endpoints[_MODEL_NAME] = api.vLLMAPI(
    url=f"http://{configs[_MODEL_NAME]['host']}:{configs[_MODEL_NAME]['port']}/v1",
    model_name=_MODEL_NAME,
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)
_MODEL_NAME = "exp_1_101" #
endpoints[_MODEL_NAME] = api.vLLMAPI(
    url=f"http://{configs[_MODEL_NAME]['host']}:{configs[_MODEL_NAME]['port']}/v1",
    model_name=_MODEL_NAME,
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)
_MODEL_NAME = "exp_1_110" #
endpoints[_MODEL_NAME] = api.vLLMAPI(
    url=f"http://{configs[_MODEL_NAME]['host']}:{configs[_MODEL_NAME]['port']}/v1",
    model_name=_MODEL_NAME,
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)
_MODEL_NAME = "exp_1_111" #
endpoints[_MODEL_NAME] = api.vLLMAPI(
    url=f"http://{configs[_MODEL_NAME]['host']}:{configs[_MODEL_NAME]['port']}/v1",
    model_name=_MODEL_NAME,
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)
_MODEL_NAME = "exp_2_010" # 
endpoints[_MODEL_NAME] = api.vLLMAPI(
    url=f"http://{configs[_MODEL_NAME]['host']}:{configs[_MODEL_NAME]['port']}/v1",
    model_name=_MODEL_NAME,
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)
_MODEL_NAME = "exp_2_011" #
endpoints[_MODEL_NAME] = api.vLLMAPI(
    url=f"http://{configs[_MODEL_NAME]['host']}:{configs[_MODEL_NAME]['port']}/v1",
    model_name=_MODEL_NAME,
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)
_MODEL_NAME = "exp_2_100" # 
endpoints[_MODEL_NAME] = api.vLLMAPI(
    url=f"http://{configs[_MODEL_NAME]['host']}:{configs[_MODEL_NAME]['port']}/v1",
    model_name=_MODEL_NAME,
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)
_MODEL_NAME = "exp_2_101" # 
endpoints[_MODEL_NAME] = api.vLLMAPI(
    url=f"http://{configs[_MODEL_NAME]['host']}:{configs[_MODEL_NAME]['port']}/v1",
    model_name=_MODEL_NAME,
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)
_MODEL_NAME = "exp_2_110" #
endpoints[_MODEL_NAME] = api.vLLMAPI(
    url=f"http://{configs[_MODEL_NAME]['host']}:{configs[_MODEL_NAME]['port']}/v1",
    model_name=_MODEL_NAME,
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)
_MODEL_NAME = "exp_2_111" #
endpoints[_MODEL_NAME] = api.vLLMAPI(
    url=f"http://{configs[_MODEL_NAME]['host']}:{configs[_MODEL_NAME]['port']}/v1",
    model_name=_MODEL_NAME,
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)

#
# Llama-3.1 Instruct endpoint
endpoints["Llama-3.1-8B-Instruct"] = api.vLLMAPI(
    url=f"http://{configs['Llama-3.1-8B-Instruct']['host']}:{configs['Llama-3.1-8B-Instruct']['port']}/v1",
    model_name="Llama-3.1-8B-Instruct",
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)

# Latxa 3.1 70B endpoint
_MODEL_NAME = "Latxa-Llama-3.1-70B-Instruct-exp_2_101"
endpoints["Latxa 3.1 70B Instruct"] = api.vLLMAPI(
    url=f"http://{configs[_MODEL_NAME]['host']}:{configs[_MODEL_NAME]['port']}/v1",
    model_name=_MODEL_NAME,
    api_key="EMPTY",
    sysprompt_format="system",
    generation_config=GENERATION_CONFIGS["default"],
)
#
endpoints["gpt-4o"] = api.OpenAIAPI(
    model_name="gpt-4o-2024-11-20", generation_config=generation_config
)
#
endpoints["claude-3-5-sonnet-20241022"] = api.AnthropicAPI(
    model_name="claude-3-5-sonnet-20241022", generation_config=generation_config
)

ALL_MODEL_NAMES = tuple(endpoints.keys())

ANOM_NAMES = {
    key: f"{name} Eredua"
    # key: key if not "318" in key else "Latxa (berria)"
    for key, name in zip(
        endpoints.keys(),
        random.sample(string.ascii_uppercase[: len(endpoints)], len(endpoints)),
    )
}
# Comment this line to remove the de-anonimization
ANOM_NAMES = {key: key for key in endpoints.keys()}
rich.print(ANOM_NAMES)

USER_WELCOME_MSG = """
<div style="padding-bottom: 10px;">
    <h1>Ongi etorri, {user}!</h1>
    Sailkapena: <b>{position}</b> <br> 
    Egindako kontribuzioak: <b>{contributions}</b> <br>
    Boleto kopurua: <b>{boletoak}</b>
</div>
"""


def get_ttl_hash(seconds=60):
    """Return the same value withing `seconds` time period"""
    return round(time.time() / seconds)


def render_user_leaderboard(user):
    print(f"Rendering leaderboard for {user}")
    user_contributions, submissions_df = get_user_dfs(
        repo_id=hf_repo,
        model_names=ALL_MODEL_NAMES,
        ttl_hash=get_ttl_hash(),
        from_local=True,
    )

    if len(submissions_df):
        submissions_df["timestamp"] = pd.to_datetime(
            submissions_df["timestamp"]
        ).dt.strftime("%Y-%m-%d")
        submissions_df["is_user"] = submissions_df["username"] == user
        submissions_df["is_user"] = submissions_df["is_user"].map(
            {False: "Besteak", True: user}
        )
        submissions_df = (
            submissions_df.groupby(["timestamp", "is_user"]).count().reset_index()
        )

        submissions_barplot = gr.BarPlot(
            value=submissions_df,
            x="timestamp",
            y="model_a",
            # y_aggregate="count",
            color="is_user",
            # x_bin="1d",
            render=True,
            x_title="Eguna",
            y_title="Bidalketak",
            color_map={user: "#10b981", "Besteak": "#e5e7eb"},
        )
    else:
        submissions_barplot = gr.BarPlot(
            value=pd.DataFrame(), x="timestamp", y="submissions", x_bin="d"
        )

    return user_contributions, submissions_barplot


def render_leaderboard():
    print("Rendering leaderboard")
    user_contributions, submissions_df, leaderboard, barplot_df = get_results_dfs(
        repo_id=hf_repo,
        BOOTSTRAP_ROUNDS=100,
        model_names=ALL_MODEL_NAMES,
        ttl_hash=get_ttl_hash(),
        from_local=True,
    )
    leaderboard = leaderboard.copy(deep=True)
    leaderboard["Model"] = leaderboard["Model"].map(ANOM_NAMES)

    barplot_df = barplot_df.copy(deep=True)

    barplot_df["result"] = barplot_df["result"].map(
        {"win": "Irabazi", "loss": "Galdu", "tie": "Berdinketa"}
    )
    barplot_df["_sort"] = barplot_df["result"].map(
        {"Irabazi": 0, "Galdu": 2, "Berdinketa": 1}
    )
    barplot_df = barplot_df.sort_values(by="_sort", ascending=True)
    barplot_df["model_a"] = barplot_df["model_a"].map(ANOM_NAMES)

    barplot = gr.BarPlot(
        value=barplot_df,
        x="model_a",
        y="model_b",
        color="result",
        render=True,
        y_title="Konparaketak",
        x_title="Hizkuntza-Eredua",
        sort=leaderboard.Model.tolist(),
        color_map={"Irabazi": "#10b981", "Berdinketa": "#e5e7eb", "Galdu": "#60a5fa"},
    )

    submissions_df["timestamp"] = pd.to_datetime(submissions_df["timestamp"])
    submissions_barplot = gr.BarPlot(
        value=submissions_df,
        x="timestamp",
        y="submissions",
        x_bin="1d",
        render=True,
        x_title="Eguna",
        y_title="Bidalketak",
    )

    return user_contributions, submissions_barplot, leaderboard, barplot


def today():
    return time.strftime("%A %B %e, %Y", time.gmtime())


def update_models(only_models=False):
    random.seed(time.time())
    if not len(endpoints):
        raise ValueError("Not enough models available")
    if len(endpoints) < 2:
        logging.warning("Only one model available, comparing the same model!.")
        _model = list(endpoints.keys())[0]
        models = [_model, _model]
    else:
        models = random.sample(list(endpoints.keys()), 2)
        logging.warning(f"Models updated: {models}")

    if only_models:
        return models

    msg_textbox = gr.Textbox(
        label="Sartu zure mezua hemen",
        submit_btn=True,
        interactive=True,
    )

    return (
        models,
        str(uuid.uuid4()),
        gr.Button("📩 Bidali balorazioa", interactive=False),
        msg_textbox,
    )


def format_history(history):
    history_openai_format = []
    if api.system_prompt is not None:
        history_openai_format.append(
            {"role": "system", "content": api.system_prompt.format(date=today())}
        )

    for message in history:
        history_openai_format.append(
            {"role": message["role"], "content": message["content"]}
        )
    return history_openai_format


def predict_single_stream(
    message,
    models,
    history,
    conv_id,
    repeat_last=False,
    model_id=None,
):
    if not history:
        history = []

    if message == "" and not repeat_last:
        return (
            "",
            history,
            gr.Button("📩 Bidali balorazioa", interactive=False),
        )

    if repeat_last and len(history) == 0:
        gr.Warning("Nothing to repeat")
        return (
            "",
            history,
            gr.Button("📩 Bidali balorazioa", interactive=False),
        )

    if repeat_last:
        history.pop()
    else:
        history.append({"role": "user", "content": message})

    gen = endpoints[models[model_id]].get_chat_stream(history)

    for _history in gen:
        history = _history
        yield (
            "",
            history,
            gr.Button("📩 Bidali balorazioa", interactive=False),
        )

    if not conv_id:
        conv_id = str(uuid.uuid4())
    path = f"history_{conv_id}_{model_id}.json"
    path = os.path.join("data/messages", path)
    with open(path, "w") as f:
        json.dump(history, ensure_ascii=False, indent=4, fp=f)


def save_history(
    models,
    history_0,
    history_1,
    user_info,
    conv_id,
    eduki_bal,
    euskara_bal,
    orokorra_bal,
):
    winner_content = {
        "👈 A da hobea": "model_a",
        "🤝 Berdinketa": "tie",
        "👉 B da hobea": "model_b",
    }[eduki_bal]

    winner_language = {
        "👈 A da hobea": "model_a",
        "🤝 Berdinketa": "tie",
        "👉 B da hobea": "model_b",
    }[euskara_bal]

    if not orokorra_bal:
        # Hau erroreak ekiditeko da, printzipioz orokorra_bal None bada gero
        # ez luke eraginik eduki beharko.
        orokorra_bal = eduki_bal if eduki_bal != "🤝 Berdinketa" else euskara_bal
    winner_general = {
        "👈 A da hobea": "model_a",
        "🤝 Berdinketa": "tie",
        "👉 B da hobea": "model_b",
    }[orokorra_bal]

    winner = winner_content if winner_content == winner_language else winner_general
    print(winner_language, winner_content, winner)

    if not conv_id:
        conv_id = str(uuid.uuid4())

    path = f"history_{conv_id}.json"
    hf_path = os.path.join("data", path)
    local_path = os.path.join("data", hf_repo, path)
    data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_a": models[0],
        "model_b": models[1],
        "hyperparameters": generation_config.__dict__,
        "hyperparemeters_a": endpoints[models[0]].generation_config.__dict__,
        "hyperparameters_b": endpoints[models[0]].generation_config.__dict__,
        "conv_a": endpoints[models[0]].format_history(
            history_0
        ),  # format_history(history_0),
        "conv_b": endpoints[models[1]].format_history(
            history_1
        ),  # format_history(history_1),
        "winner": winner,
        "winner_content": winner_content,
        "winner_language": winner_language,
        "username": user_info,
    }

    with open(local_path, "w") as f:
        json.dump(data, ensure_ascii=False, indent=4, fp=f)

    try: # Avoid showing an error in the interface because of upload fail
        huggingface_hub.upload_file(
            repo_id=hf_repo,
            repo_type="dataset",
            token=os.environ.get("TOKEN") or True,
            path_in_repo=hf_path,
            path_or_fileobj=local_path,
        )
    except Exception as e:
        logging.error(e)

    gr.Info("Zure preferentzia ondo bidali da. Eskerrik asko parte hartzeagatik!")

    msg_textbox = gr.Textbox(
        label="Sartu zure mezua hemen",
        value='Zure preferentzia bidali da! Elkarrizketa berri bat hasteko sakatu "Txat berria" botoia.',
        submit_btn=False,
        stop_btn=False,
        interactive=False,
        elem_id="placeholder_textbox",
    )

    return msg_textbox


def login(username, password, user_leaderboard):
    default_output = (
        "guest",
        gr.Row(visible=False),
        gr.Row(visible=True),
        gr.Markdown(
            USER_WELCOME_MSG.format(
                user="", contributions=0, position="Ez dago sailkapenik", boletoak=0
            ),
            visible=False,
        ),
    )
    try:
        auth_manager = AuthManager("data/users")
        auth_manager.login(username=username, password=password)

        gr.Info(f"{username} erabiltzaileak saioa hasi du.")
        return (
            auth_manager.get_current_user(),
            gr.Row(visible=True),
            gr.Row(visible=False),
            update_user_info(username=username, user_leaderboard=user_leaderboard),
        )
    except Exception as e:
        gr.Info(f"{e}")
        return default_output


def update_user_info(username, user_leaderboard):
    user_info = user_leaderboard[user_leaderboard["Erabiltzaile izena"] == username]
    if len(user_info):
        contributions = user_info["Kontribuzioak"].values[0]
        position = user_info["Rank"].values[0]
    else:
        contributions = 0
        position = "Ez dago sailkapenik"

    return gr.Markdown(
        USER_WELCOME_MSG.format(
            user=username,
            contributions=contributions,
            position=position,
            boletoak=contributions // 10,
        ),
        visible=True,
    )


def register(username, password, password2, email, hizk_maila, hezk_maila):
    default_output = (
        "guest",
        gr.Row(visible=False),
        gr.Row(visible=True),
        gr.Markdown(
            USER_WELCOME_MSG.format(
                user="", contributions=0, position="Ez dago sailkapenik", boletoak=0
            ),
            visible=False,
        ),
    )
    if password != password2:
        gr.Info("Pasahitzak ez dira berdinak.")
        return default_output

    if email == "":
        gr.Info("Emaila ezin da hutsik egon.")
        return default_output

    if not re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email):
        gr.Info("Emaila ez da baliozkoa.")
        return default_output

    if username == "":
        gr.Info("Erabiltzaile izena ezin da hutsik egon.")
        return default_output

    if len(password) < 8:
        gr.Info("Pasahitza motzegia da. Gutxienez 8 karaktere izan behar ditu.")
        return default_output
    try:
        auth_manager = AuthManager("data/users")
        auth_manager.register(
            username=username,
            password=password,
            email=email,
            hizk_maila=hizk_maila,
            hezk_maila=hezk_maila,
        )

        gr.Info(f"{username} erabiltzailea erregistratu da.")
        return (
            auth_manager.get_current_user(),
            gr.Row(visible=True),
            gr.Row(visible=False),
            gr.Markdown(
                USER_WELCOME_MSG.format(
                    user=username,
                    contributions=0,
                    position="Ez dago sailkapenik",
                    boletoak=0,
                ),
                visible=True,
            ),
        )

    except Exception as e:
        gr.Info(f"{e}")
        return default_output


def logout():
    auth_manager = AuthManager("data/users")
    auth_manager.logout()

    gr.Info("Saioa itxi da.")
    return (
        "guest",
        gr.Row(visible=False),
        gr.Row(visible=True),
    )


def activate_deactivate_send_feedback(
    chatbot_a, chatbot_b, eduki_bal, euskara_bal, orokorra_bal
):
    if chatbot_a and not len(chatbot_a) and chatbot_b and not len(chatbot_b):
        return (
            gr.Button("📩 Bidali balorazioa", interactive=False),
            gr.Radio(
                label="Edukiaren kalitatea",
                choices=[
                    "👈 A da hobea",
                    "🤝 Berdinketa",
                    "👉 B da hobea",
                ],
                value=eduki_bal,
                interactive=True,
            ),
            gr.Radio(
                label="Euskararen kalitatea",
                choices=[
                    "👈 A da hobea",
                    "🤝 Berdinketa",
                    "👉 B da hobea",
                ],
                value=euskara_bal,
                interactive=True,
            ),
            gr.Radio(
                label="Orokorrean, zein da erantzun hobea?",
                choices=[
                    "👈 A da hobea",
                    "🤝 Berdinketa",
                    "👉 B da hobea",
                ],
                value=None,
                interactive=False,
                visible=False,
            ),
        )

    if eduki_bal and euskara_bal:
        # Aldatu berdinketak kontutan hartzeko
        if eduki_bal != euskara_bal and "🤝 Berdinketa" not in [eduki_bal, euskara_bal]:
            return (
                gr.Button(
                    "📩 Bidali balorazioa", interactive=True if orokorra_bal else False
                ),
                gr.Radio(
                    label="Edukiaren kalitatea",
                    choices=[
                        "👈 A da hobea",
                        "🤝 Berdinketa",
                        "👉 B da hobea",
                    ],
                    value=eduki_bal,
                    interactive=True,
                ),
                gr.Radio(
                    label="Euskararen kalitatea",
                    choices=[
                        "👈 A da hobea",
                        "🤝 Berdinketa",
                        "👉 B da hobea",
                    ],
                    value=euskara_bal,
                    interactive=True,
                ),
                gr.Radio(
                    label="Orokorrean, zein da erantzun hobea?",
                    choices=[
                        "👈 A da hobea",
                        "🤝 Berdinketa",
                        "👉 B da hobea",
                    ],
                    value=orokorra_bal,
                    interactive=True,
                    visible=True,
                ),
            )
        else:
            return (
                gr.Button("📩 Bidali balorazioa", interactive=True),
                gr.Radio(
                    label="Edukiaren kalitatea",
                    choices=[
                        "👈 A da hobea",
                        "🤝 Berdinketa",
                        "👉 B da hobea",
                    ],
                    value=eduki_bal,
                    interactive=True,
                ),
                gr.Radio(
                    label="Euskararen kalitatea",
                    choices=[
                        "👈 A da hobea",
                        "🤝 Berdinketa",
                        "👉 B da hobea",
                    ],
                    value=euskara_bal,
                    interactive=True,
                ),
                gr.Radio(
                    label="Orokorrean, zein da erantzun hobea?",
                    choices=[
                        "👈 A da hobea",
                        "🤝 Berdinketa",
                        "👉 B da hobea",
                    ],
                    value=None,
                    interactive=False,
                    visible=False,
                ),
            )
    else:
        return (
            gr.Button("📩 Bidali balorazioa", interactive=False),
            gr.Radio(
                label="Edukiaren kalitatea",
                choices=[
                    "👈 A da hobea",
                    "🤝 Berdinketa",
                    "👉 B da hobea",
                ],
                value=eduki_bal,
                interactive=True,
            ),
            gr.Radio(
                label="Euskararen kalitatea",
                choices=[
                    "👈 A da hobea",
                    "🤝 Berdinketa",
                    "👉 B da hobea",
                ],
                value=euskara_bal,
                interactive=True,
            ),
            gr.Radio(
                label="Orokorrean, zein da erantzun hobea?",
                choices=[
                    "👈 A da hobea",
                    "🤝 Berdinketa",
                    "👉 B da hobea",
                ],
                value=None,
                interactive=False,
                visible=False,
            ),
        )


def deactivate_send_feedback(evt: gr.EventData):
    return (
        gr.Button("📩 Bidali balorazioa", interactive=False),
        gr.Radio(
            label="Edukiaren kalitatea",
            choices=[
                "👈 A da hobea",
                "🤝 Berdinketa",
                "👉 B da hobea",
            ],
            value=None,
            interactive=False,
        ),
        gr.Radio(
            label="Euskararen kalitatea",
            choices=[
                "👈 A da hobea",
                "🤝 Berdinketa",
                "👉 B da hobea",
            ],
            value=None,
            interactive=False,
        ),
        gr.Radio(
            label="Orokorrean, zein da erantzun hobea?",
            choices=[
                "👈 A da hobea",
                "🤝 Berdinketa",
                "👉 B da hobea",
            ],
            value=None,
            interactive=False,
            visible=False,
        ),
    )


with gr.Blocks(
    theme=style.White(),
    # theme=theme,
    fill_height=True,
    # fill_width=True,
    analytics_enabled=False,
    title="Txatbot ebaluatoia",
    # head="""
    # <script type="text/javascript">
    #     if (!window.location.href.startsWith("https://")) {
    #         window.location.href = "https://ebaluatoia.hitz.eus"
    #     }
    # </script>
    # """,
    css="""
        .center-text { text-align: center; } 
        footer {visibility: hidden;} 
        .avatar-container {width: 50px; height: 50px; border: none;} 
        .modal-container {max-width: 800px;}
        .modal-block {padding: 20pt;}
        textarea {margin-right: 10px;}
        .icon-button-wrapper {visibility: hidden;}
        #login-row {width: 80%; margin: auto auto;}
        #mantenimendua-row {width: 80%; margin: auto auto;}
        #submit-txtbox textarea {border-radius: 20px; padding-left: 15px;}
        #submit-txtbox button {margin: 3px;}
        #footer-logo {margin: auto auto;}
        #balorazioak .form {border: none;}
        #balorazioak button {margin-top: 10px;}
        #radio-group label {width: 32.5%;}
        #radio-group fieldset {margin-top: -15px; margin-bottom: 10px;}
        #placeholder_textbox textarea {color: gray; font-style: italic; border-radius: 20px;}
        @media only screen and (max-width: 600px) {
            #radio-group label {width: 100%;}
        }
    """,
) as demo:
    user = gr.State("guest")

    erab_tab_mkd = gr.Markdown(
        USER_WELCOME_MSG.format(
            user="", contributions=0, position="Ez dago sailkapenik", boletoak=0
        ),
        visible=False,
    )
    mantenimendua_row = gr.Row(visible=False, elem_id="mantenimendua-row")
    pre_login_row = gr.Row(visible=True, equal_height=True, elem_id="login-row")
    content_row = gr.Row(visible=False)

    with mantenimendua_row:
        with gr.Column():
            gr.Markdown(
                """
            <center>
            <h1 style="font-size: 48px">Txatbot ebaluatoia</h1>
                        
            <img src="https://raw.githubusercontent.com/hitz-zentroa/latxa/refs/heads/main/assets/latxa_round.png" width="380px">
                        
            <h1 style="font-size: 48px">Animatu zaitez!</h1>
            <br>
            <br>
            <h1>🛠️ Mantenimenduan gaude, barkatu eragozpenak. 🛠️</h1>
            </center>
            """,
                max_height="100%",
            )

    with pre_login_row:
        # Login column
        with gr.Column(elem_id="latxa-logo"):
            gr.Markdown(
                """
            <center>
            <h1 style="font-size: 48px">Txatbot ebaluatoia</h1>
                        
            <img src="https://raw.githubusercontent.com/hitz-zentroa/latxa/refs/heads/main/assets/latxa_round.png" width="380px">
                        
            <h1 style="font-size: 48px">Animatu zaitez!</h1>
            </center>
            """,
                #max_height="100%",
            )
        # ADI! visible=True jarri ebaluatoia hasteko!
        with gr.Column(visible=True):
            gr.Markdown("## Saioa hasi")
            username_log = gr.Textbox(
                label="Erabiltzaile izena", placeholder="Izena", max_length=25
            )
            password_log = gr.Textbox(
                label="Pasahitza",
                placeholder="Pasahitza",
                type="password",
                max_length=25,
            )
            login_btn = gr.Button("Saioa hasi", interactive=True)

            # Register column
            gr.Markdown("## edo erregistratu")
            email_txt = gr.Textbox(
                label="Emaila", placeholder="Emaila", type="email", max_length=50
            )
            username_reg = gr.Textbox(
                label="Erabiltzaile izena", placeholder="Izena", max_length=25
            )
            password_reg = gr.Textbox(
                label="Pasahitza",
                placeholder="Pasahitza",
                type="password",
                max_length=25,
            )
            password_reg2 = gr.Textbox(
                label="Pasahitza errepikatu",
                placeholder="Pasahitza",
                type="password",
                max_length=25,
            )


            hizk_maila = gr.Dropdown(
                label="Hizkuntza maila",
                choices=[
                    "C1 (EGA) edo C2",
                    "B1 edo B2",
                    "A1 edo A2",
                ],
                info="""
Zalantza kasuan, hautatu:
 - "C1 edo C2" maila euskara zure ama-hizkuntza bada, edo euskaraz gaztelaniaz bezain ondo ala hobeto moldatzen bazara.
 - "B1 edo B2" maila euskaraz nahiko txukun moldatzen bazara, baina ez gaztelaniaz bezain ondo.
 - "A1 edo A2" maila zure euskara-maila oso oinarrizkoa bada.

                """,
            )
            hezk_maila = gr.Dropdown(
                label="Hezkuntza maila",
                choices=["Unibertsitate ikasketak", "Lanbide-heziketa", "Batxilergoa", "DBH"],
                info="""
Aukeratu zure titulu altuena; une honetan ikaslea bazara aukeratu egun egiten ari zaren ikasketak.
                """,
            )
            
            register_btn = gr.Button("Erregistratu", interactive=True)

    with content_row:
        info_modal = Modal(visible=False)
        arena_tab = gr.Tab("Bozkatu", visible=True)
        erabiltzailea_tab = gr.Tab("Sailkapena", visible=True)
        informazioa_tab = gr.Tab("Informazioa", visible=True)

        # emaitzak_tab = gr.Tab("Emaitzak", visible=True)
        with erabiltzailea_tab:
            # post_login_row = gr.Row(visible=False)
            refresh_btn = gr.Button("Eguneratu", interactive=True, size="sm")
            competition_progress = gr.Row(visible=True)
            user_leaderboard_row = gr.Row(visible=True)

            # with post_login_row:
            #     logout_btn = gr.Button("Saioa itxi", interactive=True)

            with user_leaderboard_row:
                with gr.Column():
                    gr.Markdown("## Erabiltzaileen sailkapena")
                    user_leaderboard = gr.DataFrame(
                        column_widths=["10$", "40%", "50%"],
                        headers=["Rank", "Erabiltzaile izena", "Kontribuzioak"],
                        interactive=False,
                    )

            with competition_progress:
                with gr.Column():
                    gr.Markdown("## Bidalketak denboran zehar")
                    progress_barplot = gr.BarPlot(
                        value=pd.DataFrame(), x="timestamp", y="submissions", x_bin="d"
                    )

            gr.Markdown("""
            ## Sariak eta zozketa
            Bozketa gehien bidaltzen dituzten 10 lagunei opari bonoak emango zaizkie: lehenengoari **150 euroko 
            teknologia-bonoa**, bigarrenari **100 euroko teknologia-bonoa** eta hurrengoei **elkar dendako 50 euroko 
            bono** bana emango zaizkie. Ebaluatutako bozketa kopuruan berdinketa egonez gero, kopuru horretara 
            lehenago iritsi denari emango zaio lehentasuna.

            Ebaluatoian ematen diren zozketa zenbaki guztien artean bat izango da saritua. Irabazleak **teknologian gastatzeko 350 euroko bono bat** jasoko du.

            Zozketaren eguna aurrerago zehaztuko da.

            Zorte on!
            """)
            gr.HTML("<hr>")

        with arena_tab:
            models = gr.State(update_models(only_models=True))
            # user_info = gr.State({})
            conv_id = gr.State("")
            with gr.Row():
                with gr.Column():
                    chatbot_a = gr.Chatbot(
                        label="A eredua",
                        height="50vh",
                        show_copy_all_button=False,
                        type="messages",
                        # avatar_images=[
                        #     None,  # "https://static.vecteezy.com/system/resources/previews/019/879/186/non_2x/user-icon-on-transparent-background-free-png.png",
                        #     "https://raw.githubusercontent.com/hitz-zentroa/latxa/refs/heads/main/assets/latxa_round.png",
                        # ],
                    )

                with gr.Column():
                    chatbot_b = gr.Chatbot(
                        label="B eredua",
                        show_copy_all_button=False,
                        height="50vh",  # 550
                        type="messages",
                        # avatar_images=[
                        #     None,  # "https://static.vecteezy.com/system/resources/previews/019/879/186/non_2x/user-icon-on-transparent-background-free-png.png",
                        #     "https://raw.githubusercontent.com/hitz-zentroa/latxa/refs/heads/main/assets/latxa_round.png",
                        # ],
                    )

            with gr.Row():
                msg = gr.Textbox(
                    label="Sartu zure mezua hemen",
                    autofocus=True,
                    submit_btn=True,
                    stop_btn=True,
                    elem_id="submit-txtbox",
                )

            # submit = gr.Button("📩 Bidali")
            with gr.Row(elem_id="balorazioak", equal_height=False, variant="default"):
                berria_col = gr.Column(scale=1)
                balorazioa_col = gr.Column(scale=2, elem_id="radio-group")
                bidali_bal_col = gr.Column(scale=1)
                with balorazioa_col:
                    eduki_balorazioa = gr.Radio(
                        label="Edukiaren kalitatea",
                        info="Edukiaren kalitatea",
                        choices=[
                            "👈 A da hobea",
                            "🤝 Berdinketa",
                            "👉 B da hobea",
                        ],
                        value=None,
                        interactive=False,
                        container=False,
                    )
                    # with gr.Column(min_width=300):
                    hizk_balorazioa = gr.Radio(
                        label="Euskararen kalitatea",
                        info="Euskararen kalitatea",
                        choices=[
                            "👈 A da hobea",
                            "🤝 Berdinketa",
                            "👉 B da hobea",
                        ],
                        value=None,
                        interactive=False,
                        container=False,
                    )

                    # with gr.Row():
                    balorazio_orokorra = gr.Radio(
                        label="Orokorrean, zein da erantzun hobea?",
                        info="Orokorrean, zein da erantzun hobea?",
                        choices=[
                            "👈 A da hobea",
                            "🤝 Berdinketa",
                            "👉 B da hobea",
                        ],
                        value=None,
                        interactive=False,
                        visible=False,
                        container=False,
                    )

                with bidali_bal_col:
                    bidali_bal = gr.Button("📩 Bidali balorazioa", interactive=False)

                    # repeat = gr.Button("🔁 Errepikatu", interactive=False)

                with berria_col:
                    clear = (
                        gr.ClearButton(
                            [msg, chatbot_a, chatbot_b], value="🔁 Txat berria"
                        )
                        .click(
                            update_models,
                            # outputs=[models, conv_id, repeat, bidali_bal, msg],
                            outputs=[models, conv_id, bidali_bal, msg],
                        )
                        .then(
                            deactivate_send_feedback,
                            inputs=[],
                            outputs=[
                                bidali_bal,
                                eduki_balorazioa,
                                hizk_balorazioa,
                                balorazio_orokorra,
                            ],
                        )
                        .then(
                            lambda: gr.Textbox(
                                label="Sartu zure mezua hemen",
                                autofocus=True,
                                submit_btn=True,
                                stop_btn=True,
                                elem_id="submit-txtbox",
                            ),
                            inputs=[],
                            outputs=[msg],
                        )
                    )

            gr.HTML("<hr>")

            submit_conv_1_evt = msg.submit(
                partial(predict_single_stream, repeat_last=False, model_id=0),
                inputs=[msg, models, chatbot_a, conv_id],
                outputs=[msg, chatbot_a, bidali_bal],
            )
            submit_conv_1_evt.then(
                activate_deactivate_send_feedback,
                inputs=[
                    chatbot_a,
                    chatbot_b,
                    eduki_balorazioa,
                    hizk_balorazioa,
                    balorazio_orokorra,
                ],
                outputs=[
                    bidali_bal,
                    eduki_balorazioa,
                    hizk_balorazioa,
                    balorazio_orokorra,
                ],
            )
            # submit_conv_1_evt.then(
            #     lambda history: gr.Button("🔁 Errepikatu", interactive=True if history else False),
            #     inputs=[chatbot_a],
            #     outputs=[repeat]
            # )

            submit_conv_2_evt = msg.submit(
                partial(predict_single_stream, repeat_last=False, model_id=1),
                inputs=[msg, models, chatbot_b, conv_id],
                outputs=[msg, chatbot_b, bidali_bal],
            )
            submit_conv_2_evt.then(
                activate_deactivate_send_feedback,
                inputs=[
                    chatbot_a,
                    chatbot_b,
                    eduki_balorazioa,
                    hizk_balorazioa,
                    balorazio_orokorra,
                ],
                outputs=[
                    bidali_bal,
                    eduki_balorazioa,
                    hizk_balorazioa,
                    balorazio_orokorra,
                ],
            )
            msg.stop(
                fn=None,
                inputs=[],
                outputs=[],
                cancels=[
                    submit_conv_1_evt,
                    submit_conv_2_evt,
                    # repeat_conv_1_evt,
                    # repeat_conv_2_evt,
                ],
            )
            gr.on(
                triggers=[
                    eduki_balorazioa.input,
                    hizk_balorazioa.input,
                    balorazio_orokorra.input,
                ],
                fn=activate_deactivate_send_feedback,
                inputs=[
                    chatbot_a,
                    chatbot_b,
                    eduki_balorazioa,
                    hizk_balorazioa,
                    balorazio_orokorra,
                ],
                outputs=[
                    bidali_bal,
                    eduki_balorazioa,
                    hizk_balorazioa,
                    balorazio_orokorra,
                ],
            )

            bidali_bal.click(
                save_history,
                inputs=[
                    models,
                    chatbot_a,
                    chatbot_b,
                    user,
                    conv_id,
                    eduki_balorazioa,
                    hizk_balorazioa,
                    balorazio_orokorra,
                ],
                # outputs=[msg, repeat],
                outputs=[msg],
            ).then(
                deactivate_send_feedback,
                inputs=[],
                outputs=[
                    bidali_bal,
                    eduki_balorazioa,
                    hizk_balorazioa,
                    balorazio_orokorra,
                ],
            ).then(
                render_user_leaderboard,
                inputs=[user],
                outputs=[user_leaderboard, progress_barplot],
            ).then(
                update_user_info,
                inputs=[user, user_leaderboard],
                outputs=[erab_tab_mkd],
            )

        with informazioa_tab:
            gr.Markdown("""
            ## Informazioa eta Argibideak
            Ongi etorri txatbot sistemen **Ebaluatoira**.

            Euskaraz dakizu? 

            Lagun iezaguzu euskararako txatboten kalitatea neurtzen! Txatbot eredu ezberdinak lehian jarri ditugu eta zure iritziz zein den hobea 
            jakiteko, zure parte hartzea behar dugu. 

            Martxoaren 19tik apirilaren 2ra, egon adi! 

            ### Argibideak
            Euskararako txatbot publikoak garatzen lagun diezaguzun, HiTZ zentroan prestatu dugun ikerketa ekimen bat da Ebaluatoia. Parte hartzen 
            duzuen guztiok egundoko zozketa baterako zenbakiak lortzeko aukera izango duzue. 

            Hauxe egin beharko duzu:
            
            Galdera edo agindu bat **idatzi** eta **bidali** behar duzu:

            <div style="display: flex; justify-content: center; align-items: center; text-align: center; padding: 20px;">
                <img src="https://raw.githubusercontent.com/hitz-zentroa/arena/refs/heads/main/assets/input_textbox.png" style=""/>
            </div>
                        
            Bi txatbot eredu ezberdinek erantzuten dute. Zure lana erantzunak aztertzea eta konparatzea da, zein den hobea erabakitzeko. 
            **Edukiaren** kalitatea eta **euskararen** kalitatea neurtu nahi dugu. 

            <div style="display: flex; justify-content: center; align-items: center; text-align: center; padding: 20px;">
                <img src="https://raw.githubusercontent.com/hitz-zentroa/arena/refs/heads/main/assets/feedback_1.png" style=""/>
            </div>
                        
            Zenbait kasutan, hirugarren galdera bat egingo zaizu, zure iritzia bidali ahal izateko:
            
            <div style="display: flex; justify-content: center; align-items: center; text-align: center; padding: 20px;">
                <img src="https://raw.githubusercontent.com/hitz-zentroa/arena/refs/heads/main/assets/feedback_2.png" style=""/>
            </div>
                        
            Galdera guztiak erantzutean, zure iritzia bidaltzeko aukera izango duzu, “Bidali balorazioa” botoiaren bitartez.
                        
            <div style="display: flex; justify-content: center; align-items: center; text-align: center; padding: 20px;">
                <img src="https://raw.githubusercontent.com/hitz-zentroa/arena/refs/heads/main/assets/feedback_3.png" style=""/>
            </div>

            Galdera edo agindu berri bat idatzi ahal izateko, “Txat berria” botoia zapaldu beharko duzu.

            Laburtuz, egin beharrekoa:
                <ol>
                <li>Idatzi galdera edo agindu bat txatbotentzako. Adibidez:</li>
                    <ul>
                    <li>Patata tortila nola egiten da?</li>
                    <li>Laburtu ondoko testua:</li>
                    </ul>
                <li>Irakurri bi erantzunak eta alderatu edukiaren kalitatea eta euskararen kalitatea.</li>
                <li>Erabaki zein erantzun nahiago duzun edukiaren aldetik eta euskararen aldetik. Neurri bakoitzerako:</li>
                    <ul>
                    <li>A hobea bada, aukeratu A.</li>
                    <li>B hobea bada, aukeratu B.</li>
                    <li>Biak maila berean badaude (on, zein txar), aukeratu BERDINKETA.</li>
                    </ul>
                <li>Nahi izanez gero, elkarrizketarekin jarraitu, azalpen gehiago eskatu edo beste galdera bat probatu. 3.en pausuko 
                erantzuna aldatu ditzakezu, elkarrizketa osoaren kalitatea kontuan hartuz.</li>
                <li>Prozesua berriz hasteko, “Txat berria” botoia zapaldu.</li>
                </ol>
                        
            Zure IRITZIA nahi dugu. Baina **zintzo jokatu!** Izan ere, tarteka jasotako emaitzen analisi bat burutuko dugu eta kontrol 
            erantzunak egiaztatuko ditugu. Ez badaude zuzen, ez duzu zozketan parte hartuko. 

            Bidaltzen dituzun 10 balorazioko, zozketan parte hartzeko zenbaki bat esleituko zaizu. Gainera, ebaluazio gehien egiten dituzten 
            10 lagunei opari bono bana emango diegu. Nahi adina bozketa bidali ditzakezu eta nahi adina aldiz buelta zaitezke Ebaluatoira.  
            Horretarako, erabil itzazu izena eta pasahitza eta segi agindu eta balorazio kopurua gehitzen eta zozketa zenbakiak jasotzen!

            ### Txatbot-en inguruan
            Guztira 20 txatbot jarri ditugu lehian. Horien artean, GPT-4o edo Claude bezelako eredu pribatuak, Llama 3.1 bezalako eredu irekiak 
            eta guk garatutako batzuk ere daude. Oro har, **denetarik daude**, txatbot onak, oso onak eta txarrak ere. Azterketa honetan, txatbot 
            hauek sistematikoki ebaluatzea dugu helburu.

            **KONTAKTUA:** ebaluatoia.hitz@ehu.eus
            """)

            gr.HTML("<hr>")

    with gr.Row(elem_id="footer-logo"):
        with gr.Column():
            gr.Markdown(
                """
            <div style="display: flex; justify-content: center; align-items: center; text-align: center; position: sticky; bottom: 0; padding: 20px;">
                <img src="https://www.hitz.eus/sites/default/files/HitzLOgoa_3.png" width="350" style="margin-right: 20px"/>
                
            </div>
            """
            )  
        with gr.Column():
            gr.Markdown(
                """
            <div style="display: flex; justify-content: center; align-items: center; text-align: center; position: sticky; bottom: 0;">
                <img src="https://raw.githubusercontent.com/hitz-zentroa/arena/refs/heads/main/assets/iker-gaitu_logoa_150_0.png" width="100"  style=""/>
                
            </div>
            """
            )  
        with gr.Column():
            gr.Markdown(
                """
            <div style="display: flex; justify-content: center; align-items: center; text-align: center; position: sticky; bottom: 0; padding: 20px;">
                <img src="https://raw.githubusercontent.com/hitz-zentroa/arena/refs/heads/main/assets/LOGO_Ilenia_sinslogan.png" width="250"  style=""/>
                
            </div>
            """
            ) 

    with Modal(visible=True) as modal:
        gr.Markdown(
            """
        # Datuen erabilerari buruzko informazioa
        Ebaluatoi honetan parte hartzeko, erabiltzaile eta posta elektroniko bat adierazi beharko duzu. 
        Informazio hau, bukaerako zozketan parte hartzeko beharrezkoa da. Ebaluatoia amaitzean, jasotako 
        informazioa ezabatuko da. 
        
        **ADI! Ebaluatoian zure erabiltzaile izena publikoki ikusgai egongo da.**
        
        **ADI! Ebaluatoian datu pertsonalak eta prompt/erantzunak jasotzen direnez, 14+ urteko pertsonak bakarrik
        parte hartzeko aukera izango dute.**

        Ez da bestelako datu pertsonalik jasoko. Hori horrela, beste datu batzuk jasotzen dira, hala nola:
        * Bidalitako mezuak eta erantzunak.
        * Bidalitako preferentziak.

        Datu hauek hurrengo helburuetarako erabiliko dira:
        * Ebaluatoian parte hartzen duten txatbot-en ebaluaziorako.
        * Txatbot berrien ikerkuntzarako.

        Datu hauek etorkizunean era irekian (CC0 lizentzia) argitaratuko dira. Ebaluatoian parte hartzearekin 
        horretarako baimena ematen duzu.

        Eskerrik asko parte hartzeagatik!

        **KONTAKTUA:** ebaluatoia.hitz@ehu.eus
        """
        )

    with info_modal:
        gr.Markdown(
            """
        # Hasi aurretik

        Ebaluatoi honetan metodo zientifikoa erabiltzen dugu, hori horrela, **kalitate desberdineko ereduak daude**, 
        onak, oso onak eta **txarrak** ere. Beraz, kontutan izan eredu batzuk ingelesez erantzun dezaketela edo bere 
        burua errepikatzen hasi daitezkeela. Ebaluatoia aurrera joan ahala, eredu txarrak geroz eta gutxiago agertuko 
        dira, baina zuen esku dago hauek identifikatzea!

        Mila esker kolaboratzeagatik!

        """
        )

    gr.on(
        fn=login,
        triggers=[login_btn.click, password_log.submit],
        inputs=[username_log, password_log, user_leaderboard],
        outputs=[user, content_row, pre_login_row, erab_tab_mkd],
    ).then(fn=lambda: Modal(visible=True), outputs=[info_modal]).then(
        render_user_leaderboard,
        inputs=[user],
        outputs=[user_leaderboard, progress_barplot],
    )
    register_btn.click(
        register,
        inputs=[
            username_reg,
            password_reg,
            password_reg2,
            email_txt,
            hizk_maila,
            hezk_maila,
        ],
        outputs=[user, content_row, pre_login_row, erab_tab_mkd],
    ).then(fn=lambda: Modal(visible=True), outputs=[info_modal]).then(
        render_user_leaderboard,
        inputs=[user],
        outputs=[user_leaderboard, progress_barplot],
    )

    refresh_btn.click(
        render_user_leaderboard,
        inputs=[user],
        outputs=[user_leaderboard, progress_barplot],
    ).then(update_user_info, inputs=[user, user_leaderboard], outputs=[erab_tab_mkd])


    demo.load(
        update_models,
        inputs=[],
        # outputs=[models, conv_id, repeat, bidali_bal, msg],
        outputs=[models, conv_id, bidali_bal, msg],
    )

    demo.load(
        render_user_leaderboard,
        inputs=[user],
        outputs=[user_leaderboard, progress_barplot],
    )

demo.queue(default_concurrency_limit=40)
demo.launch(server_name="0.0.0.0", server_port=7887, share=False, root_path="https://ebaluatoia.hitz.eus")

