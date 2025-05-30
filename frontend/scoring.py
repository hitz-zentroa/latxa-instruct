"""
scoring.py
==========

This module provides functions for evaluating and scoring AI model responses in the Latxa-Instruct project.
It includes utilities for loading user feedback data, computing ELO ratings, generating leaderboards,
calculating confidence intervals, and summarizing user contributions. The module supports both local and
Hugging Face Hub data sources and provides statistical tools for comparing model performance.

Partially based on LMSys arena notebook: https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=mukqgshMarFi

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
from functools import lru_cache
import glob
import json
import math
from typing import Tuple, Any
# import plotly.express as px
from scipy.stats import ttest_ind
from collections import Counter
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd

from huggingface_hub import HfFileSystem


def get_hf_data(repo_id: str, model_names: Tuple[str]) -> list[Any]:
    fs = HfFileSystem()
    if not fs.exists(f"datasets/{repo_id}"):
        return pd.DataFrame()

    preferences = []
    for file_name in fs.glob(f"datasets/{repo_id}/data/*"):
        with fs.open(file_name, "r") as file:
            data = json.load(file)
            if data["model_a"] in model_names and data["model_b"] in model_names:
                preferences.append(data)

    return preferences
    # return pd.DataFrame(preferences).sort_values(ascending=True, by=["timestamp"])


def get_data(repo_id: str, model_names: Tuple[str]) -> pd.DataFrame:
    preferences = []
    for file_name in glob.glob(f"data/{repo_id}/history_*"):
        with open(file_name, "r") as file:
            try:
                data = json.load(file)
                if data["model_a"] in model_names and data["model_b"] in model_names:
                    preferences.append(data)
            except json.JSONDecodeError:
                continue

    return preferences

def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000, sample_weight=None):
    model_names = np.unique([*df.model_a, *df.model_b])
    base = pd.DataFrame(0, index=model_names, columns=model_names)

    ptbl_a_win = (
        base
        + pd.pivot_table(
            df[df["winner"] == "model_a"],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
    ).fillna(0)

    ptbl_b_win = (
        base
        + pd.pivot_table(
            df[df["winner"] == "model_b"],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
    ).fillna(0)

    # if no tie, create a zero matrix
    if sum(df["winner"].isin(["tie", "tie (bothbad)"])) == 0:
        ptbl_tie = pd.DataFrame(0, index=model_names, columns=model_names)
    else:
        ptbl_tie = (
            base
            + pd.pivot_table(
                df[df["winner"].isin(["tie", "tie (bothbad)"])],
                index="model_a",
                columns="model_b",
                aggfunc="size",
                fill_value=0,
            )
        ).fillna(0)
        ptbl_tie = ptbl_tie + ptbl_tie.T
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            # if nan skip
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = math.log(BASE)
            X[cur_row + 1, models[m_b]] = -math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    # if "mixtral-8x7b-instruct-v0.1" in models.index:
    #     elo_scores += 1114 - elo_scores[models["mixtral-8x7b-instruct-v0.1"]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for i in range(num_round):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def preety_print_model_ratings(ratings):
    df = (
        pd.DataFrame(
            [[n, ratings[n]] for n in ratings.keys()], columns=["Model", "Elo rating"]
        )
        .sort_values("Elo rating", ascending=False)
        .reset_index(drop=True)
    )
    # df["Elo rating"] = (df["Elo rating"] + 0.5).astype(int)
    df.index = df.index + 1
    return df


def compute_pairwise_win_fraction(battles, max_num_models=30):
    # Times each model wins as Model A
    a_win_ptbl = pd.pivot_table(
        battles[battles["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    # Table counting times each model wins as Model B
    b_win_ptbl = pd.pivot_table(
        battles[battles["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    # Table counting number of A-B pairs
    num_battles_ptbl = pd.pivot_table(
        battles, index="model_a", columns="model_b", aggfunc="size", fill_value=0
    )

    # Computing the proportion of wins for each model as A and as B
    # against all other models
    row_beats_col_freq = (a_win_ptbl + b_win_ptbl.T) / (
        num_battles_ptbl + num_battles_ptbl.T
    )

    # Arrange ordering according to proprition of wins
    prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
    prop_wins = prop_wins[:max_num_models]
    model_names = list(prop_wins.keys())
    row_beats_col = row_beats_col_freq.loc[model_names, model_names]
    return row_beats_col

def get_90_ci(df):
    """Compute confidence interval for each model in the dataframe."""
    return df.apply(
        lambda x: "/".join(
            (
                "+" + str(int(x.quantile(0.95) - x.quantile(0.50))),
                "-" + str(int(x.quantile(0.50) - x.quantile(0.05))),
            )
        ),
        axis=0,
    )


def get_matches_df(data):
    matches = []
    for _, row in data.iterrows():
        if row["winner"] in ["model_a", "model_b"]:
            matches.append(
                {
                    "model_a": row["model_a"],
                    "model_b": row["model_b"],
                    "result": "win" if row["winner"] == "model_a" else "loss",
                }
            )
            matches.append(
                {
                    "model_a": row["model_b"],
                    "model_b": row["model_a"],
                    "result": "win" if row["winner"] == "model_b" else "loss",
                }
            )
        else:
            matches.append(
                {"model_a": row["model_a"], "model_b": row["model_b"], "result": "tie"}
            )
            matches.append(
                {"model_a": row["model_b"], "model_b": row["model_a"], "result": "tie"}
            )

    matches_df = pd.DataFrame(matches)
    matches_df = (
        matches_df.groupby(["model_a", "result"])
        .count()
        .reset_index()
        .sort_values(by="model_a")
    )

    return matches_df


def get_leaderboard_df(data, BOOTSTRAP_ROUNDS=100):
    np.random.seed(42)
    bootstrap_elo_lu = get_bootstrap_result(data, compute_mle_elo, BOOTSTRAP_ROUNDS)
    ci_intervals = get_90_ci(bootstrap_elo_lu).reset_index()
    ci_intervals.columns = ["Model", "90% CI"]
    # for name1, name2 in zip(bootstrap_elo_lu.columns, bootstrap_elo_lu.columns[1:]):
    #     _, p_val = ttest_ind(bootstrap_elo_lu[name1], bootstrap_elo_lu[name2], alternative="greater")
    #     print(f"p-value: {p_val:.2E}\t| [{name1}] VS [{name2}]")
    means = bootstrap_elo_lu.T.mean(axis=1, skipna=True)
    means = means.reset_index()
    means.columns = ["Model", "ELO"]
    means["ELO"] = means["ELO"].astype(int)
    means["90% CI"] = ci_intervals["90% CI"]

    means.index = means.index + 1
    elo_results = means.sort_values(by="ELO", ascending=False)

    return elo_results


def get_user_contributions_df(data):
    user_contributions = Counter(
        [
            row["username"]
            for row in data
            if "username" in row and row["username"] != "guest"
        ]
    )
    user_contributions = pd.DataFrame(
        user_contributions.items(), columns=["User", "Contributions"]
    )
    if len(user_contributions):
        user_contributions = user_contributions.sort_values(
            by="Contributions", ascending=False
        )
        user_contributions = user_contributions.reset_index(drop=True).reset_index()
        user_contributions.columns = ["Rank", "Erabiltzaile izena", "Kontribuzioak"]
        user_contributions["Rank"] = user_contributions["Rank"] + 1
    else:
        user_contributions.columns = ["Rank", "Erabiltzaile izena", "Kontribuzioak"]

    return user_contributions


@lru_cache(maxsize=5)
def get_results_dfs(
    repo_id,
    BOOTSTRAP_ROUNDS=100,
    model_names: Tuple[str] = (),
    ttl_hash=None,
    from_local: bool = False,
):
    data = (
        get_hf_data(repo_id, model_names)
        if not from_local
        else get_data(repo_id, model_names)
    )
    if len(data):
        user_contributions_df = get_user_contributions_df(data)

        data = pd.DataFrame(data).sort_values(ascending=True, by=["timestamp"])

        # Get the submissions count
        # data["timestamp"] = data["timestamp"].map(lambda x: x.split("T")[0])
        submissions_df = data.groupby("timestamp").count()["winner"].reset_index()
        submissions_df.columns = ["timestamp", "submissions"]

        # Get the matches dataframe
        matches_df = get_matches_df(data)

        # Compute the ELO ratings
        elo_results = get_leaderboard_df(data, BOOTSTRAP_ROUNDS)
    else:
        user_contributions_df = pd.DataFrame(
            columns=["Rank", "Erabiltzaile izena", "Kontribuzioak"]
        )
        submissions_df = pd.DataFrame(columns=["timestamp", "submissions"])
        elo_results = pd.DataFrame(columns=["Model", "ELO", "90% CI"])
        matches_df = pd.DataFrame(columns=["model_a", "result", "model_b"])

    return user_contributions_df, submissions_df, elo_results, matches_df


@lru_cache(maxsize=5) # Kendu arenan!
def get_user_dfs(
    repo_id,
    model_names: Tuple[str] = (),
    ttl_hash=None,
    from_local: bool = False
):
    data = (
        get_hf_data(repo_id, model_names)
        if not from_local
        else get_data(repo_id, model_names)
    )
    if len(data):
        user_contributions_df = get_user_contributions_df(data)

        data = pd.DataFrame(data).sort_values(ascending=True, by=["timestamp"])

    else:
        user_contributions_df = pd.DataFrame(
            columns=["Rank", "Erabiltzaile izena", "Kontribuzioak"]
        )
        data = pd.DataFrame()

    return user_contributions_df, data

