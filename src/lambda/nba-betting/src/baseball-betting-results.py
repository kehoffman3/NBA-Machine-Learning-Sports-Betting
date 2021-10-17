import mlbgame
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import boto3
from io import StringIO
import logging
import pytz

logger = logging.getLogger()
logger.setLevel(logging.INFO)

team_lookup = {
        "Mets": "NYM",
        "Padres": "SDP",
        "Nationals": "WSN",
        "Braves": "ATL",
        "Yankees": "NYY",
        "Rays": "TBR",
        "Orioles": "BAL",
        "Indians": "CLE",
        "Phillies": "PHI",
        "Pirates": "PIT",
        "Rangers": "TEX",
        "Rockies": "COL",
        "Marlins": "MIA",
        "Red Sox": "BOS",
        "Cubs": "CHC",
        "Blue Jays": "TOR",
        "Astros": "HOU",
        "Angels": "LAA",
        "Dodgers": "LAD",
        "White Sox": "CHW",
        "Giants": "SFG",
        "Royals": "KCR",
        "Twins": "MIN",
        "Tigers": "DET",
        "Cardinals": "STL",
        "Brewers": "MIL",
        "D-backs": "ARI",
        "Reds": "CIN",
        "Athletics": "OAK",
        "Mariners": "SEA"

    }

def lambda_handler(event, context):
    tz = pytz.timezone('US/Eastern')
    today = datetime.now(tz)
    yesterday = today - timedelta(days=1)
    yesterday_string = yesterday.strftime("%Y-%m-%d")
    yesterdays_games = mlbgame.day(yesterday.year, yesterday.month, yesterday.day)
    games_df = get_team_results(yesterdays_games)

    preds_df = get_preds(yesterday)
    
    # Only grade all bets
    # results_ev = grade_bets_ev(preds_df, winning_teams, losing_teams, yesterday_string)
    # results_ev_file_name = f"game-betting-results/MLB_RESULTS_{yesterday.strftime('%Y_%m_%d')}.csv"
    # save_to_s3(results_ev, results_ev_file_name)

    results_full = grade_bets_full(preds_df, games_df, yesterday_string)
    results_full_file_name = f"game-betting-results/MLB_RESULTS_FULL_{yesterday.strftime('%Y_%m_%d')}.csv"
    save_to_s3(results_full, results_full_file_name)

    return {
        'statusCode': 200,
        'body': "Success!"
    }


def save_to_s3(data, file_name):
    bucket = 'kehoffmn3-baseball'
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, file_name).put(Body=csv_buffer.getvalue())

def get_preds(yesterday):
    bucket = 'kehoffmn3-baseball'

    s3_client = boto3.client('s3')
    preds_file_name = f"todays-games-preds/MLB_PREDS_{yesterday.strftime('%Y_%m_%d')}.csv"
    response = s3_client.get_object(Bucket=bucket, Key=preds_file_name)
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        preds_df = pd.read_csv(response.get("Body"))
    else:
        raise Exception("Failed to retrieve yesterday's predictions")
    return preds_df

def get_team_results(games):
    winning_teams = []
    losing_teams = []
    results = {
        "date": [],
        "time": [],
        "game_id": [],
        "away": [],
        "home": [],
        "winner": []
    }
    for game in games:
        try:
            away_team = team_lookup[game.away_team]
            home_team = team_lookup[game.home_team]
            winning_team = team_lookup[game.w_team]
            date = game.date
            time = game.game_start_time
            game_id = game.game_id
            results["date"].append(date)
            results["time"].append(time)
            results["game_id"].append(game_id)
            results["away"].append(away_team)
            results["home"].append(home_team)
            results["winner"].append(winning_team)
        except AttributeError as e:
            print(e)
            continue
    return pd.DataFrame(results)


def grade_bets_full(preds_df, games_df, yesterday_string):
    results = {
    "date":[],
    "time":[],
    "pick":[],
    "odds":[],
    "wager":[],
    "result":[],
    "profit":[]

    }

    for index,row in preds_df.iterrows():
        if not row.get("home_prob") or not row.get("away_prob") or not row.get("home_odds") or not row.get("away_odds"):
            continue
        # Bet on all games and team with highest prob
        is_bet_home = row["home_prob"] >= row["away_prob"]
        odds = row["home_odds"] if is_bet_home else row["away_odds"]
        # Ignore betting on big favorites 
        if odds < -200:
            continue

        betting_team = row["home_team"] if is_bet_home else row["away_team"]
        
        game_id = row["game_id"]
        # Try to find the game
        target = games_df[games_df["game_id"] == game_id]

        if target.empty:
            continue

        winner = target["winner"].iloc[0]
        
        results["date"].append(yesterday_string)
        results["time"].append(row["start_time"])
        results["pick"].append(betting_team)

        results["odds"].append(odds)
        # Always betting 1 unit
        wager = 1
        results["wager"].append(wager)
        

        if not winner:
            result = "P"
            profit = 0
        elif betting_team == winner:
            result = "W"
            profit = odds/100 * wager if odds > 0 else wager * -100/odds
        else:
            result = "L"
            profit = -wager

        results["result"].append(result)
        results["profit"].append(profit)
    results_df = pd.DataFrame(results)
    return results_df


def grade_bets_ev(preds_df, winning_teams, losing_teams, yesterday_string):
    
    results = {
        "date":[],
        "pick":[],
        "odds":[],
        "wager":[],
        "result":[],
        "profit":[]

    }
    for index,row in preds_df.iterrows():
        # Bet on all games with ev greater than 0.05
        if row["best_ev"] > 0.05:
            betting_team = row["betting_team"]
            results["date"].append(yesterday_string)
            results["pick"].append(betting_team)
            odds = row["home_odds"] if row["home_ev"] > row["away_ev"] else row["away_odds"]
            results["odds"].append(odds)
            # Always betting 1 unit
            wager = 1
            results["wager"].append(wager)
            result = "P"
            profit = 0
            if betting_team in winning_teams:
                result = "W"
                profit = odds/100 * wager if odds > 0 else wager * -100/odds

            elif betting_team in losing_teams:
                result = "L"
                profit = -wager
            results["result"].append(result)
            results["profit"].append(profit)
    results_df = pd.DataFrame(results)
    return results_df

