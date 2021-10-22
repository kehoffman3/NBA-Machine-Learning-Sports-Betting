from datetime import datetime
from datetime import timedelta
import pandas as pd
import boto3
from io import StringIO
import logging
import pytz
from data import get_json_data, to_data_frame, get_todays_games_json, create_todays_games

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BUCKET = 'kehoffmn3-nba'
todays_games_url = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2021/scores/00_todays_scores.json'

def lambda_handler(event, context):
    tz = pytz.timezone('US/Eastern')
    today = datetime.now(tz)
    data = get_todays_games_json(todays_games_url)
    results = get_results_from_games(data)

    games_df = get_team_results(yesterdays_games)

    preds_df = get_preds(yesterday)

    results_full = grade_bets_full(preds_df, games_df, yesterday_string)
    results_full_file_name = f"game-betting-results/NBA_RESULTS_FULL_{yesterday.strftime('%Y_%m_%d')}.csv"
    save_to_s3(results_full, results_full_file_name)

    return {
        'statusCode': 200,
        'body': "Success!"
    }

    

def save_to_s3(data, file_name):
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(BUCKET, file_name).put(Body=csv_buffer.getvalue())

def get_preds(yesterday):

    s3_client = boto3.client('s3')
    preds_file_name = f"todays-games-preds/NBA_PREDS_{yesterday.strftime('%Y_%m_%d')}.csv"
    response = s3_client.get_object(Bucket=BUCKET, Key=preds_file_name)
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        preds_df = pd.read_csv(response.get("Body"))
    else:
        raise Exception("Failed to retrieve yesterday's predictions")
    return preds_df

def get_results_from_games(games):
    results = {
        #"date": [],
        #"time": [],
        "game_id": [],
        "away": [],
        "home": [],
        "winner": []
    }
    for game in games:
        try:
            if game.get("stt") != "Final":
                continue
            home = game.get('h')
            home_score = home.get('s')
            away = game.get('v')
            away_score = away.get('s')
            gid = game.get('gid')
            winner = home if home_score > away_score else away

            home_team = home.get('tc') + ' ' + home.get('tn')
            away_team = away.get('tc') + ' ' + away.get('tn')
            winning_team = winner.get('tc') + ' ' + winner.get('tn')
            #date = game.date
            #time = game.game_start_time

            #results["date"].append(date)
            #results["time"].append(time)
            results["game_id"].append(gid)
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

if __name__ == "__main__":
    #lambda_handler(None, None)
    data = get_todays_games_json(todays_games_url)
    results = get_results_from_games(data)
    print(results)