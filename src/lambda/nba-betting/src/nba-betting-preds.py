import json
from datetime import datetime
import pandas as pd
import logging
import boto3
from botocore.exceptions import ClientError
from io import StringIO
import pytz
import numpy as np
import logging
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import load_model
from data import get_json_data, to_data_frame, get_todays_games_json, create_todays_games, get_odds_json

logger = logging.getLogger()
logger.setLevel(logging.INFO)
    

todays_games_url = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2021/scores/00_todays_scores.json'
data_url = 'https://stats.nba.com/stats/leaguedashteamstats?' \
           'Conference=&DateFrom=&DateTo=&Division=&GameScope=&' \
           'GameSegment=&LastNGames=0&LeagueID=00&Location=&' \
           'MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&' \
           'PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&' \
           'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&' \
           'Season=2021-22&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
           'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='
odds_url = "https://odds.p.rapidapi.com/v1/odds"

team_index_current = {
    'Atlanta Hawks': 0,
    'Boston Celtics': 1,
    'Brooklyn Nets': 2,
    'Charlotte Hornets': 3,
    'Chicago Bulls': 4,
    'Cleveland Cavaliers': 5,
    'Dallas Mavericks': 6,
    'Denver Nuggets': 7,
    'Detroit Pistons': 8,
    'Golden State Warriors': 9,
    'Houston Rockets': 10,
    'Indiana Pacers': 11,
    'Los Angeles Clippers': 12,
    'LA Clippers': 12,
    'Los Angeles Lakers': 13,
    'Memphis Grizzlies': 14,
    'Miami Heat': 15,
    'Milwaukee Bucks': 16,
    'Minnesota Timberwolves': 17,
    'New Orleans Pelicans': 18,
    'New York Knicks': 19,
    'Oklahoma City Thunder': 20,
    'Orlando Magic': 21,
    'Philadelphia 76ers': 22,
    'Phoenix Suns': 23,
    'Portland Trail Blazers': 24,
    'Sacramento Kings': 25,
    'San Antonio Spurs': 26,
    'Toronto Raptors': 27,
    'Utah Jazz': 28,
    'Washington Wizards': 29
}

def createTodaysGames(games, df):
    match_data = []

    for game in games:
        home_team = game[0]
        away_team = game[1]
        home_team_series = df.iloc[team_index_current.get(home_team)]
        away_team_series = df.iloc[team_index_current.get(away_team)]
        stats = home_team_series.append(away_team_series)
        match_data.append(stats)
    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T

    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'CFID', 'CFPARAMS', 'TEAM_NAME'])
    data = frame_ml.values
    data = data.astype(float)

    return data, frame_ml

def lambda_handler(event, context):
    tz = pytz.timezone('US/Eastern')
    today = datetime.now(tz)
    print("Getting todays games...")
    data = get_todays_games_json(todays_games_url)
    print("Retrieved todays game")
    games = create_todays_games(data)
    
    if not games:
        return {
            'statusCode': 200,
            'body': 'No games today!'
        }
    print("Getting stats data...")
    data = get_json_data(data_url)
    print("Retrieved stats data")
    df = to_data_frame(data)
    data, _ = createTodaysGames(games, df)
    
    data = normalize(data, axis=1)
    

    print("Loading model...")
    model = load_model('models/nba_nn_v1')
    print("Making predictions...")
    preds_df = predict_nba_games(data, games, model)

    # Make predictions and save to seperate file
    final_df = get_odds_and_ev(preds_df)
    if (event and not event.get("email_only")) or not event:
       preds_file_name = f"todays-games-preds/NBA_PREDS_{today.strftime('%Y_%m_%d')}.csv"
       save_to_s3(final_df, preds_file_name)

    #Send an email
    email_nba_preds(final_df)

    # print(final_df)
    
    return {
        'statusCode': 200,
        'body': data.to_json()
    }

def to_percent(col):
    return col.astype(float).map(lambda n: '{:.2%}'.format(n))

def email_nba_preds(df):

    formatted_df = df[['away_team', 'home_team', 'away_odds', 'home_odds', 'best_pick', 'away_prob', 'home_prob',  'best_pick_ev']]

    formatted_df['away_prob'] = to_percent(formatted_df['away_prob'])
    formatted_df['home_prob'] = to_percent(formatted_df['home_prob'])
    formatted_df['best_pick_ev'] = to_percent(formatted_df['best_pick_ev'])

    formatted_df.rename({'away_team':'Away Team', 'home_team': 'Home Team', 'away_prob': 'Away Win Probability', 'home_prob': 'Home Win Probability', 'away_odds': 'Away Odds', 'home_odds': 'Home Odds', 'best_pick': 'Model Prediction', 'best_pick_ev': "Expected Value"}, axis=1, inplace=True)

    today = datetime.now()
    

    # Replace sender@example.com with your "From" address.
    # This address must be verified with Amazon SES.
    SENDER = "Kevin Hoffman <kehoffman3@gmail.com>"

    # Replace recipient@example.com with a "To" address. If your account 
    # is still in the sandbox, this address must be verified.
    RECIPIENT = "kehoffman3@gmail.com"

    # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
    AWS_REGION = "us-east-1"

    # The subject line for the email.
    SUBJECT = f"NBA Predictions for {today.strftime('%b %d, %Y')}"
                
    # The HTML body of the email.
    BODY_HTML = formatted_df.to_html(index=False)          

    # The character encoding for the email.
    CHARSET = "UTF-8"

    # Create a new SES resource and specify a region.
    client = boto3.client('ses',region_name=AWS_REGION)

    # Try to send the email.
    try:
        #Provide the contents of the email.
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': CHARSET,
                        'Data': BODY_HTML,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER,
        )
    # Display an error if something goes wrong.	
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])

def save_to_s3(data, file_name):
    bucket = 'kehoffmn3-nba'
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, file_name).put(Body=csv_buffer.getvalue())



def predict_nba_games(data, games, model):
    away_probs = []
    home_probs = []
    home_teams = []
    away_teams = []
    game_ids = []
    for idx, row in enumerate(data):
        pred = model.predict(np.array([row]))
        
        home_probs.append(pred[0][1])
        away_probs.append(pred[0][0])
        home = games[idx][0]
        away = games[idx][1]
        game_id = games[idx][2]
        home_teams.append(home)
        away_teams.append(away)
        game_ids.append(game_id)

    return pd.DataFrame({
        'game_id': game_ids,
        'home_team': home_teams,
        'away_team': away_teams,
        'home_prob': home_probs,
        'away_prob': away_probs
    })



def calc_implied_prob(odds):
    if odds<0:
        return abs(odds / (abs(odds) + 100))
    else:
        return 100 / (odds + 100)

def get_odds_and_ev(df):

    tz = pytz.timezone('US/Eastern')
    today = datetime.now(tz)
    
    response_json = get_odds_json(odds_url)

    sportsbooks = ["betmgm", "draft_kings" "williamhill_us"]
    home_odds_list = []
    home_team_list = []
    away_odds_list = []
    away_team_list = []


    todays_games = list(filter(lambda g: datetime.fromtimestamp(g["commence_time"]).astimezone(tz).day == today.day, response_json["data"]))

    for game in todays_games:
        teams = game["teams"]
        home_team = game["home_team"]

        
        odds_dict = {}
        for site in game["sites"]:
            odds_dict[site["site_key"]] = site["odds"]["h2h"]
        
        home_odds = None
        away_odds = None
        for sportsbook in sportsbooks:
            odds = odds_dict.get(sportsbook)
            if odds:

                home_idx = 1
                away_idx = 0
                if home_team == teams[0]:
                    home_idx = 0
                    away_idx = 1
                
                home_team_abbr = teams[home_idx]
                home_odds = odds[home_idx]
                home_odds_list.append(home_odds)
                home_team_list.append(home_team_abbr)

                away_team_abbr = teams[away_idx]
                away_odds = odds[away_idx]
                away_odds_list.append(away_odds)
                away_team_list.append(away_team_abbr)
                break

    odds_df = pd.DataFrame({"home_team": home_team_list, "away_odds":away_odds_list, "home_odds": home_odds_list})

    final_df = df.merge(odds_df, left_on="home_team", right_on="home_team", how='left')
    final_df["away_implied"] = final_df.apply(lambda d: calc_implied_prob(d["away_odds"]),axis=1)
    final_df["home_implied"] = final_df.apply(lambda d: calc_implied_prob(d["home_odds"]),axis=1)
    final_df["away_ev"] = final_df["away_prob"] - final_df["away_implied"]
    final_df["home_ev"] = final_df["home_prob"] - final_df["home_implied"]
    final_df["best_ev"] = np.where(final_df["home_ev"] > final_df["away_ev"], final_df["home_ev"], final_df["away_ev"])
    final_df["betting_team"] = np.where(final_df["home_ev"] > final_df["away_ev"], final_df["home_team"], final_df["away_team"])
    final_df["best_pick_ev"] = np.where(final_df["home_prob"] >= final_df["away_prob"], final_df["home_ev"], final_df["away_ev"])
    final_df["best_pick"] = np.where(final_df["home_prob"] >= final_df["away_prob"], final_df["home_team"], final_df["away_team"])
    final_df.sort_values(by="best_pick_ev", ascending=False, inplace=True)
    #print(final_df)
    return final_df


if __name__ == "__main__":
    lambda_handler(None, None)


