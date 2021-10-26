import pandas as pd
import http.client
http.client.HTTPConnection.debuglevel = 5
import urllib3
import json

games_header = {
    'user-agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/57.0.2987.133 Safari/537.36',
    'Dnt': '1',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'en',
    'origin': 'http://stats.nba.com',
    'Referer': 'https://github.com'
}

data_headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.4 Safari/605.1.15',
    'Accept-Language': 'en-us',
    'Referer': 'https://stats.nba.com/teams/traditional/?sort=W_PCT&dir=-1&Season=2019-20&SeasonType=Regular%20Season',
    'Connection': 'keep-alive',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true'
}

odds_headers = {
    'x-rapidapi-key': '443730144emsh1ae00e467c31b68p14e65bjsnb5f3aeef6a68',
    'x-rapidapi-host': "odds.p.rapidapi.com"
    }

def get_json_data(url):

    # http = urllib3.PoolManager(10)
    # raw_data = http.urlopen('GET', url, headers=data_headers, timeout=15)
    # #raw_data = http.request("GET", url, timeout=15.0)
    # json_data = json.loads(raw_data.data)
    import requests
    raw_data = requests.get(url, headers=data_headers)
    json = raw_data.json()
    return json.get('resultSets')
    #return json_data.get('resultSets')


def get_todays_games_json(url):
    http = urllib3.PoolManager()
    raw_data = http.request("GET", url, headers=games_header)
    json_data = json.loads(raw_data.data)
    return json_data.get('gs').get('g')

def get_odds_json(url):
    http = urllib3.PoolManager()
    querystring = {"sport":"basketball_nba","region":"us","mkt":"h2h","dateFormat":"unix","oddsFormat":"american"}
    response = http.request("GET", url, headers=odds_headers, fields=querystring)
    return json.loads(response.data)

def to_data_frame(data):
    data_list = data[0]
    return pd.DataFrame(data=data_list.get('rowSet'), columns=data_list.get('headers'))

def create_todays_games(input_list):
    games = []
    for game in input_list:
        home = game.get('h')
        away = game.get('v')
        gid = game.get('gid')
        home_team = home.get('tc') + ' ' + home.get('tn')
        away_team = away.get('tc') + ' ' + away.get('tn')
        games.append([home_team, away_team, gid])
    return games