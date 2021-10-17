import pytz
from datetime import datetime, timedelta
import boto3
import pandas as pd
from botocore.exceptions import ClientError
import numpy as np

def get_result(previous_date):
    bucket = 'kehoffmn3-baseball'

    s3_client = boto3.client('s3')
    results_file_name = f"game-betting-results/MLB_RESULTS_FULL_{previous_date.strftime('%Y_%m_%d')}.csv"
    response = s3_client.get_object(Bucket=bucket, Key=results_file_name)
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        results_df = pd.read_csv(response.get("Body"))
    else:
        return None
    return results_df

def email_summary(df, today, lookback):
    

    # Replace sender@example.com with your "From" address.
    # This address must be verified with Amazon SES.
    SENDER = "Kevin Hoffman <kehoffman3@gmail.com>"

    # Replace recipient@example.com with a "To" address. If your account 
    # is still in the sandbox, this address must be verified.
    RECIPIENT = "kehoffman3@gmail.com"

    # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
    AWS_REGION = "us-east-1"

    # The subject line for the email.
    SUBJECT = f"MLB Summary: {(today - timedelta(days=lookback)).strftime('%b %d, %Y')} - {today.strftime('%b %d, %Y')}"
                
    # The HTML body of the email.
    BODY_HTML = df.to_html(index=False)          

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

def lambda_handler(event, context):
    tz = pytz.timezone('US/Eastern')
    today = datetime.now(tz)
    lookback = event.get("lookback", 7)

    wins = 0
    losses = 0
    pushes = 0
    profit = 0
    games = 0
    for i in range(1, lookback+1):
        date = today - timedelta(days=i)
        results_df = get_result(date)
        if results_df is None:
            continue
        for idx, row in results_df.iterrows():
            res = row["result"]
            prof = row["profit"] or 0
            games += 1
            if res == "W":
                wins += 1
            elif res == "L":
                losses += 1
            elif res == "P":
                pushes += 1
            else:
                games -= 1

            print(prof)
            print(profit)
            if prof and not np.isnan(prof):
                profit += float(prof)
    summary_df = pd.DataFrame({"wins": [wins], "losses": [losses], "pushes": [pushes], "profit": [profit], "bets_per_day":["{:.2f}".format(games/lookback)]})
    email_summary(summary_df, today, lookback)
    
    return {
        'statusCode': 200,
        'body': summary_df.to_json()
    }
