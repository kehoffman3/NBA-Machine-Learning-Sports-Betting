#!/bin/bash
for name in $(aws lambda list-functions --region us-east-1 --query 'Functions[?starts_with(FunctionName, `nba-betting`) == `true`].FunctionName' --output text);
do
    echo $name;
    aws lambda update-function-code --function-name $name --image-uri $IMAGE_NAME;
done;