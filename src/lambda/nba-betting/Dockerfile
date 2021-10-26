FROM amazon/aws-lambda-python:3.7
COPY requirements.txt .
RUN yum -y install git
RUN pip install -r requirements.txt
COPY src/ .
CMD echo "Define a command to run."