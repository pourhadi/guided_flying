FROM ubuntu

WORKDIR /app

ADD "model.model" /app/model.model
ADD "app.py" /app

RUN apt-get update
RUN apt-get install -y python3.6
RUN apt-get install -y python3-pip 
RUN apt-get install -y libblas3 liblapack3 libstdc++6 python-setuptools

RUN pip3 install turicreate==5.0b3 watchdog

CMD ["python3", "app.py"]