FROM ubuntu:focal

COPY requirements.txt .

RUN apt-get update
RUN apt-get install -y python3.8 python3.8-distutils python3.8-dev pip
RUN pip install -r requirements.txt
# Had to downgrade this package for some reason
RUN pip install jinja2==3.0.3
RUN modal token new

ENTRYPOINT ["sleep", "infinity"]
