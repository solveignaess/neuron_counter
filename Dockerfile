FROM python:3.8

COPY src /app/src
COPY requirements.txt /app
COPY setup.py /app

WORKDIR /app

RUN pip install -r requirements.txt

WORKDIR /app/src

CMD streamlit run test_app.py