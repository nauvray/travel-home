FROM python:3.10.6-bullseye

# Dependencies installation
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install travel-home
COPY gcpkey.json gcpkey.json
COPY travel_home travel_home
COPY setup.py setup.py
RUN pip install .

# Then the makefile and so one
COPY Makefile Makefile
# RUN make reset_local_files

CMD uvicorn travel_home.api.fast:app --host 0.0.0.0 --port $PORT
