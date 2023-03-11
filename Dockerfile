FROM python:3.10.6-bullseye

# Dependencies installation
COPY requirements_prod.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements

# Install travel-home
COPY travel-home travel-home
COPY setup.py setup.py
RUN pip install .

# Then the makefile and so one
COPY Makefile Makefile
# RUN make reset_local_files

CMD uvicorn travel-home.api.fast:app --host 0.0.0.0 --port $PORT
