# FROM gcr.io/tensorflow/tensorflow:1.7.0-rc0-py3
FROM gcr.io/google-appengine/python
RUN pip install --upgrade pip
RUN pip install pipenv

WORKDIR /source

# Only do costly pipenv install when needed
COPY source/Pipfile .
RUN pipenv install --verbose --skip-lock

COPY source .

CMD "./run-k8.sh"