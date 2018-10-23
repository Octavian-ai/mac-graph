
# Running this code

## Working with the network locally

### Prerequisites

We use the pipenv dependency/virtualenv framework:
```shell
$ pipenv install
$ pipenv shell
(mac-graph-sjOzWQ6Y) $
```

### Prediction

You can watch the model predict values from the hold-back data:
```shell
$ python -m macgraph.predict --name my_dataset --model-version 0ds9f0s

predicted_label: shabby
actual_label: derilict
src: How <space> clean <space> is <space> 3 ? <unk> <eos> <eos>
-------
predicted_label: small
actual_label: medium-sized
src: How <space> big <space> is <space> 4 ? <unk> <eos> <eos>
-------
predicted_label: medium-sized
actual_label: tiny
src: How <space> big <space> is <space> 7 ? <unk> <eos> <eos>
-------
predicted_label: True
actual_label: True
src: Does <space> 1 <space> have <space> rail <space> connections ? <unk>
-------
predicted_label: True
actual_label: False
src: Does <space> 0 <space> have <space> rail <space> connections ? <unk>
-------
predicted_label: victorian
actual_label: victorian
src: What <space> architectural <space> style <space> is <space> 1 ? <unk>
```

**TODO: Get it predicting from your typed input** 

### Building the data

To train the model, you need training data.

If you want to skip this step, you can download the pre-built data from [our public dataset](https://www.floydhub.com/davidmack/datasets/mac-graph). This repo is a work in progress so the format is still in flux.

The underlying data (a Graph-Question-Answer YAML from CLEVR-graph) must be pre-processed for training and evaluation. The YAML is transformed into TensorFlow records, and split into train-evaluate-predict tranches.

First [generate](https://github.com/Octavian-ai/clevr-graph) a `gqa.yaml` with the command:
```shell
clevr-graph$ python -m gqa.generate --count 50000 --int-names
cp data/gqa-some-id.yaml ../mac-graph/input_data/raw/my_dataset.yaml
```
Then build (that is, pre-process into a vocab table and tfrecords) the data:

```shell
mac-graph$ python -m macgraph.input.build --name my_dataset
```

#### Arguments to build
 - `--limit N` will only read N records from the YAML and only output a total of N tf-records (split across three tranches)
 - `--type-string-prefix StationProperty` will filter just questions with type string prefix "StationProperty"


### Training

Let's build a model. (Note, this requires training data from the previous section).

General advice is to have at least 40,000 training records (e.g. build from 50,000 GQA triples)

```shell
python -m macgraph.train --name my_dataset
```

Running too slow? Fan spinning too much? Use [FloydHub](https://docs.floydhub.com/guides/basics/install/) with this magic button:

[![Run on FloydHub](https://static.floydhub.com/button/button.svg)](https://floydhub.com/run)

Or manually run on FloydHub with `floyd run`

### Testing

You can easily run the unit tests for the model:

```shell
python -m macgraph.test
```

Also the model construction functions contain many assertions to help validate correctness.


## Running genetic optimization

### Run local:

To run the system locally as single process:
```shell
pipenv install
pipenv shell

python -m experiment.k8 --master-works
```


To test:
```shell
pipenv shell
./script/test.sh
```


## Deploying to Kubernetes (A cheatsheet of K8 runes)

### Install RabitMQ

- Install Helm
- Initialise Helm on your cluster as per their docs

Give helm permissions on GKE:
```
kubectl create clusterrolebinding --user system:serviceaccount:kube-system:default kube-system-cluster-admin --clusterrole cluster-admin
```

Install a queue:
```
helm install --name one --set rabbitmq.username=admin,rabbitmq.password=secretpassword,rabbitmq.erlangCookie=secretcookie     stable/rabbitmq
```

Check the console output to get the AMPQ url and password for your new queue

### Install our PBT application

Update secret.yaml to have the url to your AMQP queue.

Deploy the config:

```
kubectl create -f kubernetes/mac-graph.yaml
```

Set up permissions:
```
kubectl create serviceaccount default --namespace default
kubectl create clusterrolebinding default-cluster-rule --clusterrole=cluster-admin --serviceaccount=default:default
```


To see dashboard:
```
gcloud config config-helper --format=json | jq --raw-output '.credential.access_token'
kubectl proxy
```
