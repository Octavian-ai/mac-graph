# MacGraph
The MacGraph network. An Irish attempt at intelligence. Puns not included.

This codebase implements graph question answering, using CLEVR-graph as the dataset and MACnets as the reasoning architecture.

<img src="https://media.giphy.com/media/S5JSwmQYHOGMo/giphy.gif"/>


## Running the code

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
$ python -m mac-graph.predict

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

If you want to skip this step, you can download the pre-built data from [our public dataset](https://www.floydhub.com/davidmack/datasets/mac-graph).

The underlying data (a Graph-Question-Answer YAML from CLEVR-graph) must be pre-processed for training and evaluation. The YAML is transformed into TensorFlow records, and split into test-train-predict tranches.

First [download](https://storage.googleapis.com/octavian-static/download/english2cypher/gqa.zip) (or generate!) a `gqa.yaml`.

Then build the data:

```shell
python -m mac-graph.input.build
```

#### Arguments
 - `--limit N` will only read N records from the YAML and only output a total of N tf-records (split across three tranches)

### Training

Let's build a model. (Note, this requires training data from the previous section)

```shell
python -m mac-graph.train
```

Running too slow? Fan spinning too much? Use [FloydHub](https://docs.floydhub.com/guides/basics/install/) with this magic button:

[![Run on FloydHub](https://static.floydhub.com/button/button.svg)](https://floydhub.com/run)

Or manually run on FloydHub with `floyd run`

### Testing

You can easily run the unit tests for the model:

```shell
python -m mac-graph.test
```

Also the model construction functions contain many assertions to help validate correctness.


## AOB

### Acknowledgements

Thanks to Drew Hudson and Christopher Manning for publishing their work, [Compositional Attention Networks for Machine Reasoning](https://arxiv.org/abs/1803.03067) upon which this is based. Also, thank you [coffee](https://thebarn.de/) and [techno](https://soundcloud.com/ostgutton-official/berghain-07-function) for your company.

### A limerick

Since you're here.

> There once was an old man of Esser,<br/>
> Whose knowledge grew lesser and lesser,<br/>
> It at last grew so small<br/>
> He knew nothing at all<br/>
> And now he's a college professor.
