# MacGraph
The MacGraph network. An Irish attempt at intelligence. Puns not included.

This codebase implements graph question answering, using CLEVR-graph as the dataset and MACnets as the reasoning architecture.

### TODO:
- Add input_fn
- Add RNN encoder for question
- Wrap cell up in dynamic_rnn


## Running the code
### Prerequisites

We use the pipenv dependency/virtualenv framework:
```shell
$ pipenv install
$ pipenv shell
(mac-graph-sjOzWQ6Y) $
```

### Building the data

The data (a Graph-Question-Answer YAML from CLEVR-graph) must be pre-processed for training/evaluation. The YAML is transformed into TensorFlow records, and split into test-train-predict tranches.

Build the data:

```shell
python -m mac-graph.input.build
```

#### Arguments
 - *--limit N* will only read N records from the YAML and only output a total of N tf-records (split across three tranches)

### Testing

You can easily run the unit-tests for the model:

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
