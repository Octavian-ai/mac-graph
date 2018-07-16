# MacGraph
The MacGraph network. An Irish attempt at intelligence. Puns not included.

This codebase implements graph question answering, using [CLEVR-graph](https://github.com/Octavian-ai/clevr-graph) as the dataset and [MACnets](https://arxiv.org/abs/1803.03067) as the reasoning architecture.

<img src="https://media.giphy.com/media/S5JSwmQYHOGMo/giphy.gif"/>

## Project status

<table>
	<thead>
		<tr>
			<th>Objective</th><th>Status</th>
		</tr>
	</thead>
	<<tbody>
		<tr><td>Basic MAC cell structure</td><td><strong>Complete</strong></td></tr>
		<tr><td>Recall station (node) properties</td><td><strong>Complete</strong>: 99% accuracy on `--type-string-prefix StationProperty` questions</td></tr>
		<tr><td>Count length of shortest path between nodes</td><td><strong>In progress</strong>. No capability better than random guessing</td></tr>
		<tr><td>Find modal properties of lines</td><td>Not started. No observed capability</td></tr>
		<tr><td>Count stations meeting criteria on lines</td><td>Not started. No observed capability</td></tr>
		<tr><td>Other line-based queries</td><td>Not started. No observed capability</td></tr>
		<tr><td>List shortest routes</td><td>Not started. No capability</td></tr>
	</tbody>
</table>

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

If you want to skip this step, you can download the pre-built data from [our public dataset](https://www.floydhub.com/davidmack/datasets/mac-graph). This repo is a work in progress so the format is still in flux.

The underlying data (a Graph-Question-Answer YAML from CLEVR-graph) must be pre-processed for training and evaluation. The YAML is transformed into TensorFlow records, and split into test-train-predict tranches.

First [generate](https://github.com/Octavian-ai/clevr-graph) a `gqa.yaml` with the command:
```shell
clevr-graph$ python -m gqa.generate --count 50000 --quick --only-type StationProperty --int-names
cp data/gqa-some-id.yaml ../mac-graph/input_raw/gqa.yaml
```

Then build (that is, pre-process into tfrecords) the data:

```shell
mac-graph$ python -m mac-graph.input.build --gqa-path input_raw/gqa.yaml
```

#### Arguments
 - `--limit N` will only read N records from the YAML and only output a total of N tf-records (split across three tranches)

### Training

Let's build a model. (Note, this requires training data from the previous section).

General advice is to have at least 40,000 training records (e.g. build from 50,000 GQA triples)

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

Thanks to Drew Hudson and Christopher Manning for publishing their work, [Compositional Attention Networks for Machine Reasoning](https://arxiv.org/abs/1803.03067) upon which this is based. 

### A limerick

Since you're here.

> There once was an old man of Esser,<br/>
> Whose knowledge grew lesser and lesser,<br/>
> It at last grew so small<br/>
> He knew nothing at all<br/>
> And now he's a college professor.
