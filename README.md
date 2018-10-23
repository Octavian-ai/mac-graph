# MacGraph with Population-based Training

The Population-based training code is imported from [Genetic Curriculum](https://github.com/Octavian-ai/genetic-curriculum) experiment conducted at [Octavian.AI](https://www.octavian.ai). The idea of [Population-based training](https://arxiv.org/abs/1711.09846) originates from [DeepMind](https://deepmind.com).

The core codebase implements graph question answering (GQA), using [CLEVR-graph](https://github.com/Octavian-ai/clevr-graph) as the dataset and [MACnets](https://arxiv.org/abs/1803.03067) as the reasoning architecture.

## Project status

*Apologies that the training data isn't available - I've yet to find a quick solution to this, when I get the system working on more questions I'll publish a stable "all question" dataset with 1M items. For now, you can easily build you own data - ask David for help.*

<table>
	<thead>
		<tr>
			<th>Objective</th><th>Status</th><th>Notes</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>Basic MAC cell structure</td>
			<td><strong>Complete</strong></td>
			<td>Implemented as per paper, now diverging to achieve below objectives</td></tr>
		<tr>
			<td>Recall station (node) properties</td>
			<td><strong>Complete</strong></td>
			<td>99.9% accuracy after 10k training steps</td></tr>
		<tr>
			<td>Answer if stations adjacent</td>
			<td><strong>Complete</strong></td>
			<td>99% accuracy after 20k training steps</td>
		</tr>
		<tr>
			<td>Stations N apart</td>
			<td><strong>Semi-complete</strong></td>
			<td>98% accuracy up to ~9 apart after 25k training steps</td>
    	</tr>
    	<tr>
			<td>Station existence</td>
			<td><strong>Semi-complete</strong></td>
			<td>99.9% accuracy after 30k training steps</td>
    	</tr>
    	<tr>
			<td>Station with property adjacent</td>
			<td><strong>TBC</strong></td>
			<td></td>
    	</tr>
		<tr>
			<td>Count length of shortest path between nodes</td>
			<td><strong>TBC</strong></td>
			<td></td>
    	</tr>
	</tbody>
</table>

<img src="https://media.giphy.com/media/S5JSwmQYHOGMo/giphy.gif"/>

For more in-depth information about what works/doesn't work, check out the [experiment log](log.md).

## Running the code

See [RUNNING.md](RUNNING.md) for how to both run the network and also use a cluster to optimize its hyper-parameters.

The short summary of how to train locally:
```
$ pipenv install
$ pipenv shell
(mac-graph-sjOzWQ6Y) $ python -m macgraph.train
```

## AOB

### Acknowledgments

Thanks to Drew Hudson and Christopher Manning for publishing their work, [Compositional Attention Networks for Machine Reasoning](https://arxiv.org/abs/1803.03067) upon which this is based. Thanks also to DeepMind for publishing their Differentiable Neural Computer results in Nature with a demonstration of that architecture solving graph problems, it is a reassurance that this endeavor is not ill-founded.

### A limerick

Since you're here.

> There once was an old man of Esser,<br/>
> Whose knowledge grew lesser and lesser,<br/>
> It at last grew so small<br/>
> He knew nothing at all<br/>
> And now he's a college professor.
