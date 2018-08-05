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
			<td>99% accuracy after 10k training steps</td></tr>
		<tr>
			<td>Answer if stations adjacent</td>
			<td><strong>Complete</strong></td>
			<td>99% accuracy after 200k training steps</td>
		</tr>
		<tr>
			<td>Count length of shortest path between nodes</td>
			<td><strong>In progress</strong></td>
			<td></td>
    </tr>
		<tr>
			<td>Recall station (node) properties via PBT</td>
			<td><strong>Not started</strong></td>
			<td>No clue</td></tr>
		
	</tbody>
</table>

<img src="https://media.giphy.com/media/S5JSwmQYHOGMo/giphy.gif"/>

For more in-depth information about what works/doesn't work, check out the [experiment log](log.md).

## Running the code

### Prerequisites

## AOB

### Acknowledgments

Thanks to Drew Hudson and Christopher Manning for publishing their work, [Compositional Attention Networks for Machine Reasoning](https://arxiv.org/abs/1803.03067) upon which this is based. Thanks also to DeepMind for publishing their Differentiable Neural Computer results in Nature with a demonstration of that architecture solving graph problems, it is a reassurance that this endeavor is not ill-founded.
