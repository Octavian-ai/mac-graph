# MacGraph with Population-based Training

The Population-based training code is imported from [Genetic Curriculum](https://github.com/Octavian-ai/genetic-curriculum) experiment conducted at [Octavian.AI](https://www.octavian.ai). The idea of [Population-based training](https://arxiv.org/abs/1711.09846) originates from [DeepMind](https://deepmind.com).

The core codebase implements graph question answering (GQA), using [CLEVR-graph](https://github.com/Octavian-ai/clevr-graph) as the dataset and [MACnets](https://arxiv.org/abs/1803.03067) as the reasoning architecture.

## Project status

<table>
	<thead>
		<tr>
			<th>Objective</th><th>Status</th><th>Notes</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>Basic PBT configuration</td>
			<td><strong>In progress</strong></td>
			<td>All the custom code lies in the experiment folder</td></tr>
		<tr>
			<td>Recall station (node) properties via PBT</td>
			<td><strong>Not started</strong></td>
			<td>No clue</td></tr>
		
	</tbody>
</table>

<img src="https://media.giphy.com/media/S5JSwmQYHOGMo/giphy.gif"/>

## Running the code

### Prerequisites

## AOB

### Acknowledgments

Thanks to Drew Hudson and Christopher Manning for publishing their work, [Compositional Attention Networks for Machine Reasoning](https://arxiv.org/abs/1803.03067) upon which this is based. Thanks also to DeepMind for publishing their Differentiable Neural Computer results in Nature with a demonstration of that architecture solving graph problems, it is a reassurance that this endeavor is not ill-founded.
