#!/bin/bash

python -m macgraph.train --name spa-10k \
	--max-decode-iterations 4 \
	--disable-input-bilstm \
	--embed-width 128 \
	--disable-message-passing \
	--disable-control-cell \
	--enable-independent-iterations \
	--disable-kb-node \
	--disable-output-cell \
	--output-width 128 \
	--read-activation selu \