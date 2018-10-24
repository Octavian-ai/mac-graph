#!/bin/bash

python -m macgraph.train --name spa-10k \
	--max-decode-iterations 2 \
	--disable-input-bilstm \
	--embed-width 32 \
	--disable-message-passing