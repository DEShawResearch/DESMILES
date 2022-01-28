#!/bin/bash

time spm_encode --model ${DESMILES_DATA_DIR}/pretrained/train_val1_val2/bpe_v8000.model --output $2 $1 --output_format=id --extra_options=bos:eos
