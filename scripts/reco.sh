#!/bin/bash

BIN=$1

$BIN models/2.gram.graph models/monophone_model_I32 --sil SP -m 20 -b 200 samples.lst --prune-after



