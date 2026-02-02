#!/usr/bin/env bash
source /cm/shared/apps/amh-conda/etc/profile.d/conda.sh
conda activate verieql
python3 /home/common/txt2sql-verieql/verieql/run_llm.py "$@"
