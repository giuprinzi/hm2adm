#!/bin/bash

jq -c 'select(.works != null)' series.json | jq -s 'sort_by(-(.works | map(.books_count | tonumber) | add))[:5] | .[] | "ID: \(.id), Series: \(.title), Total Books Count: \(.works | map(.books_count | tonumber) | add)"' > report_LLM.txt
