#!/bin/bash

# Filtro i dati JSON
jq 'select(.works != null)' series.json > filtered_series.json

# Ordino i dati filtrati
jq -s 'sort_by(-(.works | map(.books_count | tonumber) | add))[:5]' filtered_series.json > sorted_series.json

# Creo il report in formato desiderato
jq -c '.[] | "ID: \(.id), Series: \(.title), Total Books Count: \(.works | map(.books_count | tonumber) | add)"' sorted_series.json > report.txt

# Pulisco i file intermedi
rm filtered_series.json sorted_series.json



# I decided to use bash's jq extension, which is used specifically to maneuver, modify, or otherwise perform actions with one's JSON file. 
# Specifically at first the file was parsed to remove any rows where works matched an empty array. 
# The filtered file was then transcribed into a new filtered_series.json file so that the modified JSON could be used.
# The second line of code our jq command takes as input the JSON as an array via the "-s" statement and then puts it in
# descending order, via the "-" at the beginning of the sort_by command, based on the total number of books_counts in each series.
# After doing this it saves in a new temporary json the first 5 series for largest total books_count.
# The third line creates for me an output in .txt form where in a compact and elegant way, thanks to jq's "-c" command, it returns us the series 
# we were looking for. it is to specify that sorted_serries.json is parsed in all its components thanks to the command ".[]"