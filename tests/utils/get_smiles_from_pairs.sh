#!/bin/bash
filename="$1"
while read -r line; do
    for word in $line; do 
	echo $word
    done
done < "$filename"
