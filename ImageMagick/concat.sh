#!/bin/bash

# Goto directory A
cd /Users/tinnguyen/Downloads/toy_class_examples/union_3/explanations/

# For each file "f" in A
for f in *.jpg; do
    # Append corresponding file from B and write to AB
    convert ../../union_1_manual/explanations/"$f" "$f" -append ../../concat/"$f"
    convert ../../concat/"$f" -gravity north -pointsize 30 -annotate +0+0 'Toy example' ../../concat/"$f"
    convert ../../concat/"$f" -gravity center -pointsize 30 -annotate +0+20 'XCLIP' ../../concat/"$f"

done
