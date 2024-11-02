#!/bin/bash

git status | grep "post/.*/index.md" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/[[:space:]]\+/ /g' > .modified_post
