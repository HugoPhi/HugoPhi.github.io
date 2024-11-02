#!/bin/bash

git show --name-only --pretty="" | grep "post/.*/index.md" > .modified_post
