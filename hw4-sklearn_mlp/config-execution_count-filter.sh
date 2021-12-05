#!/usr/bin/env sh
git config filter.execution_count.clean 'sed "s/\(^\s*"\""execution_count"\"": \)[0-9]\+\(,"\$"\)/\1null\2/"'
