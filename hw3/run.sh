#!/bin/bash
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

module load SpectrumMPI/10.1.0

mpisubmit.pl -p $1 -w 00:30 -t $2 --stdout logs/$1_$2_$3_$4.out --stderr logs/$1_$2_$3_$4.err ./app -- $3 $4
