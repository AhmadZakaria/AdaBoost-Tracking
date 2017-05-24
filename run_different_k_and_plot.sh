#!/bin/bash
rm -f results
for i in 1 2 3 4 5 10 15 20 30 40 50 ; do acc=`./bin/nr2 splice/splice.train splice/splice.test $i | tail -1 | cut -d" " -f 2`; echo "$i $acc" >> results; done
echo "As we notice, the accuracy saturates at some point, and increasing the number of weak classifiers becomes useless."
gnuplot plot_results.gnuplot


