set title "Accuracy vs number of weak classifiers"
set xlabel "k"
set ylabel "Accuracy"
plot 'results' using 1:2 w lines title "Adaboost accuracy" 
pause -1
