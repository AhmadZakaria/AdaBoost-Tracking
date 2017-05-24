Adaboost tracking.

Group members:

Valentina Osipova;
Ahmad Z. Mohammad.

To run the code: 

``
cmake .
make

./bin/nr1 face-model.xml img2.jpg

./bin/nr2 splice/splice.train splice/splice.test 50

./bin/nr3 nemo/frames.train nemo/frames.test 50
``

Notes on the number of weak classifiers:
as the number increases, the overall performance gets better, but it is very noisy as the stochastic search for a split value/attribute is different.
However, the accuracy saturates at some point, and even shows some degradation after 20
Kindly check the attached graph.



For tracking nemo, 
Using non-overlapping negative examples enhanced the confidence drastically. 
Using overlapping negative examples confused the weak classifiers, and produced not-so-satisfactory results (unpredictable and inconsistent)

