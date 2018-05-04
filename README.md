# ebrahimnasr/DFCN
The code, related to skin lesion segmentation, is provided here and segmentation results are accessible in the "Result" folder.
DFCN is implemented on "caffe" framework; thus, your system requires "caffe" framework as prerequisite to run this project.

You may follow these steps:
First, run "CreateDataBase.m" in matlab to create the skin dataset and prepare it for network training.
Next, run "TrainModel.py" to start network training on the python platform.
Last, run "TestModel.py" to test the network under python platform.

The implemented layers of DFCN, convolutional and dense pooling layers, are available in "ReshapeLayers.py".



Should you have any question, contact us via email: ebrahim.nasr@gmail.com
