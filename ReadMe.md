# Purpose

The MANTIS satellite must make inferences on images on board the sattellite to decide what information should be downlinked. There are two primary objectives for this satellite, one is to monitor change in tree phenology, the other is to detect harmfula algal blooms in bodies of water. Because the two objectives do not overlap, to reduce commputation only images of water should be processed in the HAB detection model, and only images that have trees in them should be processed through the Tree Phen model. 

A solution is to use a segmentation model to preprocess the images to decide which model the image should be processed by. 

The proposed solution will classify pixels into three classes, cloud, land, and water.

