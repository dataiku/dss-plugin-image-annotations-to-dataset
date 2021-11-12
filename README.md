# Plugin information

This repository contains a recipe to transform annotations from common object detection formats 
- [COCO](https://cocodataset.org/#format-data)
- [VOC pascal](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#SECTION00050000000000000000) 

into a Dataset compatible for object detection in DSS.


Warning: note that in official Pascal-VOC format "The top-left pixel in the image has coordinates (1,1)". Thus the 
recipe will remove this 1-pixel offset to return 0-based coordinates that are DSS compliant.