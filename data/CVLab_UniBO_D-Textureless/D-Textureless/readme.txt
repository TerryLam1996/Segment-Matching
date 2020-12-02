+============================================================+
|                   D-Textureless dataset                    |
+============================================================+
|                                                            |
|                         CVLab                              |
|   Department of Computer Science and Engineering (DISI)    |
|              University of Bologna, Italy                  |
|                                                            |
|               www.vision.deis.unibo.it                     |
|                                                            |
+------------------------------------------------------------+
|                                                            |
| Changelog:                                                 |
|                                                            |
| [2013.09.24] 1st release                                   |
|                                                            |
+------------------------------------------------------------+

This dataset includes 9 texture-less train models and 55 test
scenes with clutter and occlusions.

It has been acquired with a webcam and comes with hand-labeled
groundtruth.

Each scene includes one or more models, but one instance of each
at most.

Ground truth file structure (model indices are 0-based):

> N                     --> number of models
> model index #1
> model index #2
> ...
> model index #N

---------------------------------------------------------------

A Line2D training file is also included in the "train" directory.
It can be loaded through OpenCV by calling method

cv::linemod::Detector::read()

Note that LineMOD is available starting from OpenCV version 2.4.

References:

 - Project page: www.vision.deis.unibo.it/BOLD
 - "BOLD features to detect texture-less objects", ICCV 2013