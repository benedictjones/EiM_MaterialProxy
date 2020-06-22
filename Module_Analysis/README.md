# Analysis Object & Modules

These modules deal with the handling of previously generated data from either Results folders or Experiment Folders.
An Analysis object can be called which references the location of the data which is to be plotted.

The function anly_run.py in the main directory has examples of this.

Animations can be produced to show how the best member of the population evolves etc. 
**By default this is not enabled**. 
To enable install ImageMagick (https://imagemagick.org/script/download.php). 
Specifically ffmpeg is used to save an animation. 
Then navigate to line 31 in Analysis.py and uncomment #Animation.init(self)