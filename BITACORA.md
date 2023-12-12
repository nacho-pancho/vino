# Sandbox for the VID project

Some stuff I've done so far:
* tried opencv to read MP4 directly instead of decoding frames; worked

What I want to do:
* I want to segment grapes using dictionary learning, the old way
* I think the best is to label some images and train a dictionary for the background and the foreground
* I may want to revive COSMOS/IRP to train them
* I will use Labelme for this:
  * `conda install -c conda-forge pyqt`
  * `conda install -c conda forge labelme`
