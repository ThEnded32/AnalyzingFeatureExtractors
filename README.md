# Analyzing Feature Extractors
Official repository of the paper "Analyzing the Feature Extractor Networks for Face Image Synthesis"

This repository contains added/modified codes from 3 different repositories;
  https://github.com/GaParmar/clean-fid
  https://github.com/NVlabs/stylegan3
  https://github.com/kynkaat/role-of-imagenet-classes-in-fid

You can download these repositories and copy the given codes to the required directories.

The experiments were done in three steps (excluding the face generation or downsampling);
  1-) First, features are extracted with the codes under the cleanfid folder.
  2-) Evaluating metric results are calculated with the codes under the stylegan3 folder.
  3-) The heatmaps are extracted with codes under the role-of-imagenet-classes-in-fid folder

In addition to these, PacMAP plots are created with the pacmap_plots.py file.

For the environment, the current situation can be explained only this -> https://www.reddit.com/r/ProgrammerHumor/comments/8pdebc/only_god_and_i_knew/

Joke aside, I used two environments following StyleGAN3 and Role of Imagenet Classes in FID repositories. StyleGAN3 repository was not modified, the remaining libraries including PacMAP or CleanFID's dependencies were installed in the other repository.


# TODOs
Add comments to where the modifications are made

Add a jupyter notebook file for demonstration
