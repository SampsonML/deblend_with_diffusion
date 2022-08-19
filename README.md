# Diffusion models for deblending

## Matt Sampson - Semester 1 2022 Fall project
### Background info
Large scale images such as Hubble deep field, now JWST deep field, contain many astrophysical feature in high density. A common issue is the pixel densities of two or more potentially overlapping galaxies. This could be a physical interaction, or more likely a situation where the galaxies lie in the same same line-of-sight however are completely independent. To analyse these galaxies we ideally would like to eliminate any non-physical pixel blending ie. we wish to deblend the galaxy images.

This project will be tied into work with SCARLET (developed by P Melchior) https://ui.adsabs.harvard.edu/abs/2018A&C....24..129M. Scarlet deblends images and reducing noise and interactions from independant galaxies.
![Melchior+2022](https://github.com/SampsonML/deblend_with_diffusion/blob/main/images/scarlet_deep_field.png)

To deal with this problem we may approach the task of deblending through the lens of probabilistic machine learning. To be more clear we explicitly the problem is as such (from Francios+2019)

Consider a general linear inverse problems of the form:

y = Ax + n

where y are the observations, x is the unknown signal to recover, A is a linear degradation operator, n is some observational noise. We would like to determine x. A baysian approach tells us to generate a posterior distribution for y as below,

p(x|y) \propto / p(y|x) p(x).

Where, p(y|x) is just the explicit observational data we havethe data likelihood term, and p(x) is our prior. It in this prior in which we aim to develop  diffusion based machine learning model to construct. Similar to work here but with diffusion models instead of neural networks  (https://ui.adsabs.harvard.edu/abs/2019arXiv191203980L/abstract). 
## Project Aims
* Aims:
  1. Create a diffusion model for our prior p(x)
  2. Train this model on large sets of data
  3. Use the newly constructed prior to then build a posterior, which may then be updated with new images without the need to update p(x)


## Plan

* General steps to completion (to be filled out later on)
  1. Step 1
  2. Step 2
  3. Step 3
  4. Step 4
  
* Important notes
  1. Compute infrastructure, Princeton HPC?
  2. Build with jax - allowing for GPU or CPU utalisation with no code changes
  3. Learn most efficient way to utalise jax's jit (just in time) compilation
  4. How will we build our initial prior? Do we need a baseline prior which then is optimised with the ML routine?
 

## Useful papers
### For context scientific context:

Scarlet paper: (https://ui.adsabs.harvard.edu/abs/2018A&C....24..129M)

### Similar work:

Burke+2019 (https://arxiv.org/abs/1908.02748)

Huang+2022 (https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)

## Useful other rescources
Lilian Weng blogpost (https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

Yang Song blogpost (https://yang-song.github.io/blog/2021/score/)

Ryan O'Conner AssemblyAI (https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)
  
  
  



