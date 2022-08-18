# Diffusion models for deblending

## Matt Sampson 2022 - Semester 1 2022 Fall project
### Background info
This project will be tied into work with SCARLET (developed by P Melchior) https://ui.adsabs.harvard.edu/abs/2018A&C....24..129M. Scarlet deblends images reducing noise and interactions from independant galaxies.
![Melchior+2022](https://github.com/SampsonML/deblend_with_diffusion/blob/main/images/scarlet_deep_field.png)


We may approach the task of deblending through the lens of probabilistic machine learning. 

The problem from Francios+2019
We consider a general linear inverse problems of the form:

y = Ax + n

where y are the observations, x is the unknown signal to recover, A is a linear degradation operator, n is some observational noise. A baysian approach tells us to generate a posterior distribution for y as below,

p(x|y) \propto / p(y|x) p(x).

Where, p(y|x) is just the explicit observational data we havethe data likelihood term, and p(x) is our prior. It in this prior in which we aim to develop ma diffusion based machine learning model to calculate. Similar to work here but with diffusion models instead of neural networks (https://ui.adsabs.harvard.edu/abs/2019arXiv191203980L/abstract). 
## Project Aims

Galaxy images from large surveys will onclude many blended images. Performing a deblending of these is not trivial.
One approach

## Plan

* General steps to completion
  1. Step 1
  2. Step 2
  3. Step 3
  4. Step 4
 

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
  
  
  



