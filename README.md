# aperture-controlled-focus-stacking

1. Environment setup

Python version requirement <= 3.7

Install dependencies 
```python3
pip3 install -r requirements.txt
```

We use PWC-net (Sun et al. 2018. PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume, CVPR 2018)as the optical flow calculation, the weights need to be downloaded first, and the path to the weights should be specified in OpticalFlow.py. Please refer to
 https://github.com/philferriere/tfoptflow
For more information about the optical flow prediction by PWC-net 

2. Data preparation

This project needs pre-prepared images and depth maps.  For a given reference lens position and desired mimic f number, one should calculate the corresponding lens positions that can cover the desired depth of field based on the specific camera specifications. The image frames captured at corresponding lens positions, and the depth map of these frames should be input for the pipeline. Note that these images and depth maps should be field-of-view corrected beforehand.

3. Run exemplar case

We prepare a set of example data, which shows reference lens positions at 165 and mimic f number is 4 times larger. Run the example by

```python3
cd path_to_repo
python3 main.py --image_path ./example_data/fov_corrected_image --depth_path ./example_data/fov_corrected_depth --device_name Pixel3-device --ref_p 165
```  
