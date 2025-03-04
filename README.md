### Author: Gil Pinsky
### Email: gilpinsky1@gmail.com

### Python Version: 3.12.7
### PyTorch Version: 2.3.0
### CUDA Version: 12.1

### Project Description
This project is a mini-solution I developed for zero-shot defect detection.
It is assumed that the 2 images are not aligned, but that the transformation between the images is affine.
The algorithm first aligns the 2 images by using classical methods - RANSAC with SIFT followed by ECC.
Then, the algorithm computes feature maps for both the reference and inspected images using the encoder of a pre-trained segmentation model (resnest269e).
The feature maps are then used to compute the Euclidean distance between the 2 images and thresholding is performed to determine the defect segmentation map.

### Project Structure 

* ```~/solution.ipynb``` - main notebook file, pipeline entrypoint;
* ```~/data``` - directory for the data loading class ```CaseImagePairLoader```;
* ```~/models``` - directory for the model class ```DefectDetector```;
* ```~/utils``` - directory for the image alignment class ```CaseImagePairAligner```;
* ```~requirements.txt``` - list of required dependencies;

### Note
* To  reproduce the solution, install torch 2.3.0 manually