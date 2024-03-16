# Depth Estimation and Boundary Extraction

This project implements a multi-step pipeline for extracting boundary depth from images. The process involves estimating a depth map from a monocular image, back-projecting this image into a 3D triangular mesh, and then extracting planar surfaces to approximate boundary conditions. This approach is inspired by the methodology described in the paper "LOOSECONTROL: Lifting ControlNet for Generalized Depth Conditioning."

## Installation

Before running the script, ensure that you have Python 3.6 or newer installed on your system. You will also need to install the following dependencies:

```bash
pip install torch torchvision Pillow transformers numpy open3d scipy matplotlib opencv-python
```

## How It Works:

1. **Depth Map Estimation:** The first step is to estimate the depth map of the given image using a monocular depth estimator. In the provided code, this is accomplished using the ```DepthEstimator``` class, which utilizes the ```DPTForDepthEstimation``` model from the transformers library. The input image is processed and passed through the model to obtain a depth map.

2. **3D Triangular Mesh Back-Projection:** Once the depth map is obtained, the next step is to back-project the image into a 3D triangular mesh within the world space. This involves converting the depth map into a set of 3D points that represent the scene geometry. In the provided code, this step is performed by the createObj method within the ```BoundaryDepthExtractor``` class, which generates a 3D object file (model.obj) from the depth map.
   
3. **Vertical Plane Extraction:** For efficiency during training, the code focuses only on vertical planes. This reduces the 3D boundary extraction problem to a simpler 2D problem. The ```verticalPlaneExtraction``` method in the ```BoundaryDepthExtractor``` class is responsible for this step, although the actual implementation is not provided in the code snippet.
   
4. **Orthographic Projection:** The 3D mesh of the scene is projected onto a horizontal plane using orthographic projection. This projection facilitates the precise delineation of the 2D boundary that encapsulates the scene. The ```orthogonalProjection``` method in the ```BoundaryDepthExtractor``` class performs this step by projecting the 3D points onto a vertical plane.
   
5. **2D Boundary Delineation:** After projection, the next step is to delineate the 2D boundary that encapsulates the scene. This is achieved by determining the convex hull of the projected points, which represents the outer boundary of the scene. The ```boundaryDelineation``` method in the ```BoundaryDepthExtractor``` class performs this step.
   
6. **Polygon Approximation:** The 2D boundary is then approximated with a polygon to simplify the representation. This approximation is done using the Douglas-Peucker algorithm, which reduces the number of points in the boundary while maintaining its overall shape. The ```polygonApproximation``` method in the ```BoundaryDepthExtractor``` class performs this step.

## Input:
<img width="500" alt="Screenshot 2024-03-07 at 10 54 16 PM" src="https://github.com/ritessshhh/BoundaryDepth/assets/81812754/15230bb3-9046-46d1-b18a-387e770e12df">

## Polygon Approximation:

<img width="500" alt="Screenshot 2024-03-15 at 8 33 01 PM" src="https://github.com/ritessshhh/BoundaryDepth/assets/81812754/33c1305c-ece7-42c5-a67a-9071be5a23ab">


