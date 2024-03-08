import torch
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import DPTImageProcessor, DPTForDepthEstimation
import numpy as np
import os
import math
import open3d as o3d

# Assuming implementation for MeshProjector and LooseControlNet are available as described previously

class DepthEstimator:
    def __init__(self, model_name="Intel/dpt-hybrid-midas", device="cpu"):
        self.device = device
        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name).to(device)
        self.model.eval()

    def predict_depth(self, image):
        """Predict depth map from an image."""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

            # Interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],  # Width, Height of the original image
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()  # Remove batch dimension and transfer to CPU

        # Normalize the depth map for visualization or further processing
        normalized_depth_map = (prediction * 255 / np.max(prediction)).astype("uint8")

        return normalized_depth_map

class BoundaryDepthExtractor:
    def __init__(self, depth_model_checkpoint, controlnet_checkpoint, sd_checkpoint):
        # Initialize the Depth Estimator
        self.depth_estimator = DepthEstimator(model_name=depth_model_checkpoint)
        self.start = """# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {0}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {0}
DATA ascii
"""
    def vete(self,v, vt):
        if v == vt:
            return str(v)
        return str(v) + "/" + str(vt)

    def create_obj(self, img, objPath='model.obj', mtlPath='model.mtl', matName='colored', useMaterial=False):
        w = img.shape[1]
        h = img.shape[0]

        FOV = math.pi / 4
        D = (img.shape[0] / 2) / math.tan(FOV / 2)

        if max(objPath.find('\\'), objPath.find('/')) > -1:
            os.makedirs(os.path.dirname(mtlPath), exist_ok=True)

        with open(objPath, "w") as f:
            if useMaterial:
                f.write("mtllib " + mtlPath + "\n")
                f.write("usemtl " + matName + "\n")

            ids = np.zeros((img.shape[1], img.shape[0]), int)
            vid = 1

            all_x = []
            all_y = []
            all_z = []

            for u in range(0, w):
                for v in range(h - 1, -1, -1):

                    d = img[v, u]

                    ids[u, v] = vid
                    if d == 0.0:
                        ids[u, v] = 0
                    vid += 1

                    x = u - w / 2
                    y = v - h / 2
                    z = -D

                    norm = 1 / math.sqrt(x * x + y * y + z * z)

                    t = d / (z * norm)

                    x = -t * x * norm
                    y = t * y * norm
                    z = -t * z * norm

                    f.write("v " + str(x) + " " + str(y) + " " + str(z) + "\n")

            for u in range(0, img.shape[1]):
                for v in range(0, img.shape[0]):
                    f.write("vt " + str(u / img.shape[1]) +
                            " " + str(v / img.shape[0]) + "\n")

            for u in range(0, img.shape[1] - 1):
                for v in range(0, img.shape[0] - 1):

                    v1 = ids[u, v]
                    v3 = ids[u + 1, v]
                    v2 = ids[u, v + 1]
                    v4 = ids[u + 1, v + 1]

                    if v1 == 0 or v2 == 0 or v3 == 0 or v4 == 0:
                        continue

                    f.write("f " + self.vete(v1, v1) + " " +
                            self.vete(v2, v2) + " " + self.vete(v3, v3) + "\n")
                    f.write("f " + self.vete(v3, v3) + " " +
                            self.vete(v2, v2) + " " + self.vete(v4, v4) + "\n")

    def extract_boundary_depth(self, image_path):
        input_image = Image.open(image_path)
        depth_map = self.depth_estimator.predict_depth(input_image)
        # Convert to PIL Image and display
        img = Image.fromarray(depth_map)
        depth_array = np.array(img)

        # Invert the depth image
        max_depth = np.max(depth_array)
        min_depth = np.min(depth_array)
        inverted_depth_array = max_depth - depth_array + min_depth
        print("Creating the object....")
        self.create_obj(inverted_depth_array)
        print("Converting....")
        with open('model.obj', "r") as infile:
            obj = infile.read()
        points = []
        for line in obj.split("\n"):
            if (line != ""):
                line = line.split()
                if (line[0] == "v"):
                    point = [float(line[1]), float(line[2]), float(line[3])]
                    points.append(point)
        with open('model.pcd', "w") as outfile:
            outfile.write(self.start.format(len(points)))

            for point in points:
                outfile.write("{} {} {}\n".format(point[0], point[1], point[2]))

        # Load the PCD file
        pcd = o3d.io.read_point_cloud('model.pcd')

        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])
