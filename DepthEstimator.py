import torch
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
import numpy as np
import os
import math
import open3d as o3d
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import cv2

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


    def extract_boundary_depth(self, image_path, filename="model.pcd"):
        input_image = Image.open(image_path)
        depth_map = self.depth_estimator.predict_depth(input_image)
        print("Depth map shape:", depth_map.shape)
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
        with open(filename, "w") as outfile:
            outfile.write(self.start.format(len(points)))

            for point in points:
                outfile.write("{} {} {}\n".format(point[0], point[1], point[2]))

        os.remove("model.obj")

        # pcd = o3d.io.read_point_cloud(filename)
        # points_3d = np.asarray(pcd.points)
        # print("Number of 3D points:", len(points_3d))
        #
        # projected_points_2d = self.project_mesh_to_plane(points_3d)
        # print("Number of projected 2D points:", len(projected_points_2d))
        #
        # return projected_points_2d

    def polygon_approx(self, file):
        pcd = o3d.io.read_point_cloud(file)

        # Convert to numpy array
        points = np.asarray(pcd.points)

        # Orthographic projection (omit Z)
        points_2d = points[:, :2]

        # Determine the boundary of the points (convex hull)
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]

        # Invert the y-axis to correct upside down issue
        hull_points[:, 1] = -hull_points[:, 1]

        # Epsilon parameter for approximation accuracy (adjust as needed)
        epsilon = 0.01 * cv2.arcLength(hull_points.astype(np.float32), True)
        approx_polygon = cv2.approxPolyDP(hull_points.astype(np.float32), epsilon, True)

        # Plot the results
        plt.plot(hull_points[:, 0], hull_points[:, 1], 'k--', lw=1)  # Hull boundary in black dashed line
        plt.plot(approx_polygon[:, 0, 0], approx_polygon[:, 0, 1], 'b-', lw=2)  # Approximated polygon in blue line
        plt.fill(hull_points[:, 0], hull_points[:, 1], 'lightgray')  # Fill the convex hull for visualization

        plt.title('Polygon Approximation of 2D Orthographic Projection')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()  # Ensure y-axis is not inverted for visualization
        plt.show()


    def extrude_polygon_to_3d(self, polygon_vertices_2d, scene_height):
        # Create an empty list to store the 3D vertices
        vertices_3d = []

        # First, add the base vertices (at z=0)
        for vertex in polygon_vertices_2d:
            vertices_3d.append([vertex[0], vertex[1], 0])

        # Then, add the top vertices (at z=scene_height)
        for vertex in polygon_vertices_2d:
            vertices_3d.append([vertex[0], vertex[1], scene_height])

        # Convert vertices list to numpy array
        vertices_3d = np.array(vertices_3d)

        # Create an empty mesh
        mesh = o3d.geometry.TriangleMesh()

        # Set the vertices of the mesh
        mesh.vertices = o3d.utility.Vector3dVector(vertices_3d)

        # Prepare to create triangles for the sides of the mesh
        triangles = []
        num_base_vertices = len(polygon_vertices_2d)

        # Forming sides by connecting four points
        for i in range(num_base_vertices):
            # Indices of the base vertices
            base_vertex_index = i
            next_base_vertex_index = (i + 1) % num_base_vertices

            # Indices of the top vertices
            top_vertex_index = i + num_base_vertices
            next_top_vertex_index = (next_base_vertex_index + num_base_vertices) % (2 * num_base_vertices)

            # Create two triangles for the current side
            triangles.append([base_vertex_index, next_base_vertex_index, top_vertex_index])
            triangles.append([next_base_vertex_index, next_top_vertex_index, top_vertex_index])

        # Add triangles to the mesh
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))

        # Optionally, add top and bottom faces here

        # Compute normals for the mesh
        mesh.compute_vertex_normals()
        print("Number of vertices in the extruded mesh:", len(mesh.vertices))
        print("Number of triangles in the extruded mesh:", len(mesh.triangles))

        return mesh