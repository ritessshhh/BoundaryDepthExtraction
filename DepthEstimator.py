import torch
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import DPTImageProcessor, DPTForDepthEstimation

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

    def extract_boundary_depth(self, image_path):
        input_image = Image.open(image_path)
        depth_map = self.depth_estimator.predict_depth(input_image)
        return depth_map
        # mesh_projector = MeshProjector()
        # point_cloud = mesh_projector.depth_map_to_point_cloud(depth_map, intrinsic_matrix=[525, 525, depth_map.shape[1]/2, depth_map.shape[0]/2])
        #
        # planes, plane_models = mesh_projector.detect_planes_with_ransac(point_cloud)