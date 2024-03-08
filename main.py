# Example usage
from IPython.display import display
from DepthEstimator import BoundaryDepthExtractor
from PIL import Image

# Assuming boundary_depth_map is correctly generated above
if __name__ == "__main__":
    boundary_depth_extractor = BoundaryDepthExtractor(
        depth_model_checkpoint='Intel/dpt-hybrid-midas',
        controlnet_checkpoint='shariqfarooq123/LooseControl',
        sd_checkpoint='runwayml/stable-diffusion-v1-5'
    )

    image_path = 'bedroom01.jpg'
    boundary_depth_map = boundary_depth_extractor.extract_boundary_depth(image_path)

    # Convert to PIL Image and display
    img = Image.fromarray(boundary_depth_map)
    img.show()
