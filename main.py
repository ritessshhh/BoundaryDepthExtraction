from DepthEstimator import BoundaryDepthExtractor

# Assuming boundary_depth_map is correctly generated above
if __name__ == "__main__":
    boundary_depth_extractor = BoundaryDepthExtractor(
        depth_model_checkpoint='Intel/dpt-hybrid-midas',
        controlnet_checkpoint='shariqfarooq123/LooseControl',
        sd_checkpoint='runwayml/stable-diffusion-v1-5'
    )

    image_path = 'images/empty_room.jpg'
    boundary_depth_extractor.extract_boundary_depth(image_path, filename="model.pcd")
    boundary_depth_extractor.polygon_approx("model.pcd")