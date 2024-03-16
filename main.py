import json
from DepthEstimator import BoundaryDepthExtractor
import open3d as o3d
import numpy as np

def saveVerticesToJson(data, filename="vertices.json"):
    """Save the vertices and camera data to a JSON file."""
    # Convert any ndarray objects to lists
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data[key] = value.tolist()

    # Flatten the vertices list
    data['vertices'] = [vertex[0] for vertex in data['vertices']]

    with open(filename, "w") as outfile:
        json.dump(data, outfile, indent=4)

if __name__ == "__main__":
    boundaryDepthExtractor = BoundaryDepthExtractor(
        depth_model_checkpoint='Intel/dpt-hybrid-midas'
    )

    image_path = 'images/bedroom01.jpg'
    # boundaryDepthExtractor.extractBoundaryDepth(image_path, filename="model.pcd")
    pcd = o3d.io.read_point_cloud('model.pcd')

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
    points = boundaryDepthExtractor.verticalPlaneExtraction("model.pcd")

    # boundaryDepthExtractor.visualizeVerticalPlaneExtraction(points)

    points_2d = boundaryDepthExtractor.orthogonalProjection(points)
    # boundaryDepthExtractor.visualizeOrthogonicProjection(points_2d)

    hull_points = boundaryDepthExtractor.boundaryDelineation(points_2d)
    # boundaryDepthExtractor.visualizeBoundaryDelineation(hull_points)

    vertices = boundaryDepthExtractor.polygonApproximation(hull_points)
    boundaryDepthExtractor.visualizePolygonApproximation(hull_points, vertices)

    # Save the vertices to a JSON file
    JSON = {}
    camera = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    camera.intrinsic_matrix
    JSON['camera'] = camera.intrinsic_matrix
    JSON['vertices'] = vertices.tolist()
    saveVerticesToJson(JSON, filename="verticesAndCamera.json")
