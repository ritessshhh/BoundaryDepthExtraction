import json
from DepthEstimator import BoundaryDepthExtractor
import open3d as o3d

def saveVerticesToJson(vertices, filename="vertices.json"):
    """Save the vertices of the approximated polygon to a JSON file."""
    with open(filename, "w") as outfile:
        json.dump(vertices.tolist(), outfile)

if __name__ == "__main__":
    boundaryDepthExtractor = BoundaryDepthExtractor(
        depth_model_checkpoint='Intel/dpt-hybrid-midas'
    )

    image_path = 'images/teddy.jpg'
    boundaryDepthExtractor.extractBoundaryDepth(image_path, filename="model.pcd")
    # pcd = o3d.io.read_point_cloud('model.pcd')

    # Visualize the point cloud
    # o3d.visualization.draw_geometries([pcd])
    points = boundaryDepthExtractor.verticalPlaneExtraction("model.pcd")

    # boundaryDepthExtractor.visualizeVerticalPlaneExtraction(points)

    points_2d = boundaryDepthExtractor.orthogonicProjection(points)
    # boundaryDepthExtractor.visualizeOrthogonicProjection(points_2d)

    hull_points = boundaryDepthExtractor.boundaryDelineation(points_2d)
    # boundaryDepthExtractor.visualizeBoundaryDelineation(hull_points)

    vertices = boundaryDepthExtractor.polygonApproximation(hull_points)
    # boundaryDepthExtractor.visualizePolygonApproximation(hull_points, vertices)

    # Save the vertices to a JSON file
    saveVerticesToJson(vertices, filename="vertices.json")