import os
import cv2

def raw_shapes_to_edges():
    for elem in os.listdir("H:/shapes/output"):
        path_orig = os.path.join("H:/shapes/output",elem)
        path_processed = os.path.join("H:/shapes/processed",elem)
        img = cv2.imread(path_orig, cv2.IMREAD_COLOR)
        edges = cv2.Canny(img, 100, 200)
        cv2.imwrite(path_processed,edges)

if __name__ == "__main__":
    raw_shapes_to_edges()