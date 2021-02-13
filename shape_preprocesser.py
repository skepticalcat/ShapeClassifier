import os
import cv2
import pickle
import numpy as np
import sys

def raw_shapes_to_edges(path):
    data = []
    for elem in os.listdir(path):
        path_orig = os.path.join(path,elem)
        img = cv2.imread(path_orig, cv2.IMREAD_COLOR)
        edges = cv2.Canny(img, 100, 200)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        data.append((elem.split("_")[0].lower(),edges))
    return data

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python shape_preprocessor.py path/to/folder/containing/images")
        exit(-1)

    path = sys.argv[1]
    data = raw_shapes_to_edges(path)
    with open("data.pickle", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)