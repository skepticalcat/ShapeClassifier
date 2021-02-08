import os
import cv2
import pickle
import numpy as np

def raw_shapes_to_edges():
    data = []
    for elem in os.listdir("H:/shapes/output"):
        path_orig = os.path.join("H:/shapes/output",elem)
        img = cv2.imread(path_orig, cv2.IMREAD_COLOR)
        edges = cv2.Canny(img, 100, 200)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        #cv2.imwrite("test.png",edges)
        data.append((elem.split("_")[0].lower(),edges))
    return data

if __name__ == "__main__":
    data = raw_shapes_to_edges()
    with open("data.pickle", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)