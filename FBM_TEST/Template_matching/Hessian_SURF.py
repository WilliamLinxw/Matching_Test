import cv2
import numpy as np

img = cv2.imread('chessboard.jpg')

def fd(algorithm):
    if algorithm == 'SIFT':
        return cv2.xfeatures2d.SIFT_create()
    if algorithm == 'SURF':
        