import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN
import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
import itertools
import csv
import imutils
import math

class OMRProcessor:
    def __init__(self, darkness_factor=0.8):
        self.flagged_papers = []
        self.darkness_factor = darkness_factor

    def resize_for_display(self, img, height):
        h, w = img.shape[:2]
        if h == 0: return img
        scale = height / h
        return cv2.resize(img, (int(w * scale), height))

    def four_point_transform(self, image, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        margin = 100
        dst = np.array([
            [margin, margin],
            [maxWidth - 1 - margin, margin],
            [maxWidth - 1 - margin, maxHeight - 1 - margin],
            [margin, maxHeight - 1 - margin]], dtype="float32")
            
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped, M, dst

    def group_and_filter_coordinates(self, coords, tolerance=5, min_group_size=5):
        if not coords:
            return []
        coords.sort()
        groups = [[coords[0]]]
        for coord in coords[1:]:
            if abs(coord - groups[-1][-1]) <= tolerance:
                groups[-1].append(coord)
            else:
                groups.append([coord])
        filtered_groups = [group for group in groups if len(group) >= min_group_size]
        return [int(np.mean(group)) for group in filtered_groups]

    def preprocess_image(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray_image)
        blurred_image = cv2.GaussianBlur(clahe_image, (5, 5), 0)
        return blurred_image

    def process_image(self, image, filename):
        preprocessed_image = self.preprocess_image(image)
        bw_image = cv2.adaptiveThreshold(preprocessed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        contours, _ = cv2.findContours(bw_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bubble_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            (x, y, w, h) = cv2.boundingRect(c)
            if h
