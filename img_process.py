import cv2
import numpy as np

from sklearn.cluster import DBSCAN

from io import BytesIO
import csv

class OMRProcessor:
    def __init__(self):
        # Class variables to store the results
        self.flagged_papers = []
        self.warped_image_with_dots = None
        self.warped_original_image = None
        self.dot_coordinates = []

    def _four_point_transform(self, image, pts):
        """Applies a perspective transform to an image given four points."""
        # Obtain a consistent order of the points and unpack them
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
        
        margin_factor = 0.1
        margin_x = int(maxWidth * margin_factor)
        margin_y = int(maxHeight * margin_factor)
        
        dst = np.array([
            [margin_x, margin_y],
            [maxWidth - 1 - margin_x, margin_y],
            [maxWidth - 1 - margin_x, maxHeight - 1 - margin_y],
            [margin_x, maxHeight - 1 - margin_y]], dtype="float32")
            
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped, M

    def _rotate_to_correct_orientation(self, image):
        """Rotates the image to the correct orientation based on content analysis."""
        # This function is a placeholder. A robust implementation would involve
        # OCR or other content analysis to determine the correct orientation.
        return image

    def process_image(self, image_obj, file_name):
        # Read the CSV data and convert to a NumPy array
        with BytesIO(image_obj.getvalue()) as f:
            reader = csv.reader(f)
            data = list(reader)
        
        # Assume the CSV represents a grayscale image for simplicity
        image_array = np.array(data, dtype=np.uint8)
        image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

        # 2. Convert to grayscale and blur
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # 3. Apply Adaptive Thresholding
        bw_image = cv2.adaptiveThreshold(
            blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5
        )
        
        # 4. Find all contours
        contours, _ = cv2.findContours(
            bw_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 5. Filter for OMR Bubbles
        bubble_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            (x, y, w, h) = cv2.boundingRect(c)
            if h == 0: continue
            aspect_ratio = w / float(h)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 250 < area < 1500 and 0.8 < aspect_ratio < 1.2 and circularity > 0.8:
                bubble_contours.append(c)
        
        # 6. Flag if too few bubbles
        if len(bubble_contours) < 385:
            self.flagged_papers.append(file_name)
            return
        
        # 7. Calculate centers and remove stray points
        centers = []
        for c in bubble_contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))
        
        if not centers: return

        centers = np.array(centers)
        db = DBSCAN(eps=50, min_samples=4).fit(centers)
        labels = db.labels_
        non_stray_centers = centers[labels != -1] if -1 in labels else centers
        
        if len(non_stray_centers) < 4: return

        # 8. Find and sort dot coordinates
        self.dot_coordinates = sorted(non_stray_centers.tolist(), key=lambda p: (p[1], p[0]))
        
        # 9. Find corner points for warp
        x_coords = non_stray_centers[:, 0]
        y_coords = non_stray_centers[:, 1]
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        
        corners = []
        corners.append(min(non_stray_centers, key=lambda p: (p[0] - min_x)**2 + (p[1] - min_y)**2))
        corners.append(min(non_stray_centers, key=lambda p: (p[0] - max_x)**2 + (p[1] - min_y)**2))
        corners.append(min(non_stray_centers, key=lambda p: (p[0] - min_x)**2 + (p[1] - max_y)**2))
        corners.append(min(non_stray_centers, key=lambda p: (p[0] - max_x)**2 + (p[1] - max_y)**2))
        
        final_corners = [
            min(corners, key=lambda p: p[0] + p[1]),
            min(corners, key=lambda p: p[1] - p[0]),
            max(corners, key=lambda p: p[1] - p[0]),
            max(corners, key=lambda p: p[0] + p[1])
        ]
        
        # 10. Warp the original image
        self.warped_original_image, M = self._four_point_transform(image, np.float32(final_corners))

        # 11. Transform and draw dots on the warped image
        non_stray_centers_warped = cv2.perspectiveTransform(np.float32([non_stray_centers]), M)[0]
        self.warped_image_with_dots = self.warped_original_image.copy()

        for center in non_stray_centers_warped:
            cv2.circle(self.warped_image_with_dots, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

        # 12. Re-identify and color corner points on the warped image
        x_coords_warped = non_stray_centers_warped[:, 0]
        y_coords_warped = non_stray_centers_warped[:, 1]
        min_x_w, max_x_w = np.min(x_coords_warped), np.max(x_coords_warped)
        min_y_w, max_y_w = np.min(y_coords_warped), np.max(y_coords_warped)

        corners_warped = [
            min(non_stray_centers_warped, key=lambda p: (p[0] - min_x_w)**2 + (p[1] - min_y_w)**2),
            min(non_stray_centers_warped, key=lambda p: (p[0] - max_x_w)**2 + (p[1] - min_y_w)**2),
            min(non_stray_centers_warped, key=lambda p: (p[0] - min_x_w)**2 + (p[1] - max_y_w)**2),
            min(non_stray_centers_warped, key=lambda p: (p[0] - max_x_w)**2 + (p[1] - max_y_w)**2)
        ]

        for corner in corners_warped:
            cv2.circle(self.warped_image_with_dots, (int(corner[0]), int(corner[1])), 8, (255, 0, 0), -1)
            
        # 13. Rotate both warped images
        self.warped_image_with_dots = self._rotate_to_correct_orientation(self.warped_image_with_dots)
        self.warped_original_image = self._rotate_to_correct_orientation(self.warped_original_image)
        
        # Convert final images to CSV format and store in class variables
        self.warped_image_with_dots = self._image_to_csv(self.warped_image_with_dots)
        self.warped_original_image = self._image_to_csv(self.warped_original_image)
        
    def _image_to_csv(self, image):
        # Convert a NumPy array to a CSV object
        csv_file = BytesIO()
        writer = csv.writer(csv_file)
        writer.writerows(image.tolist())
        return csv_file
