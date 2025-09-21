import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN
import imutils
import math
import itertools
from io import BytesIO
import csv
from PIL import Image
import streamlit as st
import pandas as pd

# The OMRProcessor class from your img_process.py file, with fixes
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

    def process_image(self, image_array, file_name):
        # CORRECTED: This function now directly accepts a NumPy array `image_array`
        # and no longer expects a file object with a .getvalue() method.
        if len(image_array.shape) == 3:
            # Convert to grayscale for processing
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image_array
            
        # 2. Convert to grayscale and blur
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
        # Note: The input to `_four_point_transform` needs to be the original image array.
        self.warped_original_image, M = self._four_point_transform(image_array, np.float32(final_corners))

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
        # This method is not used in the main app logic and may cause memory issues with large images.
        # It's better to convert the image to a format like PNG or JPEG for download.
        csv_file = BytesIO()
        writer = csv.writer(csv_file)
        writer.writerows(image.tolist())
        return csv_file

def get_answer_key(uploaded_file):
    """
    Loads the answer key from an uploaded CSV file.
    """
    if uploaded_file is None:
        return None
        
    try:
        # Read the uploaded file as a string
        string_data = uploaded_file.getvalue().decode("utf-8")
        reader = csv.reader(string_data.splitlines())
        
        # Skip header and blank rows
        rows = [row for row in reader if row and 'Python' not in row and 'Pyt' not in row]
        
        answer_key = {}
        for row in rows:
            for col in row:
                col = col.strip()
                if ' - ' in col:
                    parts = col.split(' - ')
                    question_number = int(parts[0])
                    answer_letter = parts[1].strip().lower()
                    answer_key[question_number] = answer_letter
                elif '. ' in col:
                     parts = col.split('. ')
                     question_number = int(parts[0])
                     answer_letter = parts[1].strip().lower()
                     answer_key[question_number] = answer_letter
        return answer_key
    except Exception as e:
        st.error(f"Error processing answer key: {e}")
        return None

def find_closest_dot(point, dots):
    """
    Finds the closest dot to a given point.
    """
    min_dist = float('inf')
    closest_dot = None
    for dot in dots:
        dist = np.sqrt((point[0] - dot[0])**2 + (point[1] - dot[1])**2)
        if dist < min_dist:
            min_dist = dist
            closest_dot = dot
    return closest_dot

def save_data(data_row):
    """
    Appends the student's data to the DATA.CSV file.
    """
    file_exists = os.path.isfile('DATA.CSV')
    with open('DATA.CSV', 'a', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        if not file_exists:
            writer.writerow(["Name", "Paper Set", "PYTHON", "DATA ANALYSIS", "MY SQL", "POWER BI", "ADV STATS", "TOTAL", "PERCENTAGE"])
        writer.writerow(data_row)

# Initialize flagged papers list and page state
if 'flagged_papers' not in st.session_state:
    st.session_state.flagged_papers = []
if 'page' not in st.session_state:
    st.session_state.page = "Enter data"
if 'rotation_angle' not in st.session_state:
    st.session_state.rotation_angle = 0
if 'answer_keys' not in st.session_state:
    st.session_state.answer_keys = {}
if 'data_df' not in st.session_state:
    st.session_state.data_df = pd.DataFrame(columns=["Name", "Paper Set", "PYTHON", "DATA ANALYSIS", "MY SQL", "POWER BI", "ADV STATS", "TOTAL", "PERCENTAGE"])


# Set the page configuration to a wide layout.
st.set_page_config(
    page_title="Data Management App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
if st.sidebar.button("Enter data"):
    st.session_state.page = "Enter data"
    st.rerun()
if st.sidebar.button("Enter Answer Key"):
    st.session_state.page = "Enter Answer Key"
    st.rerun()
if st.sidebar.button("Reports"):
    st.session_state.page = "Reports"
    st.rerun()
if st.sidebar.button("Export Data"):
    st.session_state.page = "Export Data"
    st.rerun()
if st.sidebar.button("Check flagged papers"):
    st.session_state.page = "Check flagged papers"
    st.rerun()

# --- Page Content based on Selection ---
if st.session_state.page == "Enter Answer Key":
    st.title("Enter Answer Key")
    set_letter = st.text_input("Exam Set Letter", max_chars=1)
    key_file = st.file_uploader("Upload an answer key as a csv", type=["csv", "xlsx"])
    
    if st.button("Save Answer Key"):
        if key_file and set_letter:
            try:
                if key_file.name.endswith('.csv'):
                    # Process CSV file
                    string_data = key_file.getvalue().decode("utf-8")
                    reader = csv.reader(string_data.splitlines())
                    
                    rows = [row for row in reader if row and 'Python' not in row and 'Pyt' not in row]
                    
                    answer_key = {}
                    for row in rows:
                        for col in row:
                            col = col.strip()
                            if ' - ' in col:
                                parts = col.split(' - ')
                                question_number = int(parts[0])
                                answer_letter = parts[1].strip().lower()
                                answer_key[question_number] = answer_letter
                            elif '. ' in col:
                                parts = col.split('. ')
                                question_number = int(parts[0])
                                answer_letter = parts[1].strip().lower()
                                answer_key[question_number] = answer_letter
                
                elif key_file.name.endswith('.xlsx'):
                    # Process XLSX file
                    df_key = pd.read_excel(key_file)
                    answer_key = {}
                    for _, row in df_key.iterrows():
                        for col in df_key.columns:
                            value = str(row[col]).strip()
                            if ' - ' in value:
                                parts = value.split(' - ')
                                question_number = int(parts[0])
                                answer_letter = parts[1].strip().lower()
                                answer_key[question_number] = answer_letter
                            elif '. ' in value:
                                parts = value.split('. ')
                                question_number = int(parts[0])
                                answer_letter = parts[1].strip().lower()
                                answer_key[question_number] = answer_letter
                
                # Store the key in session state
                st.session_state.answer_keys[set_letter.upper()] = answer_key
                st.success(f"Answer key for Set {set_letter.upper()} saved successfully!")
            except Exception as e:
                st.error(f"Error processing answer key: {e}")
        else:
            st.warning("Please provide both a set letter and an answer key file.")


elif st.session_state.page == "Enter data":
    st.title("Enter data")
    with st.form(key='data_entry_form'):
        student_name = st.text_input("Student's Name")
        exam_set_letter = st.text_input("Exam Set Letter", max_chars=1)
        uploaded_file = st.file_uploader("Upload a scanned image of the paper", type=["png", "jpg", "jpeg"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            rotate_left = st.form_submit_button(label='Rotate Left')
        with col2:
            rotate_right = st.form_submit_button(label='Rotate Right')

        submit_button = st.form_submit_button(label='Submit')

    if uploaded_file:
        if rotate_left:
            st.session_state.rotation_angle -= 90
        if rotate_right:
            st.session_state.rotation_angle += 90

        img = Image.open(uploaded_file)
        rotated_img = img.rotate(st.session_state.rotation_angle, expand=True)
        st.image(rotated_img, caption='Uploaded Image')
        scan = np.array(rotated_img)

    if submit_button and uploaded_file:
        try:
            if exam_set_letter.upper() not in st.session_state.answer_keys:
                st.error(f"Answer key for Set {exam_set_letter.upper()} not found. Please upload it first.")
                st.warning("Paper flagged for manual review.")
                st.session_state.flagged_papers.append(uploaded_file.name)
                st.rerun()
                
            omr_processor = OMRProcessor()
            omr_processor.process_image(scan, uploaded_file.name)
            
            if uploaded_file.name in omr_processor.flagged_papers:
                st.warning("This paper has been flagged for manual review.")
                st.session_state.flagged_papers.append(uploaded_file.name)
            else:
                st.success("Paper processed successfully!")
                
                # Get the answer key from session state
                answer_key = st.session_state.answer_keys[exam_set_letter.upper()]
                
                # Compartmentalize coordinates
                compartmentalized_coords = []
                coords = omr_processor.dot_coordinates
                
                # Sort the coordinates to group by question
                coords.sort(key=lambda p: (p[1], p[0]))

                # Divide into groups of 4 (for A, B, C, D)
                for i in range(0, len(coords), 4):
                    compartmentalized_coords.append(coords[i:i+4])

                student_answers = {}
                for i, group in enumerate(compartmentalized_coords):
                    q_num = i + 1
                    filled_in_options = []
                    
                    for j, dot_coord in enumerate(group):
                        x, y = dot_coord
                        pixel_value = scan[y, x] # Assumes grayscale
                        if np.mean(pixel_value) < 100:
                                filled_in_options.append(j)

                    if len(filled_in_options) == 1:
                        student_answers[q_num] = chr(ord('a') + filled_in_options[0])
                    else:
                        student_answers[q_num] = None

                # Grade the answers
                total_marks = 0
                for q_num, correct_ans in answer_key.items():
                    student_ans = student_answers.get(q_num)
                    if student_ans is not None and student_ans == correct_ans:
                        total_marks += 1
                
                # Prepare data row for CSV
                subject_marks = {
                    "PYTHON": 0, "DATA ANALYSIS": 0, "MY SQL": 0,
                    "POWER BI": 0, "ADV STATS": 0
                }
                
                # Logic to map questions to subjects
                python_q = range(1, 21)
                data_q = range(21, 41)
                sql_q = range(41, 61)
                bi_q = range(61, 81)
                stats_q = range(81, 101)
                
                for q_num, student_ans in student_answers.items():
                    correct_ans = answer_key.get(q_num)
                    if student_ans is not None and student_ans == correct_ans:
                        if q_num in python_q: subject_marks["PYTHON"] += 1
                        elif q_num in data_q: subject_marks["DATA ANALYSIS"] += 1
                        elif q_num in sql_q: subject_marks["MY SQL"] += 1
                        elif q_num in bi_q: subject_marks["POWER BI"] += 1
                        elif q_num in stats_q: subject_marks["ADV STATS"] += 1
                
                percentage = (total_marks / len(answer_key)) * 100
                
                data_dict = {
                    "Name": student_name,
                    "Paper Set": exam_set_letter.upper(),
                    "PYTHON": subject_marks["PYTHON"],
                    "DATA ANALYSIS": subject_marks["DATA ANALYSIS"],
                    "MY SQL": subject_marks["MY SQL"],
                    "POWER BI": subject_marks["POWER BI"],
                    "ADV STATS": subject_marks["ADV STATS"],
                    "TOTAL": total_marks,
                    "PERCENTAGE": f"{percentage:.2f}%"
                }

                # Append data to the DataFrame in session state
                st.session_state.data_df = pd.concat([st.session_state.data_df, pd.DataFrame([data_dict])], ignore_index=True)
                
                st.write("### Results")
                st.write(f"**Total Marks:** {total_marks}/{len(answer_key)}")
                st.write(f"**Percentage:** {percentage:.2f}%")
                st.dataframe(pd.DataFrame([data_dict], columns=st.session_state.data_df.columns))
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("Paper flagged for manual review.")
            st.session_state.flagged_papers.append(uploaded_file.name)


elif st.session_state.page == "Reports":
    st.title("📈 Reports")
    st.write("View analytical reports and visualizations here.")
    st.write("---")

    if not st.session_state.data_df.empty:
        st.dataframe(st.session_state.data_df)
        
        st.subheader("Subject-wise Performance")
        subject_cols = ["PYTHON", "DATA ANALYSIS", "MY SQL", "POWER BI", "ADV STATS"]
        subject_df = st.session_state.data_df[subject_cols].mean().reset_index()
        subject_df.columns = ['Subject', 'Average Marks']
        st.bar_chart(subject_df.set_index('Subject'))
    else:
        st.info("No data available yet. Please submit some papers first.")


elif st.session_state.page == "Export Data":
    st.title("💾 Export Data")
    st.write("Download your data in various formats (e.g., CSV, JSON).")
    st.write("---")
    
    if not st.session_state.data_df.empty:
        # Convert DataFrame to CSV string for download
        csv_data = st.session_state.data_df.to_csv(index=False)
        st.download_button(
            label="Download data as CSV",
            data=csv_data,
            file_name="data.csv",
            mime="text/csv",
        )
    else:
        st.info("No data available to export.")


elif st.session_state.page == "Check flagged papers":
    st.title("🚨 Check Flagged Papers")
    st.write("Review and manage papers that have been flagged for review.")
    st.write("---")
    
    if st.session_state.flagged_papers:
        st.write("The following papers have been flagged:")
        
        # Display list of flagged papers with a removal button
        for paper in st.session_state.flagged_papers:
            col_paper, col_button = st.columns([0.8, 0.2])
            with col_paper:
                st.write(f"- {paper}")
            with col_button:
                if st.button("Remove", key=f"remove_btn_{paper}"):
                    st.session_state.flagged_papers.remove(paper)
                    st.rerun()
    else:
        st.info("No papers have been flagged for review.")
