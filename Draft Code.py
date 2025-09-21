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
            if h == 0: continue
            aspect_ratio = w / float(h)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 250 < area < 1500 and 0.8 < aspect_ratio < 1.2 and circularity > 0.8:
                bubble_contours.append(c)

        if len(bubble_contours) < 385:
            self.flagged_papers.append(filename)
            return None, None

        all_centers = []
        if bubble_contours:
            centers = []
            for c in bubble_contours:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centers.append((cX, cY))
            if centers:
                all_centers = np.array(centers)
                db = DBSCAN(eps=50, min_samples=4).fit(all_centers)
                labels = db.labels_

                if -1 in labels:
                    non_stray_centers = all_centers[labels != -1]
                else:
                    non_stray_centers = all_centers

                if len(non_stray_centers) > 0:
                    x_coords = [p[0] for p in non_stray_centers]
                    y_coords = [p[1] for p in non_stray_centers]
                    min_x, max_x = np.min(x_coords), np.max(x_coords)
                    min_y, max_y = np.min(y_coords), np.max(y_coords)
                    corners = [
                        min(non_stray_centers, key=lambda p: (p[0] - min_x)**2 + (p[1] - min_y)**2),
                        min(non_stray_centers, key=lambda p: (p[0] - max_x)**2 + (p[1] - min_y)**2),
                        min(non_stray_centers, key=lambda p: (p[0] - min_x)**2 + (p[1] - max_y)**2),
                        min(non_stray_centers, key=lambda p: (p[0] - max_x)**2 + (p[1] - max_y)**2)
                    ]
                    
                    corners_sorted = sorted(corners, key=lambda p: (p[0], p[1]))
                    top_points = sorted(corners_sorted[:2], key=lambda p: p[1])
                    bottom_points = sorted(corners_sorted[2:], key=lambda p: p[1], reverse=True)
                    final_corners = [top_points[0], top_points[1], bottom_points[0], bottom_points[1]]
                else:
                    self.flagged_papers.append(filename)
                    return None, None
            else:
                self.flagged_papers.append(filename)
                return None, None
        else:
            self.flagged_papers.append(filename)
            return None, None

        preprocessed_warped_image, M, _ = self.four_point_transform(preprocessed_image, np.float32(final_corners))
        bubble_points = np.array(all_centers, dtype='float32').reshape(-1, 1, 2)
        warped_points = cv2.perspectiveTransform(bubble_points, M).reshape(-1, 2)
        
        warped_x_coords = [p[0] for p in warped_points]
        warped_y_coords = [p[1] for p in warped_points]
        unique_x = self.group_and_filter_coordinates(warped_x_coords, tolerance=5, min_group_size=5)
        unique_y = self.group_and_filter_coordinates(warped_y_coords, tolerance=5, min_group_size=1)
        
        filled_bubbles_labels = {}
        # Define the starting numbers for each group of 4 columns
        start_numbers = [1, 21, 41, 61, 81]
        
        for row_idx, y_coord in enumerate(unique_y):
            for col_idx, x_coord in enumerate(unique_x):
                warped_x = int(x_coord)
                warped_y = int(y_coord)

                if 0 <= warped_x < preprocessed_warped_image.shape[1] and 0 <= warped_y < preprocessed_warped_image.shape[0]:
                    bubble_roi_size = 5
                    background_roi_size = 15

                    x_start_bubble = max(0, int(warped_x - bubble_roi_size))
                    y_start_bubble = max(0, int(warped_y - bubble_roi_size))
                    x_end_bubble = min(preprocessed_warped_image.shape[1], int(warped_x + bubble_roi_size))
                    y_end_bubble = min(preprocessed_warped_image.shape[0], int(warped_y + bubble_roi_size))

                    x_start_bg = max(0, int(warped_x - background_roi_size))
                    y_start_bg = max(0, int(warped_y - background_roi_size))
                    x_end_bg = min(preprocessed_warped_image.shape[1], int(warped_x + background_roi_size))
                    y_end_bg = min(preprocessed_warped_image.shape[0], int(warped_y + background_roi_size))
                        
                    if (x_end_bubble > x_start_bubble and y_end_bubble > y_start_bubble and
                        x_end_bg > x_start_bg and y_end_bg > y_start_bg):
                        
                        bubble_roi = preprocessed_warped_image[y_start_bubble:y_end_bubble, x_start_bubble:x_end_bubble]
                        bg_roi = preprocessed_warped_image[y_start_bg:y_end_bg, x_start_bg:x_end_bg]
                        
                        avg_intensity_bubble = np.mean(bubble_roi)
                        avg_intensity_bg = np.mean(bg_roi)
                        
                        if avg_intensity_bubble < avg_intensity_bg * self.darkness_factor:
                            q_group = col_idx // 4
                            q_num = start_numbers[q_group] + row_idx
                            option_letter = chr(ord('A') + (col_idx % 4))
                            
                            filled_bubbles_labels.setdefault(q_num, []).append(option_letter)

        return filled_bubbles_labels, self.flagged_papers

def get_answer_key(uploaded_file):
    """
    Loads the answer key from an uploaded CSV or XLSX file.
    """
    if uploaded_file is None:
        return None
        
    try:
        if uploaded_file.name.endswith('.csv'):
            string_data = uploaded_file.getvalue().decode("utf-8")
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
            return answer_key
        
        elif uploaded_file.name.endswith('.xlsx'):
            df_key = pd.read_excel(uploaded_file)
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
            return answer_key
            
    except Exception as e:
        st.error(f"Error processing answer key: {e}")
        return None

def save_data(data_row):
    file_exists = os.path.isfile('DATA.CSV')
    with open('DATA.CSV', 'a', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        if not file_exists:
            writer.writerow(["Name", "Paper Set", "PYTHON", "DATA ANALYSIS", "MY SQL", "POWER BI", "ADV STATS", "TOTAL", "PERCENTAGE"])
        writer.writerow(data_row)

if 'flagged_papers' not in st.session_state:
    st.session_state.flagged_papers = []
if 'waitlist' not in st.session_state:
    st.session_state.waitlist = []
if 'page' not in st.session_state:
    st.session_state.page = "Enter data"
if 'rotation_angle' not in st.session_state:
    st.session_state.rotation_angle = 0
if 'answer_keys' not in st.session_state:
    st.session_state.answer_keys = {}
if 'data_df' not in st.session_state:
    st.session_state.data_df = pd.DataFrame(columns=["Name", "Paper Set", "PYTHON", "DATA ANALYSIS", "MY SQL", "POWER BI", "ADV STATS", "TOTAL", "PERCENTAGE"])

st.set_page_config(
    page_title="Data Management App",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
if st.sidebar.button("Check waitlist"):
    st.session_state.page = "Check waitlist"
    st.rerun()

def process_and_grade_paper(student_name, exam_set_letter, uploaded_file):
    try:
        if uploaded_file is None:
            st.warning("Please upload a file.")
            return

        img = Image.open(uploaded_file)
        rotated_img = img.rotate(st.session_state.rotation_angle, expand=True)
        scan = np.array(rotated_img)

        omr_processor = OMRProcessor()
        filled_bubbles_labels, flagged_papers_list = omr_processor.process_image(scan, uploaded_file.name)

        if filled_bubbles_labels is None:
            st.warning("This paper has been flagged for manual review.")
            st.session_state.flagged_papers.append(uploaded_file.name)
            return

        if exam_set_letter.upper() not in st.session_state.answer_keys:
            st.info("Answer key not found. Paper added to waitlist.")
            st.session_state.waitlist.append({
                'name': student_name,
                'set': exam_set_letter.upper(),
                'file': uploaded_file.name,
                'image_data': uploaded_file.getvalue()
            })
        else:
            st.success("Paper processed successfully!")
            answer_key = st.session_state.answer_keys[exam_set_letter.upper()]
            student_answers = {}
            for q_num, options in filled_bubbles_labels.items():
                if len(options) == 1:
                    student_answers[q_num] = options[0].lower()
                else:
                    student_answers[q_num] = None

            total_marks = 0
            for q_num, correct_ans in answer_key.items():
                student_ans = student_answers.get(q_num)
                if student_ans is not None and student_ans == correct_ans:
                    total_marks += 1
            
            subject_marks = {
                "PYTHON": 0, "DATA ANALYSIS": 0, "MY SQL": 0,
                "POWER BI": 0, "ADV STATS": 0
            }
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

            st.session_state.data_df = pd.concat([st.session_state.data_df, pd.DataFrame([data_dict])], ignore_index=True)
            
            st.write("### Results")
            st.write(f"**Total Marks:** {total_marks}/{len(answer_key)}")
            st.write(f"**Percentage:** {percentage:.2f}%")
            st.dataframe(pd.DataFrame([data_dict], columns=st.session_state.data_df.columns))
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("Paper flagged for manual review.")
        st.session_state.flagged_papers.append(uploaded_file.name)

if st.session_state.page == "Enter Answer Key":
    st.title("Enter Answer Key")
    set_letter = st.text_input("Exam Set Letter", max_chars=1)
    key_file = st.file_uploader("Upload an answer key as a csv", type=["csv", "xlsx"])
    
    if st.button("Save Answer Key"):
        if key_file and set_letter:
            answer_key = get_answer_key(key_file)
            if answer_key:
                st.session_state.answer_keys[set_letter.upper()] = answer_key
                st.success(f"Answer key for Set {set_letter.upper()} saved successfully!")
                
                # Check waitlist for this set and process papers
                if any(item['set'] == set_letter.upper() for item in st.session_state.waitlist):
                    st.info(f"Processing papers from waitlist for Set {set_letter.upper()}...")
                    papers_to_process = [item for item in st.session_state.waitlist if item['set'] == set_letter.upper()]
                    for paper in papers_to_process:
                        uploaded_file = BytesIO(paper['image_data'])
                        uploaded_file.name = paper['file']
                        process_and_grade_paper(paper['name'], paper['set'], uploaded_file)
                        st.session_state.waitlist.remove(paper)
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
        
    if submit_button:
        process_and_grade_paper(student_name, exam_set_letter, uploaded_file)

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

elif st.session_state.page == "Check waitlist":
    st.title("⏳ Check Waitlist")
    st.write("These papers are waiting for their answer key to be uploaded.")
    st.write("---")
    if st.session_state.waitlist:
        waitlist_df = pd.DataFrame(st.session_state.waitlist).drop(columns=['image_data'])
        st.dataframe(waitlist_df)
        
        st.warning("Once the answer key for a set is uploaded, all papers for that set will be automatically processed.")
    else:
        st.info("The waitlist is currently empty.")
