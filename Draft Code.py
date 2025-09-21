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

# IMPORTANT: You must provide the code for the OMRProcessor class
# I am creating a placeholder class here to allow the app to run.
# You need to implement the real logic based on your `img_process.py` file.
class OMRProcessor:
    def __init__(self):
        self.flagged_papers = []
        self.dot_coordinates = []

    def process_image(self, image_array, filename):
        # The original error was likely here. Do NOT use .getvalue() on image_array.
        # This function should use the numpy array 'image_array' for processing.
        # Placeholder logic:
        # Check if a specific condition in your processing flags the paper
        if "bad_scan" in filename:
            self.flagged_papers.append(filename)
        else:
            # Placeholder for found coordinates
            self.dot_coordinates = [(10, 10), (10, 20), (10, 30), (10, 40),
                                    (20, 10), (20, 20), (20, 30), (20, 40)]
        return True

def get_answer_key(exam_set_letter):
    """
    Loads the answer key from a CSV file based on the exam set letter.
    """
    try:
        file_path = f"Key (Set A and B).xlsx - Set - {exam_set_letter.upper()}.csv"
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
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
    except FileNotFoundError:
        st.error(f"Answer key for Set {exam_set_letter.upper()} not found.")
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
    st.experimental_rerun()
if st.sidebar.button("Enter Answer Key"):
    st.session_state.page = "Enter Answer Key"
    st.experimental_rerun()
if st.sidebar.button("Reports"):
    st.session_state.page = "Reports"
    st.experimental_rerun()
if st.sidebar.button("Export Data"):
    st.session_state.page = "Export Data"
    st.experimental_rerun()
if st.sidebar.button("Check flagged papers"):
    st.session_state.page = "Check flagged papers"
    st.experimental_rerun()

# --- Page Content based on Selection ---
if st.session_state.page == "Enter Answer Key":
    st.title("Enter Answer Key")
    set_letter = st.text_input("Exam Set Letter", max_chars=1)
    key_file = st.file_uploader("Upload an answer key as a csv", type=["csv"])
    if key_file and set_letter:
        with open(f"Key (Set A and B).xlsx - Set - {set_letter.upper()}.csv", "wb") as f:
            f.write(key_file.getvalue())
        st.success(f"Answer key for Set {set_letter.upper()} saved successfully!")


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
            omr_processor = OMRProcessor()
            # The uploaded file is a file-like object; the `scan` variable is a numpy array.
            # You must ensure your `OMRProcessor` correctly handles this.
            omr_processor.process_image(scan, uploaded_file.name)
            
            if uploaded_file.name in omr_processor.flagged_papers:
                st.warning("This paper has been flagged for manual review.")
                st.session_state.flagged_papers.append(uploaded_file.name)
            else:
                st.success("Paper processed successfully!")
                
                # Get the answer key
                answer_key = get_answer_key(exam_set_letter)
                if not answer_key:
                    pass

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
                data_row = [
                    student_name, exam_set_letter.upper(),
                    subject_marks["PYTHON"], subject_marks["DATA ANALYSIS"],
                    subject_marks["MY SQL"], subject_marks["POWER BI"],
                    subject_marks["ADV STATS"], total_marks, f"{percentage:.2f}%"
                ]
                
                save_data(data_row)
                
                st.write("### Results")
                st.write(f"**Total Marks:** {total_marks}/{len(answer_key)}")
                st.write(f"**Percentage:** {percentage:.2f}%")
                st.dataframe(pd.DataFrame([data_row], columns=["Name", "Paper Set", "PYTHON", "DATA ANALYSIS", "MY SQL", "POWER BI", "ADV STATS", "TOTAL", "PERCENTAGE"]))
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("Paper flagged for manual review.")
            st.session_state.flagged_papers.append(uploaded_file.name)


elif st.session_state.page == "Reports":
    st.title("📈 Reports")
    st.write("View analytical reports and visualizations here.")
    st.write("---")

    try:
        df = pd.read_csv("DATA.CSV", sep='\t')
        st.dataframe(df)
        
        st.subheader("Subject-wise Performance")
        subject_cols = ["PYTHON", "DATA ANALYSIS", "MY SQL", "POWER BI", "ADV STATS"]
        subject_df = df[subject_cols].mean().reset_index()
        subject_df.columns = ['Subject', 'Average Marks']
        st.bar_chart(subject_df.set_index('Subject'))
        
    except FileNotFoundError:
        st.info("No data available yet. Please submit some papers first.")
    except Exception as e:
        st.error(f"Could not load data for reports: {e}")


elif st.session_state.page == "Export Data":
    st.title("💾 Export Data")
    st.write("Download your data in various formats (e.g., CSV, JSON).")
    st.write("---")
    
    try:
        with open("DATA.CSV", "r") as f:
            csv_data = f.read()
            st.download_button(
                label="Download data as CSV",
                data=csv_data,
                file_name="data.csv",
                mime="text/csv",
            )
    except FileNotFoundError:
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
                    st.experimental_rerun()
    else:
        st.info("No papers have been flagged for review.")

