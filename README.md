# INNOMATICS-HACKATHON-SUBMISSION
INNOMATICS HACKATHON SUBMISSION THEME 1  - OMR EVALUATION SYSTEM
OMR Grading and Data Management Application
This application is a Streamlit-based tool for automatically grading OMR (Optical Mark Recognition) sheets and managing student performance data. It uses computer vision techniques with OpenCV to process scanned OMR sheets, identify filled-in bubbles, and score the papers against a provided answer key.

Key Features
Automated Grading: Processes uploaded OMR sheet images to automatically detect and grade answers.

Answer Key Management: Allows users to upload and save answer keys for different exam sets in CSV or XLSX format.

Data Management: Stores student performance data, including marks for individual subjects and overall scores.

Reporting: Generates analytical reports and visualizations of student performance.

Error Handling: Flags papers that cannot be processed automatically for manual review.

Waitlist System: Manages papers that are submitted before their corresponding answer keys are available, processing them automatically once the key is uploaded.

How It Works
The core of the application is the OMRProcessor class, which handles the image processing pipeline:

Image Preprocessing: The uploaded image is converted to grayscale, and contrast is enhanced using CLAHE to improve bubble detection.

Contour Detection: The application finds circular contours that represent the OMR bubbles.

Perspective Transform: Using the corner points of the OMR sheet, a perspective transform is applied to correct for any skew or distortion in the scanned image.

Answer Detection: The average pixel intensity of each bubble is measured. If a bubble's intensity is below a certain darkness threshold compared to the surrounding area, it is identified as a filled-in answer.

Grading: The detected answers are compared against the uploaded answer key to calculate the student's score.

Installation and Usage
To run this application, you need to have Python and the required libraries installed.

Prerequisites
Python 3.7+

Streamlit

OpenCV

NumPy

pandas

scikit-learn

imutils

You can install the dependencies using pip:

pip install streamlit opencv-python numpy pandas scikit-learn imutils

Running the App
Save the code as app.py.

Run the application from your terminal:

streamlit run app.py

The application will open in your web browser. Follow the instructions on the screen to upload answer keys and student papers.

User Guide
The application is organized into several pages, accessible via the sidebar on the left.

1. Enter data
This is the main page for processing student papers.

Student's Name: Enter the name of the student.

Exam Set Letter: Input the letter corresponding to the exam set (e.g., 'A', 'B', 'C').

Upload a scanned image of the paper: Upload the OMR sheet image.

Rotate Buttons: If the image is not correctly oriented, use the Rotate Left and Rotate Right buttons to adjust it.

Submit: Click Submit to process the paper. The application will grade it, display the results, and add the data to the system.

2. Enter Answer Key
This page is for uploading the correct answers for each exam set.

Exam Set Letter: Enter the letter for the exam set (e.g., 'A').

Upload an answer key: Upload a CSV or XLSX file containing the answer key. The file should have question numbers and their correct answers.

Save Answer Key: Click this button to save the key. Any papers on the waitlist for this set will be automatically processed.

3. Reports
This page provides analytical insights into the performance data.

Dataframe: A table showing all the processed student data.

Subject-wise Performance: A bar chart visualizing the average marks in each subject.

4. Export Data
Use this page to download all the collected data.

Download data as CSV: Click this button to download a CSV file of all the student data.

5. Check flagged papers
This page lists papers that the system could not process automatically due to issues like poor scan quality.

You can review the list of flagged papers here.

6. Check waitlist
This page shows papers that have been submitted but are awaiting their corresponding answer key.

Once the answer key for a particular set is uploaded, the papers on this list will be automatically processed.

Project Structure
app.py: The main Streamlit application file, containing all the code for the UI, image processing, and data management logic.

DATA.CSV: (Will be created automatically) A file to store the processed student data.

Note on File Paths
The code is configured to look for sample images in a specific directory structure. You will need to update the file path in the OMRProcessor class to match the location of your images.

Credits
This application was developed as a solution for automated OMR sheet grading.
