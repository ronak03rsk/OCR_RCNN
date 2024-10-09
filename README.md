# OCR_RCNN
Steps to Run the Code
Install Required Packages: Make sure you have all the required libraries installed:

Copy code
pip install tensorflow keras pandas opencv-python openpyxl
Prepare Your Dataset: Ensure your images are in the images/ directory, and your Excel file is named output_dataset.xlsx with the appropriate columns.

Run crnn_ocr.py:

Open your terminal or command prompt.
Navigate to your project directory:


Copy code
cd C:\Users\272241\OCR_rcnn\OCR_RCNN\
Run the training script:

Copy code
python crnn_ocr.py
Run predict_ocr.py: After the model is trained, you can test it with new images by modifying the path in predict_ocr.py and running:
