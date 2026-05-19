"""Document Tools Exercise

In this exercise, we explore how to extract text from different document formats.

1. List all available documents in the configured document directory.
2. Extract text from a PDF document using pdfminer.
3. Extract text from an image file using RapidOCR.

"""

import time
from pathlib import Path
from pdfminer.high_level import extract_text
from rapidocr import RapidOCR

# Define file paths and constants
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
MY_DOCUMENTS = DATA_DIR / "my_documents"

#################################################################
# STEP 1 - List all available documents in the directory
#################################################################

# List all files in the MY_DOCUMENTS directory to see what documents are available for preprocessing.
available_documents = sorted(
    path.name
    for path in MY_DOCUMENTS.iterdir()
    if path.is_file() and not path.name.startswith(".")
)
print(f"Available documents: {available_documents}")

#################################################################
# STEP 2 - Extract PDF text
#################################################################

pdf_path = MY_DOCUMENTS / "Service_Agreement.pdf"

# Extract the text from the PDF using pdfminer (returns a single string with the full text)
start = time.time()

# SOLUTION - PDF extraction:
# pdfminer's extract_text reads the PDF file and returns its text as one string.
doc_pdfminer = extract_text(str(pdf_path))

end = time.time()

print(f"Using file: {pdf_path.name}")
print(
    f"🕒 pdfminer extracted {len(doc_pdfminer)} characters in {end - start:.2f} seconds"
)
print(f"Preview of first 500 characters:\n{doc_pdfminer[:500]}")

#################################################################
# STEP 3 - Extract text from image file
#################################################################

image_path = MY_DOCUMENTS / "Fondue_Recipe.png"

# Extract the text from the image using RapidOCR (returns a list of recognized text segments in result.txts)
start = time.time()

# SOLUTION - OCR engine:
# RapidOCR creates the callable engine used below to read text from the image.
engine = RapidOCR()

result = engine(str(image_path))
doc_ocr = "\n".join(result.txts or [])
end = time.time()

print(f"Using file: {image_path.name}")
print(f"🕒 RapidOCR extracted {len(doc_ocr)} characters in {end - start:.2f} seconds")
print(f"Preview of first 500 characters:\n{doc_ocr[:500]}")
