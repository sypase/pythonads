from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import os
import re
import shutil
from docx import Document
import PyPDF2
from typing import List

app = FastAPI(title="PDF Word Count and Redaction Service", description="A service to manage PDF files, including word count, redaction, and classification.")

# Middleware to allow CORS for testing on different environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/tmp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Define max file size (4MB to stay under Vercel's limit)
MAX_FILE_SIZE = 101 * 1024 * 1024  # 101MB in bytes

def count_words_in_pdf(pdf_path):
    """Counts words in a PDF, ignoring empty lines and comments."""
    try:
        doc = fitz.open(pdf_path)
        text = " ".join([page.get_text("text") for page in doc])
        # Remove comments (lines starting with # or //)
        text = re.sub(r"^\s*(#|//).*$", "", text, flags=re.MULTILINE)
        # Remove extra whitespace and count words
        words = re.findall(r"\b\w+\b", text)
        return len(words)
    except Exception as e:
        print(f"Error counting words in PDF: {e}")
        return None

def count_words_in_docx(docx_path):
    """Counts words in a DOCX file."""
    try:
        doc = Document(docx_path)
        text = " ".join([para.text for para in doc.paragraphs])
        words = re.findall(r"\b\w+\b", text)
        return len(words)
    except Exception as e:
        print(f"Error counting words in DOCX: {e}")
        return None

def count_words_in_text(text_path):
    """Counts words in a text file."""
    try:
        with open(text_path, "r") as file:
            text = file.read()
        words = re.findall(r"\b\w+\b", text)
        return len(words)
    except Exception as e:
        print(f"Error counting words in text file: {e}")
        return None

def validate_file_extension(file: UploadFile):
    """Validates the file extension to ensure it's one of the allowed types."""
    allowed_extensions = ['.pdf', '.docx', '.txt']
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF, DOCX, and TXT files are allowed.")
    return file_extension

@app.post("/count-words")
async def count_words_endpoint(file: UploadFile = File(...)):
    """Processes an uploaded file and counts the words in it."""
    # Check file size first before processing
    file_content = await file.read(1024)
    content_length = file.size if hasattr(file, 'size') else None

    # If we can't get size from headers, estimate based on first chunk
    if content_length is None:
        await file.seek(0)
        content = await file.read()
        content_length = len(content)
        await file.seek(0)
    else:
        await file.seek(0)

    if content_length > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE/1024/1024}MB."
        )

    file_extension = validate_file_extension(file)
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    word_count = None
    if file_extension == '.pdf':
        word_count = count_words_in_pdf(input_path)
    elif file_extension == '.docx':
        word_count = count_words_in_docx(input_path)
    elif file_extension == '.txt':
        word_count = count_words_in_text(input_path)

    if word_count is None:
        raise HTTPException(status_code=500, detail="Error processing the file.")

    response = {
        "total_words": word_count
    }

    return JSONResponse(content=response)

def redact_submission_ids(input_pdf, output_pdf):
    """Redacts Submission IDs and 'Document Details' from a PDF."""
    doc = fitz.open(input_pdf)

    for page_num, page in enumerate(doc):
        text_instances = page.search_for("Submission ID trn:oid:::")
        for inst in text_instances:
            rect = fitz.Rect(inst.x0, inst.y0, inst.x1 + 100, inst.y1)
            page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))

        if page_num == 0:
            details_instances = page.search_for("Document Details")
            for inst in details_instances:
                rect = fitz.Rect(0, inst.y0 - 50, page.rect.x1, inst.y0)
                page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))

    doc.save(output_pdf)

@app.post("/redact")
async def redact_pdf(file: UploadFile = File(...)):
    """Redacts sensitive information from an uploaded PDF and returns the file directly."""
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    output_path = os.path.join(UPLOAD_DIR, f"redacted_{file.filename}")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    redact_submission_ids(input_path, output_path)
    
    return FileResponse(output_path, media_type="application/pdf", filename=f"redacted_{file.filename}")

def extract_second_page_text(pdf_path):
    """Extracts text from the second page of a PDF."""
    try:
        doc = fitz.open(pdf_path)
        if len(doc) < 2:
            return None
        return doc[1].get_text("text")
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def classify_pdf(pdf_path):
    """Classifies a PDF based on plagiarism and AI detection patterns."""
    text = extract_second_page_text(pdf_path)
    if not text:
        return {"error": "PDF does not have a second page or could not be read."}

    similarity_match = re.search(r"(\d+)%\s*Overall Similarity", text)
    ai_match = re.search(r"(?:(\d+)\*?%|(\*%))(?:\s*detected as AI)", text)

    result = {
        "type": "Unknown",
        "Overall Similarity": None,
        "AI Detection": None,
        "AI Detection Asterisk": False,
        "Below_Threshold": False
    }

    if similarity_match and ai_match:
        result["type"] = "Plagiarism and AI Detection Report"
        result["Overall Similarity"] = int(similarity_match.group(1))
        if ai_match.group(1):
            result["AI Detection"] = int(ai_match.group(1))
            result["AI Detection Asterisk"] = '*' in ai_match.group(0)
        else:
            result["AI Detection"] = -1  # "<20" AI detection case
            result["Below_Threshold"] = True
            result["AI Detection Asterisk"] = True
    elif similarity_match:
        result["type"] = "Plagiarism Report"
        result["Overall Similarity"] = int(similarity_match.group(1))
    elif ai_match:
        result["type"] = "AI Detection Report"
        if ai_match.group(1):
            result["AI Detection"] = int(ai_match.group(1))
            result["AI Detection Asterisk"] = '*' in ai_match.group(0)
        else:
            result["AI Detection"] = -1
            result["Below_Threshold"] = True
            result["AI Detection Asterisk"] = True

    return result

@app.post("/classify")
async def classify_pdf_endpoint(file: UploadFile = File(...)):
    """Processes an uploaded PDF and classifies it based on AI detection and plagiarism patterns."""
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = classify_pdf(input_path)
    return JSONResponse(content=result)

@app.get("/")
async def read_root():
    """Root endpoint that confirms the service is running and provides route information."""
    return JSONResponse({
        "message": "PDF and Document Service is online. Access /docs for the API documentation.",
        "routes": [
            {"path": "/count-words", "description": "Count words in a document"},
            {"path": "/redact", "description": "Redact sensitive information from a PDF"},
            {"path": "/classify", "description": "Classify a PDF for AI and plagiarism detection"},
            {"path": "/status", "description": "Get status of processed files"}
        ]
    })

@app.get("/status")
async def get_status():
    """Returns a list of files and their status in the upload directory."""
    try:
        files = os.listdir(UPLOAD_DIR)
        if not files:
            return JSONResponse(status_code=200, content={"message": "No files found in the upload directory."})

        file_list = [{"filename": file, "status": "redacted" if file.startswith("redacted_") else "uploaded"} for file in files]
        return JSONResponse(status_code=200, content={"files": file_list})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def merge_pdfs(pdf_list: List[str], output_filename: str):
    """Merge multiple PDF files into a single PDF."""
    merger = PyPDF2.PdfMerger()
    try:
        for pdf in pdf_list:
            merger.append(pdf)
        merger.write(output_filename)
        merger.close()
        return True
    except Exception as e:
        print(f"Error merging PDFs: {e}")
        return False

def merge_docx(docx_list: List[str], output_filename: str):
    """Merge multiple DOCX files into a single DOCX."""
    try:
        merged_doc = Document()
        for index, file in enumerate(docx_list):
            doc = Document(file)
            for element in doc.element.body:
                merged_doc.element.body.append(element)
            
            if index != len(docx_list) - 1:
                merged_doc.add_page_break()
        
        merged_doc.save(output_filename)
        return True
    except Exception as e:
        print(f"Error merging DOCX files: {e}")
        return False


    """Merge multiple PDF or DOCX files into a single file."""
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="At least 2 files are required for merging.")
    
    # Check if all files are of the same type
    file_extensions = [os.path.splitext(file.filename)[1].lower() for file in files]
    if not all(ext == file_extensions[0] for ext in file_extensions):
        raise HTTPException(status_code=400, detail="All files must be of the same type (PDF or DOCX).")
    
    file_type = file_extensions[0]
    if file_type not in ['.pdf', '.docx']:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported for merging.")
    
    # Save uploaded files
    input_files = []
    for file in files:
        input_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        input_files.append(input_path)
    
    # Generate output filename
    output_filename = os.path.join(UPLOAD_DIR, f"merged_{files[0].filename}")
    
    # Merge files based on type
    success = False
    if file_type == '.pdf':
        success = merge_pdfs(input_files, output_filename)
    else:  # .docx
        success = merge_docx(input_files, output_filename)
    
    if not success:
        raise HTTPException(status_code=500, detail="Error merging files.")
    
    return FileResponse(
        output_filename,
        media_type="application/pdf" if file_type == '.pdf' else "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"merged_{files[0].filename}"
    )