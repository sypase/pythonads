from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import os
import re
import shutil
from docx import Document
import PyPDF2
from typing import List
import io
from langdetect import detect
import docx2txt

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
            {"path": "/status", "description": "Get status of processed files"},
            {"path": "/merge-files", "description": "Merge multiple PDF or DOCX files into a single file"},
            {"path": "/check-english", "description": "Check the percentage of English content in a file"}
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

def merge_pdfs_in_memory(pdf_files: List[UploadFile]) -> bytes:
    """Merge multiple PDF files in memory."""
    merger = PyPDF2.PdfMerger()
    try:
        for pdf_file in pdf_files:
            # Read the file content
            content = pdf_file.file.read()
            # Create a BytesIO object with the content
            pdf_stream = io.BytesIO(content)
            merger.append(pdf_stream)
            # Reset the file pointer for the next iteration
            pdf_file.file.seek(0)
        
        output = io.BytesIO()
        merger.write(output)
        merger.close()
        return output.getvalue()
    except Exception as e:
        print(f"Error merging PDFs: {str(e)}")
        return None
    finally:
        merger.close()

def merge_docx_in_memory(docx_files: List[UploadFile]) -> bytes:
    """Merge multiple DOCX files in memory."""
    try:
        merged_doc = Document()
        
        for index, file in enumerate(docx_files):
            # Read the file content
            content = file.file.read()
            # Create a BytesIO object with the content
            doc_stream = io.BytesIO(content)
            doc = Document(doc_stream)
            
            # Copy all elements from the source document
            for element in doc.element.body:
                merged_doc.element.body.append(element)
            
            # Add page break between documents (except after the last one)
            if index != len(docx_files) - 1:
                merged_doc.add_page_break()
            
            # Reset the file pointer for the next iteration
            file.file.seek(0)
        
        # Save to BytesIO
        output = io.BytesIO()
        merged_doc.save(output)
        return output.getvalue()
    except Exception as e:
        print(f"Error merging DOCX files: {str(e)}")
        return None

@app.post("/merge-files")
async def merge_files(files: List[UploadFile] = File(...)):
    """Merge multiple PDF or DOCX files into a single file."""
    try:
        if len(files) < 2:
            raise HTTPException(status_code=400, detail="At least 2 files are required for merging.")
        
        # Check if all files are of the same type
        file_extensions = [os.path.splitext(file.filename)[1].lower() for file in files]
        if not all(ext == file_extensions[0] for ext in file_extensions):
            raise HTTPException(status_code=400, detail="All files must be of the same type (PDF or DOCX).")
        
        file_type = file_extensions[0]
        if file_type not in ['.pdf', '.docx']:
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported for merging.")
        
        # Merge files based on type
        merged_content = None
        if file_type == '.pdf':
            merged_content = merge_pdfs_in_memory(files)
        else:  # .docx
            merged_content = merge_docx_in_memory(files)
        
        if not merged_content:
            raise HTTPException(status_code=500, detail="Error merging files. Please check the file formats and try again.")
        
        # Create a BytesIO object with the merged content
        file_stream = io.BytesIO(merged_content)
        
        return StreamingResponse(
            file_stream,
            media_type="application/pdf" if file_type == '.pdf' else "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f"attachment; filename=merged_{files[0].filename}"
            }
        )
    except Exception as e:
        print(f"Error in merge_files endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the files: {str(e)}")

def extract_pdf_content(pdf_file):
    """Extract text content from a PDF file."""
    try:
        print("Starting PDF content extraction...")
        reader = PyPDF2.PdfReader(pdf_file)
        print(f"Number of pages in PDF: {len(reader.pages)}")
        
        content = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                print(f"Page {i+1} text length: {len(text) if text else 0}")
                if text and text.strip():
                    content.append(text.strip())
            except Exception as page_error:
                print(f"Error extracting text from page {i+1}: {str(page_error)}")
                continue
        
        if not content:
            print("No text content extracted from PDF")
            return ""
            
        full_content = ' '.join(content)
        print(f"Total extracted text length: {len(full_content)}")
        return full_content
    except Exception as e:
        print(f"Error in extract_pdf_content: {str(e)}")
        return ""

def extract_docx_content(docx_file):
    """Extract text content from a DOCX file."""
    try:
        doc = Document(docx_file)
        content = []
        for para in doc.paragraphs:
            if para.text.strip():
                content.append(para.text)
        return ' '.join(content) if content else ""
    except Exception as e:
        print(f"Error in extract_docx_content: {str(e)}")
        return ""

def extract_doc_content(doc_file):
    """Extract text content from a DOC file."""
    try:
        content = docx2txt.process(doc_file)
        return content if content else ""
    except Exception as e:
        print(f"Error in extract_doc_content: {str(e)}")
        return ""

def check_english(content):
    """Check if content is in English."""
    try:
        if not content or len(content.strip()) < 10:  # Minimum content length check
            return False
        lang = detect(content)
        return lang == 'en'
    except Exception as e:
        print(f"Error in check_english: {str(e)}")
        return False

def process_file(file: UploadFile):
    """Process a file and return its content parts."""
    try:
        print(f"Processing file: {file.filename}")
        file_extension = os.path.splitext(file.filename)[1].lower()
        print(f"File extension: {file_extension}")
        content = ""

        # Read file content into memory
        file_content = file.file.read()
        print(f"File size: {len(file_content)} bytes")
        if not file_content:
            raise ValueError("Empty file content")
            
        file.file.seek(0)  # Reset file pointer
        
        # Create BytesIO object
        file_stream = io.BytesIO(file_content)
        
        if file_extension == ".pdf":
            content = extract_pdf_content(file_stream)
        elif file_extension == ".docx":
            content = extract_docx_content(file_stream)
        elif file_extension == ".doc":
            content = extract_doc_content(file_stream)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        print(f"Extracted content length: {len(content) if content else 0}")
        if not content or len(content.strip()) < 50:  # Minimum content length check
            raise ValueError("No sufficient content could be extracted from the file")

        # Split content into parts, ensuring minimum part size
        content_length = len(content)
        print(f"Content length before splitting: {content_length}")
        
        if content_length < 300:  # If content is too short, just use it as one part
            content_parts = [content]
        else:
            part_size = max(content_length // 30, 100)  # Minimum 100 characters per part
            content_parts = [content[i:i+part_size] for i in range(0, content_length, part_size)]
        
        # Filter out empty or too small parts
        content_parts = [part for part in content_parts if len(part.strip()) >= 10]
        print(f"Number of content parts after filtering: {len(content_parts)}")
        
        if not content_parts:
            raise ValueError("No valid content parts could be created")
            
        return content_parts
    except Exception as e:
        print(f"Error in process_file: {str(e)}")
        raise

@app.post("/check-english")
async def check_english_percentage(file: UploadFile = File(...)):
    """Check the percentage of English content in a file."""
    try:
        print(f"Starting English check for file: {file.filename}")
        content_parts = process_file(file)
        print(f"Number of content parts to analyze: {len(content_parts)}")
        
        if not content_parts:
            raise ValueError("No content parts to analyze")
            
        english_count = 0
        total_parts = len(content_parts)
        
        for i, part in enumerate(content_parts):
            try:
                print(f"Analyzing part {i+1}/{total_parts} (length: {len(part)})")
                is_english = check_english(part)
                if is_english:
                    english_count += 1
                print(f"Part {i+1}/{total_parts}: {'English' if is_english else 'Not English'}")
            except Exception as part_error:
                print(f"Error processing part {i+1}: {str(part_error)}")
                continue

        # Calculate percentage of English parts
        english_percentage = (english_count / total_parts) * 100 if total_parts > 0 else 0
        print(f"Final results - English parts: {english_count}/{total_parts} ({english_percentage:.2f}%)")
        
        return JSONResponse(
            status_code=200,
            content={
                "filename": file.filename,
                "english_percentage": round(english_percentage, 2),
                "total_parts_analyzed": total_parts,
                "english_parts": english_count
            }
        )
    except Exception as e:
        print(f"Error in check_english_percentage: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")