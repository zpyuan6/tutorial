import os
import requests
import mimetypes
import google.generativeai as genai
import re
import pypdf
import pandas as pd
import PIL.Image

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(CURRENT_FILE_DIR, "workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

def _get_safe_path(filename: str) -> str:
    local_filename = os.path.basename(filename)
    return os.path.join(WORKSPACE_DIR, local_filename)


def write_file(filename: str, content: str) -> str:
    
    try:
        filepath = _get_safe_path(filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {filename}."
    except Exception as e:
        return f"Error writing file: {str(e)}"
    
def read_text_file(filename: str) -> str:
    """
    Reads plain text files (.txt, .csv, .json, .py, .md).
    DO NOT use this for .xlsx, .pdf, or images.
    """
    try:
        filepath = _get_safe_path(filename)
        if not os.path.exists(filepath):
            return f"Error: File {filename} not found."
        
        _, ext = os.path.splitext(filename)
        if ext.lower() in ['.xlsx', '.xls']:
            return f"Error: {filename} is an Excel file. Use `read_excel` tool."
        if ext.lower() in ['.pdf']:
            return f"Error: {filename} is a PDF. Use `read_pdf` tool."
            
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            if len(content) > 10000:
                return f"Content ({filename}):\n{content[:10000]}...\n(Truncated, file is too large)"
            return content
            
    except UnicodeDecodeError:
        return f"Error: File {filename} is not plain text. Try another tool."
    except Exception as e:
        return f"Error reading text file: {str(e)}"
    
def read_excel(filename: str) -> str:
    """
    Reads an Excel file (.xlsx) and returns its content as a Markdown table.
    """
    print(f"Tool: Reading Excel {filename}...")
    try:
        filepath = _get_safe_path(filename)
        if not os.path.exists(filepath):
            return f"Error: File {filename} not found."
        
        df = pd.read_excel(filepath, nrows=50)
        

        info = f"Shape: {df.shape} (Rows, Columns)\nColumns: {list(df.columns)}\n"
        

        markdown_table = df.to_markdown(index=False)
        
        return f"Excel Content ({filename}):\n{info}\n{markdown_table}\n\n(Note: Only first 50 rows displayed. If you need more analysis, use `execute_python` with pandas.)"
        
    except Exception as e:
        return f"Error reading Excel: {str(e)}"
    
def list_files() -> str:
    """Lists all files in the current workspace."""
    try:
        files = os.listdir(WORKSPACE_DIR)
        return f"Files in workspace: {', '.join(files)}" if files else "Workspace is empty."
    except Exception as e:
        return f"Error listing files: {str(e)}"


def visit_webpage(url: str) -> str:
    """Fetches text content from a URL using standard Python libraries."""
    print(f"Tool: Visiting {url}...")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
 
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        
        html = resp.text
        

        html = re.sub(r'<(script|style).*?>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)

        text = re.sub(r'<[^>]+>', ' ', html)

        text = re.sub(r'\s+', ' ', text).strip()
        
        return f"URL Content ({url}):\n{text[:6000]}...\n(Content truncated)"
        
    except Exception as e:
        return f"Error visiting {url}: {str(e)}"
    

def read_pdf(filename: str, page_number: int = 0) -> str:
    """
    Reads text from a PDF file.
    Args:
        filename: The name of the PDF file in workspace.
        page_number: (Optional) The specific page number to read (1-based index). 
                     If not provided, reads the whole document.
    """
    print(f"Tool: Reading PDF {filename}...")
    try:
        filepath = _get_safe_path(filename)
        if not os.path.exists(filepath):
            return f"Error: File {filename} not found."
        
        reader = pypdf.PdfReader(filepath)
        total_pages = len(reader.pages)
        
        text_content = []
        
        if page_number:
            idx = int(page_number) - 1
            if 0 <= idx < total_pages:
                text = reader.pages[idx].extract_text()
                return f"--- Page {page_number} of {filename} ---\n{text}"
            else:
                return f"Error: Page {page_number} out of range (Total pages: {total_pages})"
        else:
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                text_content.append(f"--- Page {i+1} ---\n{text}")
            
            full_text = "\n".join(text_content)
            return f"PDF Content ({filename}):\n{full_text[:8000]}...\n(Content truncated)"
            
    except Exception as e:
        return f"Error reading PDF: {str(e)}"
    
def inspect_image(filename: str, question: str) -> str:
    print(f"Tool: Inspecting image {filename}...")
    try:
        filepath = _get_safe_path(filename)
        if not os.path.exists(filepath):
            return f"Error: Image {filename} not found."


        try:
            img = PIL.Image.open(filepath)
        except Exception as e:
            return f"Error: The file exists but is not a valid image. ({str(e)})"


        vision_model = genai.GenerativeModel('gemini-2.0-flash-exp')

        response = vision_model.generate_content([question, img])
        

        if response.parts:
            return f"Vision Response: {response.text}"
        else:
            return "Error: Vision model refused to answer (safety filters triggered)."
        
    except Exception as e:
        return f"Error inspecting image: {str(e)}"