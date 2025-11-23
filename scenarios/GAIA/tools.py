import os
import requests
import mimetypes
# import google.generativeai as genai
import re
import pypdf

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
    
def read_file(filename: str) -> str:
    try:
        filepath = _get_safe_path(filename)
        if not os.path.exists(filepath):
            return f"Error: File {filename} does not exist."
        
        _, ext = os.path.splitext(filename)
        if ext.lower() in ['.xlsx', '.xls', '.png', '.jpg', '.jpeg', '.pdf', '.zip', '.mp3']:
            return f"Error: File {filename} is a binary file ({ext}). Please use Python code (e.g., pandas for excel) to process it instead of `read_file`."

        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        return "Error: Unable to decode file as text. It might be a binary file."
    except Exception as e:
        return f"Error reading file: {str(e)}"
    

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
    
# def inspect_image(filename: str, question: str) -> str:
#     print(f"Tool: Inspecting image {filename}...")
#     try:
#         filepath = _get_safe_path(filename)
#         if not os.path.exists(filepath):
#             return f"Error: Image {filename} not found."

#         mime_type, _ = mimetypes.guess_type(filepath)
#         if not mime_type:
#             mime_type = "image/png" 

#         with open(filepath, "rb") as f:
#             image_data = f.read()

#         image_blob = {
#             "mime_type": mime_type,
#             "data": image_data
#         }

#         vision_model = genai.GenerativeModel('gemini-1.5-flash')
#         response = vision_model.generate_content([question, image_blob])
        
#         return f"Vision Response: {response.text}"
        
#     except Exception as e:
#         return f"Error inspecting image: {str(e)}"