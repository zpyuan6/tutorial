import os
import requests
import mimetypes
import google.generativeai as genai
import re
import pypdf
import pandas as pd
import PIL.Image
import yfinance as yf
import subprocess
import sys
import re
import string
import warnings
from pathlib import Path

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(CURRENT_FILE_DIR, "workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)
MAX_TOOL_OUTPUT_CHARS = int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "8000"))

_GENAI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if _GENAI_API_KEY:
    genai.configure(api_key=_GENAI_API_KEY)

def _get_safe_path(filename: str) -> str:
    local_filename = os.path.basename(filename)
    return os.path.join(WORKSPACE_DIR, local_filename)


def _truncate_text(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    head_len = limit // 2
    tail_len = limit - head_len
    head = text[:head_len]
    tail = text[-tail_len:] if tail_len else ""
    return f"{head}\n...[truncated {len(text) - limit} chars]...\n{tail}"


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
        filepath = filename
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
        filepath = filename
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
    

async def read_file_from_artifact(filename: str, tool_context):
    """Fetches content from user uploaded artifact."""
    part = await tool_context.load_artifact(filename=filename)  # :contentReference[oaicite:1]{index=1}
    inline = getattr(part, "inline_data", None)

    print(f"Tool: Reading file from artifact {filename}...")
    
    if not inline or not getattr(inline, "data", None):
        return f"Error: Artifact {filename} has no inline data."

    mime = inline.mime_type
    data: bytes = inline.data

    suffix = {
        "application/pdf": ".pdf",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "text/csv": ".csv",
        "application/json": ".json",
        "text/plain": ".txt",
    }.get(mime, "")

    out_dir = Path("tmp/adk_artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = Path(filename).name  # 防止路径穿越
    if suffix and not base.lower().endswith(suffix):
        base = f"{Path(base).stem}{suffix}"

    out_path = out_dir / base
    out_path.write_bytes(data)

    print("Tool: Written artifact to", str(out_path))

    # call specific reader based on mime type
    if mime == "application/pdf":
        content = read_pdf(str(out_path))
    elif mime == "text/plain":
        content = read_text_file(str(out_path))
    elif mime in ['image/png', 'image/jpeg']:
        content = inspect_image(str(out_path), "Describe the content of the image.")
    elif mime in ['text/csv', 'application/json', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
        content = read_excel(str(out_path))
    else:
        content = f"Please use code to read {mime} file: {out_path}."
    
    return content


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
        filepath = filename
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
    print(f"Tool: Inspecting image {filename}, with question {question}..")
    try:
        filepath = filename
        if not os.path.exists(filepath):
            return f"Error: Image {filename} not found."
        
        try:
            img = PIL.Image.open(filepath)
        except Exception as e:
            return f"Error: The file exists but is not a valid image. ({str(e)})"

        vision_model_name = os.getenv("VISION_MODEL", "gemini-2.0-flash")
        vision_model = genai.GenerativeModel(vision_model_name)

        response = vision_model.generate_content([question, img])

        if response.parts:
            return f"Vision Response: {response.text}"
        else:
            return "Error: Vision model refused to answer (safety filters triggered)."
        
    except Exception as e:
        return f"Error inspecting image: {str(e)}"
    
def get_stock_prices(ticker: str) -> str:
    """
    Gets the recent stock market data for a specific ticker symbol (e.g., 'AAPL', 'TSLA', 'NVDA').
    Returns the last 1 month of daily data including Open, High, Low, Close, and Volume.
    """
    print(f"Tool: Getting stock prices for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        

        hist = stock.history(period="1mo")
        
        if hist.empty:
            return f"Error: No data found for symbol '{ticker}'. Check if the ticker is correct."
            
  
        recent_data = hist.tail(10)
        return f"Stock Data for {ticker} (Last 10 trading days):\n{recent_data.to_markdown()}"
        
    except Exception as e:
        return f"Error fetching stock data: {str(e)}"
    
def execute_python(code: str) -> str:
    """
    Executes Python code in a separate process.
    The code runs in the current environment (so it can use installed libraries like pandas, yfinance).
    Standard output (stdout) and error (stderr) are captured and returned.
    
    Args:
        code: The Python code string to execute.
    """
    print(f"🐍 Tool: Executing Python Code...\n{'-'*20}\n{code}\n{'-'*20}")
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60, 
            cwd=WORKSPACE_DIR 
        )
        

        output = _truncate_text(result.stdout.strip(), MAX_TOOL_OUTPUT_CHARS)
        error = _truncate_text(result.stderr.strip(), MAX_TOOL_OUTPUT_CHARS)
        
        if result.returncode == 0:
            if not output:
                return "Code executed successfully, but printed nothing. (Did you forget to `print(...)`?)"
            return f"Execution Output:\n{output}"
        else:

            return f"Execution Error:\n{error}"
            
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (limit: 60s)."
    except Exception as e:
        return f"System Error executing code: {str(e)}"
    
def get_wikipedia_history(page_title: str) -> str:
    """
    Gets the revision history of a Wikipedia page using the Wikimedia API.
    Returns the first 50 revisions (creation history) and the last 20 revisions (recent history).
    Useful for finding when a page was created, who edited it, or specific edits on dates.
    """
    print(f"Tool: Fetching history for '{page_title}'...")
    try:
        url = "https://en.wikipedia.org/w/api.php"
        
        params_old = {
            "action": "query",
            "prop": "revisions",
            "titles": page_title,
            "rvlimit": "50",
            "rvprop": "timestamp|user|comment",
            "rvdir": "newer", 
            "format": "json"
        }
        
        resp = requests.get(url, params=params_old, headers={"User-Agent": "AgentiX-Bot/1.0"})
        data = resp.json()
        
        pages = data.get("query", {}).get("pages", {})
        page_id = list(pages.keys())[0]
        
        if page_id == "-1":
            return f"Error: Wikipedia page '{page_title}' not found."
            
        revisions = pages[page_id].get("revisions", [])
        
        result = f"--- Edit History for '{page_title}' ---\n"
        result += "=== First 50 Edits (Oldest) ===\n"
        for rev in revisions:
            result += f"Date: {rev['timestamp']} | User: {rev.get('user', 'Hidden')} | Comment: {rev.get('comment', '')}\n"
            
        return result[:8000] 
        
    except Exception as e:
        return f"Error fetching Wikipedia history: {str(e)}"


def normalize_number_str(number_str: str) -> float:
    # we replace these common units and commas to allow
    # conversion to float
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        print(f"String {number_str} cannot be normalized to number str.")
        return float("inf")


def split_string(
    s: str,
    char_list: list[str] = [",", ";"],
) -> list[str]:
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def question_scorer(
    model_answer: str,
    ground_truth: str,
) -> bool:
    def is_float(element: any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False
        
    if model_answer is None:
        model_answer = "None"

    # if gt is a number
    if is_float(ground_truth):
        print(f"Evaluating {model_answer} as a number.")
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth)

    # if gt is a list
    elif any(char in ground_truth for char in [",", ";"]):
        print(f"Evaluating {model_answer} as a comma separated list.")
        # question with the fish: normalization removes punct

        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        # check length is the same
        if len(gt_elems) != len(ma_elems):
            warnings.warn(
                "Answer lists have different lengths, returning False.", UserWarning
            )
            return False

        # compare each element as float or str
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                # we do not remove punct since comparisons can include punct
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    # if gt is a str
    else:
        print(f"Evaluating {model_answer} as a string.")
        return normalize_str(model_answer) == normalize_str(ground_truth)


def normalize_str(input_str, remove_punct=True) -> str:
    """
    Normalize a string by:
    - Removing all white spaces
    - Optionally removing punctuation (if remove_punct is True)
    - Converting to lowercase
    Parameters:
    - input_str: str, the string to normalize
    - remove_punct: bool, whether to remove punctuation (default: True)
    Returns:
    - str, the normalized string
    """
    # Remove all white spaces. Required e.g for seagull vs. sea gull
    no_spaces = re.sub(r"\s", "", input_str)

    # Remove punctuation, if specified.
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()
