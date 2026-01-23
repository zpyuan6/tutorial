# Extended GAIA Benchmark

## Abstract

This benchmark evaluates general-purpose AI assistants on an extended GAIA-style suite combining three challenging datasets:

1. **GAIA** - General AI Assistant benchmark with real-world questions requiring multi-step reasoning and tool use
2. **DocVQA** - Document Visual Question Answering tasks testing understanding of document images, layout, and embedded text
3. **SEALQA** - Search-augmented QA tasks stressing evidence selection and reasoning under noisy/conflicting web results

**GAIA Difficulty Levels:**
- **Level 1**: Minimal tool use, typically <5 steps
- **Level 2**: ~5–10 steps, requiring multiple tools and cross-source synthesis
- **Level 3**: Long-horizon, open-ended tool use and robust multi-step execution (near "general assistant" capability)

**Evaluation Capabilities:**
- **Reasoning** over real-world queries challenging for AI but straightforward for humans
- **Web browsing** and evidence gathering from multiple sources
- **Tool-use proficiency** - deciding when and how to use external tools for verifiable answers
- **Multimodal handling** including images, PDFs, Excel files, and other document formats
- **Code execution** for data analysis, web scraping, and computational tasks
- **Wikipedia API access** for historical edit information
- **Stock market data** retrieval and analysis


## How to start
1. Clone (or fork) the repo:
```
git clone git@github.com:agentbeats/tutorial
cd agentbeats-tutorial
```
2. Install dependencies
```
uv sync
```
3. Set environment variables
```
cp sample.env .env
```
Add your Google API key to the .env file

```
hf auth login
```
Login your huggingface account for accessing GAIA dataset

4. Run the GAIA Evaluator
```
uv run agentbeats-run scenarios/GAIA/scenario.toml
```
This command will:
- Start the agent servers using the commands specified in scenario.toml
- Construct an `assessment_request` message containing the participant's role-endpoint mapping and the assessment config
- Send the `assessment_request` to the green agent and print streamed responses

**Note:** Use `--show-logs` to see agent outputs during the assessment, and `--serve-only` to start agents without running the assessment.

To run this example manually, start the agent servers in separate terminals, and then in another terminal run the A2A client on the scenario.toml file to initiate the assessment.

## Configuration Options

The `scenario.toml` file supports the following configuration options:

### GAIA Evaluation
- `run_gaia` (boolean, default: true) - Enable/disable GAIA benchmark evaluation
- `evaluation_level` (string) - Select difficulty level: "all", "l1", "l2", or "l3"
- `gaia_split` (string, default: "validation") - Dataset split to use: "validation" or "test"

### DocVQA Evaluation
- `run_docvqa` (boolean, default: true) - Enable/disable DocVQA evaluation
- `docvqa_num_samples` (integer, default: 100) - Number of samples to evaluate
- `docvqa_seed` (integer, default: 0) - Random seed for sample selection

### SEALQA Evaluation
- `run_sealqa` (boolean, default: true) - Enable/disable SEALQA evaluation
- `sealqa_subset` (string, default: "seal_0") - Dataset subset to evaluate

### Environment Variables
- `ASSISTANT_MODEL` - Model for the assistant agent (default: "gemini-2.0-flash")
- `EVALUATOR_MODEL` - Model for the evaluator (default: "gemini-2.0-flash")
- `VISION_MODEL` - Model for image inspection (default: "gemini-2.0-flash")
- `MAX_TOOL_OUTPUT_CHARS` - Maximum characters in tool output (default: 8000)
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` - Required for Gemini models
- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` - Required for accessing GAIA dataset

5. Docker Build
Building evaluator on local(Please change Env Variable Format according to your system)
```
<!-- Build evaluator -->
docker build --platform linux/amd64 --build-arg HF_TOKEN=%HF_TOKEN% -t ghcr.io/zpyuan6/tutorial-gaia_extension:latest -f scenarios\GAIA\Dockerfile.GAIA-evaluator .
<!-- Build assistant -->
docker build --platform linux/amd64 --build-arg GOOGLE_API_KEY=%GOOGLE_API_KEY% -t ghcr.io/zpyuan6/tutorial-gaia_agent:latest -f scenarios\GAIA\Dockerfile.GAIA-agent .
```

6. Docker Run

```
docker run --network host -d ghcr.io/zpyuan6/tutorial-gaia_extension:latest --host 127.0.0.1
docker run --network host -d ghcr.io/zpyuan6/tutorial-gaia_agent:latest --host 127.0.0.1
```

## Project Structure
```
scenarios/GAIA/
├─ assistant.py                      # Purple agent (Google ADK) - the assistant being evaluated
├─ assistant_evaluator.py            # Green agent - evaluates assistants on GAIA, DocVQA, and SEALQA
├─ assistant_evaluation_common.py    # Shared models and agent card definitions
├─ tools.py                          # Tool implementations for the assistant
├─ scenario.toml                     # Configuration for the evaluation
├─ gaia_scenario.toml                # Alternative scenario configuration
├─ Dockerfile.GAIA-agent             # Docker configuration for assistant
├─ Dockerfile.GAIA-evaluator         # Docker configuration for evaluator
├─ check_model.py                    # Model validation utility
└─ workspace/                        # Working directory for file operations
    └─ results/                      # Evaluation results stored here

src/agentbeats/
├─ green_executor.py                 # Base A2A green agent executor
├─ models.py                         # Pydantic models for green agent IO
├─ client.py                         # A2A messaging helpers
├─ tool_provider.py                  # Tool provider for agent interactions
└─ run_scenario.py                   # Orchestrates agent startup and assessment
```

## Component Details

### Assistant Agent (Purple Agent)
The assistant agent ([assistant.py](assistant.py)) is implemented using Google ADK and provides:
- **Model**: Configurable via `ASSISTANT_MODEL` environment variable (default: gemini-2.0-flash)
- **Capabilities**: Streaming responses, file handling, multimodal inputs
- **Artifact Handling**: Automatically processes user-uploaded files and saves them as artifacts
- **Tools**: Full suite of tools for web browsing, file operations, code execution, and more

### Evaluator Agent (Green Agent)
The evaluator ([assistant_evaluator.py](assistant_evaluator.py)) orchestrates the evaluation:
- Loads datasets from HuggingFace (GAIA, DocVQA, SEALQA)
- Sends queries to the assistant agent with any required files
- Scores responses using dataset-specific metrics:
  - **GAIA**: Exact match scoring with normalization for numbers and lists
  - **DocVQA**: ANLS (Average Normalized Levenshtein Similarity) score
  - **SEALQA**: Strict accuracy and safe refusal rate
- Generates comprehensive JSON results with per-query details
- Computes overall capability score across all three benchmarks

### Tools Available to Assistant
The assistant has access to these tools ([tools.py](tools.py)):

1. **File Operations**:
   - `write_file(filename, content)` - Write text to files
   - `read_text_file(filename)` - Read plain text files
   - `read_excel(filename)` - Read Excel files as markdown tables
   - `read_pdf(filename, page_number)` - Extract text from PDFs
   - `read_file_from_artifact(filename)` - Read user-uploaded files
   - `list_files()` - List workspace files

2. **Web & Data**:
   - `visit_webpage(url)` - Fetch and parse web content
   - `get_wikipedia_history(page_title)` - Get Wikipedia edit history
   - `get_stock_prices(ticker)` - Retrieve stock market data

3. **Vision & Code**:
   - `inspect_image(filename, question)` - Analyze images using vision model
   - `execute_python(code)` - Execute Python code for data analysis

4. **Scoring**:
   - `question_scorer(model_answer, ground_truth)` - Evaluate answers

## Evaluation Metrics

### GAIA Scoring
Uses exact match after normalization:
- **Numbers**: Strips currency symbols, percentages, commas and compares as floats
- **Lists**: Splits on commas/semicolons, matches length and element-wise comparison
- **Strings**: Normalizes whitespace and punctuation for comparison
- Reports accuracy per level (Level 1, 2, 3) and total accuracy

### DocVQA Scoring
Uses ANLS (Average Normalized Levenshtein Similarity):
- Computes edit distance between prediction and each gold answer
- Normalizes by max length of prediction and gold answer
- Scores below 0.5 threshold are set to 0
- Returns best score across all gold answers
- Final metric is average ANLS across all samples

### SEALQA Scoring
Uses strict accuracy metrics:
- **Strict Accuracy**: Exact match after text normalization (lowercase, remove punctuation, remove whitespace)
- **Safe Refusal Rate**: Percentage of responses containing refusal patterns like "I don't know", "cannot", "unsure"
- Tracks both metrics to evaluate answer correctness and appropriate uncertainty

## Output Format

Results are saved as JSON files in `workspace/results/`:

### result.json
```json
{
  "gaia": {
    "evaluation_level": "all",
    "split": "validation",
    "num_items": {"1": 45, "2": 67, "3": 23},
    "score": {
      "Level 1": 0.85,
      "Level 2": 0.72,
      "Level 3": 0.45,
      "Total": 0.68
    },
    "average_time_to_answer_sec": 12.5,
    "responses_records": [...],
    "errors": [...]
  },
  "docvqa": {
    "dataset": "lmms-lab/DocVQA",
    "split": "validation",
    "num_samples": 100,
    "actual_num_samples": 98,
    "seed": 0,
    "anls": 0.75,
    "records": [...]
  },
  "sealqa": {
    "dataset": "vtllms/sealqa",
    "subset": "seal_0",
    "split": "test",
    "num_samples": 150,
    "strict_accuracy": 0.68,
    "safe_refusal_rate": 0.12,
    "records": [...]
  },
  "capability_score": 0.70
}
```

### errors.json
Contains detailed error logs for failed queries from each benchmark.

## Assistant Instructions

The assistant is configured with specific instructions to handle various task types:

1. **Never Give Up**: Always attempt to find a solution using available tools
2. **Wikipedia History**: Use `execute_python` with Wikipedia API or direct Wikimedia API calls for edit history queries
3. **File Handling**: Automatically processes artifacts and uses appropriate reader based on file type
4. **Python Execution**: Use for mathematical calculations, data analysis, and web scraping
5. **Output Limits**: Print only final values or small slices to avoid overwhelming the context
6. **No Hallucination**: Always verify information from actual sources

## Advanced Usage

### Running Individual Benchmarks

Modify `scenario.toml` to run specific benchmarks:
```toml
[config]
run_gaia = true
run_docvqa = false
run_sealqa = false
evaluation_level = "l1"
gaia_split = "validation"
```

### Custom Sampling

For DocVQA, control sampling:
```toml
[config]
docvqa_num_samples = 50  # Reduce sample size
docvqa_seed = 42         # Change random seed for different samples
```

### Using Different Models

Set environment variables before running:
```bash
export ASSISTANT_MODEL="gemini-1.5-pro"
export EVALUATOR_MODEL="gemini-2.0-flash"
export VISION_MODEL="gemini-1.5-pro-vision"
```

### Analyzing Results

Results include detailed per-query information:
- Individual predictions and ground truth
- Correctness flags
- Time taken per query
- Error messages for failed queries

Use this data for error analysis, model comparison, and identifying weak areas.

## Troubleshooting

### Dataset Access Issues
- Ensure `HF_TOKEN` is set and valid
- Run `hf auth login` to authenticate with HuggingFace
- Check that you have access to the gaia-benchmark/GAIA dataset

### File Download Failures
- Files are downloaded to `workspace/` directory
- Check network connectivity to HuggingFace mirrors
- Verify file paths in error logs

### Model API Errors
- Confirm `GOOGLE_API_KEY` or `GEMINI_API_KEY` is set
- Check API quota and rate limits
- Verify model names are valid for your API access

### Tool Execution Timeouts
- Python code execution has 60-second timeout
- Increase `MAX_TOOL_OUTPUT_CHARS` if truncation is too aggressive
- Check logs for specific tool failure reasons

## Contributing

To improve the benchmark:
1. Add new tools to `tools.py` for expanded capabilities
2. Implement additional evaluation metrics in `assistant_evaluator.py`
3. Extend scoring functions for new question types
4. Add support for more datasets

## References

- [GAIA Benchmark Paper](https://arxiv.org/abs/2311.12983)
- [DocVQA Dataset](https://www.docvqa.org/)
- [SEALQA Dataset](https://github.com/microsoft/SEALQA)
- [A2A Protocol](https://a2a-protocol.org/)
- [Google ADK Documentation](https://google.github.io/adk-docs/)
