# GAIA Extension Benchmarking (Green Agent) for AgentBeats

This is a repository for extended GAIA benchmarking implementation, following the AgentBeats framework.

This benchmark evaluate general AI assistant on real-world questions requiring multi-step reasoning, tool use, and web search. 

## Overview

This is an **agentified** implementation of an extended [GAIA Benchmark](https://huggingface.co/gaia-benchmark) on the [AgentBeats](https://agentbeats.org) platform. This benchmark coverd following datasets.

1. **GAIA** - General AI Assistant benchmark with 450+ real-world questions requiring multi-step reasoning and tool use
2. **DocVQA** - Document Visual Question Answering tasks testing understanding of document images, layout, and embedded text
3. **SEALQA** - Search-augmented QA tasks stressing evidence selection and reasoning under noisy/conflicting web results

The evaluated AI assistant should have following **Capabilities:**

- **Reasoning** over real-world queries challenging for AI but straightforward for humans
- **Web browsing** and evidence gathering from multiple sources
- **Tool-use proficiency** - deciding when and how to use external tools for verifiable answers
- **Multimodal handling** including images, PDFs, Excel files, and other document formats
- **Code execution** for data analysis, web scraping, and computational tasks
- **Wikipedia API access** for historical edit information
- **Stock market data** retrieval and analysis

## How to start
1. Clone the repo
```
git clone https://github.com/zpyuan6/tutorial.git
cd tutorial
```
2. Install dependencies
```
uv sync
```
3. Set environment variables

Add your Google API key to the .env file and login to HuggingFace (for access GAIA benchmarking dataset) by 
```
hf auth login
```

4. Run the extended GAIA Evaluator

File scenario.toml define the running parameter of evaluating scenario. You can control the evaluation process by adjust the parameter setting.

You can run all evaluating, covering three dataset by
```
uv run agentbeats-run scenarios/GAIA/scenario.toml
```
This command will:
- Start the agent servers using the commands specified in scenario.toml
- Construct an `assessment_request` message containing the participant's role-endpoint mapping and the assessment config
- Send the `assessment_request` to the green agent and print streamed responses

This will evaluate a general-purpose assistant on three challenging benchmarks:
- **GAIA** - 100+ real-world questions (validation dataset) across 3 difficulty levels
- **DocVQA** - Document visual question answering (images, PDFs, layouts)
- **SEALQA** - Search-augmented QA with noisy web results

**Note:** Use `--show-logs` to see agent outputs during the assessment, and `--serve-only` to start agents without running the assessment.

## Project Structure
```
src/agentbeats/
├─ green_executor.py           # Base A2A green agent executor
├─ models.py                   # Pydantic models for green agent IO
├─ client.py                   # A2A messaging helpers
├─ tool_provider.py            # Tool provider for agent interactions
└─ run_scenario.py             # Orchestrates agent startup and assessment

scenarios/GAIA/                       # EXTENDED GAIA BENCHMARK (Primary)
├─ assistant.py             # Purple agent - General-purpose assistant
├─ assistant_evaluator.py   # Green agent - Evaluates on GAIA/DocVQA/SEALQA
├─ assistant_evaluation_common.py  # Shared models and definitions
├─ tools.py                 # Tool suite (web, files, vision, code execution)
├─ scenario.toml            # Configuration for GAIA benchmark
├─ Dockerfile.GAIA-agent    # Docker for assistant
├─ Dockerfile.GAIA-evaluator # Docker for evaluator
└─ workspace/               # Working directory for evaluation

```

## 🌟 GAIA Benchmark (Primary Scenario)

The GAIA scenario is the recommended starting point for evaluating AI assistants:

**What it evaluates:**
- **General AI Assistants** with real-world problem-solving capabilities
- **Three benchmark datasets**: GAIA (QA), DocVQA (visual docs), SEALQA (web search)
- **Three difficulty levels**: Level 1 (simple), Level 2 (complex), Level 3 (expert)

**Key capabilities tested:**
- Multi-step reasoning and planning
- Tool-use proficiency (web browsing, file operations, code execution)
- Multimodal understanding (text, images, PDFs, Excel files)
- Wikipedia and stock market data retrieval
- Error handling and safe refusals


**See detailed docs:** [GAIA README](scenarios/GAIA/README.md)


## Dockernize

This project provide dockerfiles and Github Workflow for dockernizing the green (evaluator) and purple (assistant) agents.

### Manually Build Docker

Build docker image is not neccessary, as the corresponding images are built with Github Workflow. You can directly pull the corresponding images.
But if you want, you can use following comments to build docker image in local. 
```
<!-- Build evaluator -->
docker build --platform linux/amd64 --build-arg HF_TOKEN=%HF_TOKEN% -t ghcr.io/zpyuan6/tutorial-gaia_extension:latest -f scenarios\GAIA\Dockerfile.GAIA-evaluator .
<!-- Build assistant -->
docker build --platform linux/amd64 --build-arg GOOGLE_API_KEY=%GOOGLE_API_KEY% -t ghcr.io/zpyuan6/tutorial-gaia_agent:latest -f scenarios\GAIA\Dockerfile.GAIA-agent .
```

### Run with Docker in Local 

As containers require network connectivity, running under host model is most easy way. You can run the images with following commands.
```
docker run --network host -d ghcr.io/zpyuan6/tutorial-gaia_extension:latest --host 127.0.0.1
docker run --network host -d ghcr.io/zpyuan6/tutorial-gaia_agent:latest --host 127.0.0.1
```
Then, you can run the senario.

```
uv run agentbeats-run scenarios/GAIA/scenario.toml
```

# AgentBeats Tutorial
Welcome to the AgentBeats Tutorial! 🤖🎵

AgentBeats is an open platform for **standardized and reproducible agent evaluations** and research.

This tutorial is designed to help you get started, whether you are:
- 🔬 **Researcher** → running controlled experiments and publishing reproducible results
- 🛠️ **Builder** → developing new agents and testing them against benchmarks
- 📊 **Evaluator** → designing benchmarks, scenarios, or games to measure agent performance
- ✨ **Enthusiast** → exploring agent behavior, running experiments, and learning by tinkering

By the end, you’ll understand:
- The core concepts behind AgentBeats - green agents, purple agents, and A2A assessments
- How to run existing evaluations on the platform via the web UI
- How to build and test your own agents locally
- Share your agents and evaluation results with the community

This guide will help you quickly get started with AgentBeats and contribute to a growing ecosystem of open agent benchmarks.

## Core Concepts

**Green agents** orchestrate and manage evaluations of one or more purple agents by providing an evaluation harness.
- In GAIA: The green agent loads datasets, sends queries, scores responses, and generates results
- A green agent may implement a single-player benchmark or a multi-player game where agents compete or collaborate
- It sets the rules, hosts the match, and decides results

**Purple agents** are the participants being evaluated. They possess certain skills that green agents assess.
- In GAIA: The purple agent is the assistant being evaluated on QA, visual QA, and search tasks
- They demonstrate capabilities like reasoning, tool-use, and multimodal understanding
- In security-themed games, agents are referred to as red and blue (attackers and defenders)

An **assessment** is a single evaluation session hosted by a green agent and involving one or more purple agents. Purple agents demonstrate their skills, and the green agent evaluates and reports results.

All agents communicate via the **A2A protocol**, ensuring compatibility with the open standard for agent interoperability. Learn more about A2A [here](https://a2a-protocol.org/latest/).

## Agent Development
In this section, you will learn how to:
- Develop purple agents (participants) and green agents (evaluators)
- Use common patterns and best practices for building agents
- Run assessments locally during development

### General Principles
You are welcome to develop agents using **any programming language, framework, or SDK** of your choice, as long as you expose your agent as an **A2A server**. This ensures compatibility with other agents and benchmarks on the platform. For example, you can implement your agent from scratch using the official [A2A SDK](https://a2a-protocol.org/latest/sdk/), or use a downstream SDK such as [Google ADK](https://google.github.io/adk-docs/).

#### Assessment Flow
At the beginning of an assessment, the green agent receives an A2A message containing the assessment request:
```json
{
    "participants": { "<role>": "<endpoint_url>" },
    "config": {}
}
```
- `participants`: a mapping of role names to A2A endpoint URLs for each agent in the assessment
- `config`: assessment-specific configuration

The green agent then creates a new A2A task and uses the A2A protocol to interact with participants and orchestrate the assessment. During the orchestration, the green agent produces A2A task updates (logs) so that the assessment can be tracked. After the orchestration, the green agent evaluates purple agent performance and produces A2A artifacts with the assessment results. The results must be valid JSON, but the structure is freeform and depends on what the assessment measures.

#### Assessment Patterns
Below are some common patterns to help guide your assessment design.

- **Artifact submission**: The purple agent produces artifacts (e.g. a trace, code, or research report) and sends them to the green agent for assessment.
- **Traced environment**: The green agent provides a traced environment (e.g. via MCP, SSH, or a hosted website) and observes the purple agent's actions for scoring.
- **Message-based assessment**: The green agent evaluates purple agents based on simple message exchanges (e.g. question answering, dialogue, or reasoning tasks).
- **Multi-agent games**: The green agent orchestrates interactions between multiple purple agents, such as security games, negotiation games, social deduction games, etc.

#### Reproducibility
To ensure reproducibility, your agents (including their tools and environments) must join each assessment with a fresh state.

### 🌟 Primary Example: GAIA Benchmark

The GAIA scenario demonstrates a sophisticated evaluation system:
- **Green agent** (`GAIAAssistantEvaluator`) orchestrates three separate benchmark evaluations
  - Loads datasets from HuggingFace (GAIA, DocVQA, SEALQA)
  - Sends queries to the assistant with attachments (documents, images)
  - Scores responses using task-specific metrics
  - Generates detailed JSON results with per-query analysis
  - Computes overall capability scores

- **Purple agent** (`Assistant`) is a general-purpose AI assistant with:
  - Web browsing and search capabilities
  - File reading/writing (PDF, Excel, images, text)
  - Python code execution environment
  - Wikipedia API access
  - Stock market data retrieval
  - Artifact handling for user uploads

To run, execute: `uv run agentbeats-run scenarios/GAIA/scenario.toml`

Results include detailed metrics and are saved to `scenarios/GAIA/workspace/results/`.

### Secondary Example: Debate Scenario

For a simpler introduction to agent development:
- Green agent (`DebateJudge`) orchestrates debate between two agents
- Purple agents (`Debater`) present arguments for their assigned positions
- LLM-as-Judge evaluation determines winner

To run, execute: `uv run agentbeats-run scenarios/debate/scenario.toml`

### Dockerizing Agent

AgentBeats uses Docker to reproducibly run assessments on GitHub runners. Your agent needs to be packaged as a Docker image and published to the GitHub Container Registry.

**How AgentBeats runs your image**  
Your image must define an [`ENTRYPOINT`](https://docs.docker.com/reference/dockerfile/#entrypoint) that starts your agent server and accepts the following arguments:
- `--host`: host address to bind to
- `--port`: port to listen on
- `--card-url`: the URL to advertise in the agent card

**Build and publish steps**
1. Create a Dockerfile for your agent. See example [Dockerfiles](./scenarios/debate).
2. Build the image
```bash
docker build --platform linux/amd64 -t ghcr.io/yourusername/your-agent:v1.0 .
```
**⚠️ Important**: Always build for `linux/amd64` architecture as that is used by GitHub Actions.

3. Push to GitHub Container Registry
```bash
docker push ghcr.io/yourusername/your-agent:v1.0
```

We recommend setting up a GitHub Actions [workflow](.github/workflows/publish.yml) to automatically build and publish your agent images.

## Best Practices 💡

Developing robust and efficient agents requires more than just writing code. Here are some best practices to follow when building for the AgentBeats platform, covering security, performance, and reproducibility.

### API Keys and Cost Management

AgentBeats uses a Bring-Your-Own-Key (BYOK) model. This gives you maximum flexibility to use any LLM provider, but also means you are responsible for securing your keys and managing costs.

-   **Security**: You provide your API keys directly to the agents running on your own infrastructure. Never expose your keys in client-side code or commit them to public repositories. Use environment variables (like in the tutorial's `.env` file) to manage them securely.

-   **Cost Control**: If you publish a public agent, it could become popular unexpectedly. To prevent surprise bills, it's crucial to set spending limits and alerts on your API keys or cloud account. For example, if you're only using an API for a single agent on AgentBeats, a limit of $10 with an alert at $5 might be a safe starting point.

#### Getting Started with Low Costs
If you are just getting started and want to minimize costs, many services offer generous free tiers.
-   **Google Gemini**: Often has a substantial free tier for API access.
-   **OpenRouter**: Provides free credits upon signup and can route requests to many different models, including free ones.
-   **Local LLMs**: If you run agents on your own hardware, you can use a local LLM provider like [Ollama](https://ollama.com/) to avoid API costs entirely.

#### Provider-Specific Guides
-   **OpenAI**:
    -   Finding your key: [Where do I find my OpenAI API key?](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)
    -   Setting limits: [Usage limits](https://platform.openai.com/settings/organization/limits)

-   **Anthropic (Claude)**:
    -   Getting started: [API Guide](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
    -   Setting limits: [Spending limits](https://console.anthropic.com/settings/limits)

-   **Google Gemini**:
    -   Finding your key: [Get an API key](https://ai.google.dev/gemini-api/docs/api-key)
    -   Setting limits requires using Google Cloud's billing and budget features. Be sure to set up [billing alerts](https://cloud.google.com/billing/docs/how-to/budgets).

-   **OpenRouter**:
    -   Request a key from your profile page under "Keys".
    -   You can set a spending limit directly in the key creation flow. This limit aggregates spend across all models accessed via that key.

### Efficient & Reliable Assessments

#### Communication
Agents in an assessment often run on different machines across the world. They communicate over the internet, which introduces latency.

-   **Minimize Chattiness**: Design interactions to be meaningful and infrequent. Avoid back-and-forth for trivial information.
-   **Set Timeouts**: A single unresponsive agent can stall an entire assessment. Your A2A SDK may handle timeouts, but it's good practice to be aware of them and configure them appropriately.
-   **Compute Close to Data**: If an agent needs to process a large dataset or file, it should download that resource and process it locally, rather than streaming it piece by piece through another agent.

#### Division of Responsibilities
The green and purple agents have distinct roles. Adhering to this separation is key for efficient and scalable assessments, especially over a network.

-   **Green agent**: A lightweight verifier or orchestrator. Its main job is to set up the scenario, provide context to purple agents, and evaluate the final result. It should not perform heavy computation.
-   **Purple agent**: The workhorse. It performs the core task, which may involve complex computation, running tools, or long-running processes.

Here's an example for a security benchmark:
1.  The **green agent** defines a task (e.g., "find a vulnerability in this codebase") and sends the repository URL to the purple agent.
2.  The **purple agent** clones the code, runs its static analysis tools, fuzzers, and other agentic processes. This could take a long time and consume significant resources.
3.  Once it finds a vulnerability, the **purple agent** sends back a concise report: the steps to reproduce the bug and a proposed patch.
4.  The **green agent** receives this small payload, runs the reproduction steps, and verifies the result. This final verification step is quick and lightweight.

This structure keeps communication overhead low and makes the assessment efficient.

### Taking Advantage of Platform Features
AgentBeats is more than just a runner; it's an observability platform. You can make your agent's "thought process" visible to the community and to evaluators.

-   **Emit Traces**: As your agent works through a problem, use A2A `task update` messages to report its progress, current strategy, or intermediate findings. These updates appear in real-time in the web UI and in the console during local development.
-   **Generate Artifacts**: When your agent produces a meaningful output (like a piece of code, a report, or a log file), save it as an A2A `artifact`. Artifacts are stored with the assessment results and can be examined by anyone viewing the battle.

Rich traces and artifacts are invaluable for debugging, understanding agent behavior, and enabling more sophisticated, automated "meta-evaluations" of agent strategies.

### Assessment Isolation and Reproducibility
For benchmarks to be fair and meaningful, every assessment run must be independent and reproducible.

-   **Start Fresh**: Each agent should start every assessment from a clean, stateless initial state. Avoid carrying over memory, files, or context from previous battles.
-   **Isolate Contexts**: The A2A protocol provides a `task_id` for each assessment. Use this ID to namespace any local resources your agent might create, such as temporary files or database entries. This prevents collisions between concurrent assessments.
-   **Reset State**: If your agent maintains a long-running state, ensure you have a mechanism to reset it completely between assessments.

Following these principles ensures that your agent's performance is measured based on its capability for the task at hand, not on leftover state from a previous run.

## Next Steps

### Evaluate Your Assistant
- 🌟 **Run GAIA Benchmark** → Test your assistant on GAIA, DocVQA, and SEALQA (primary scenario)
- 📊 **Analyze Results** → Review detailed metrics in `scenarios/GAIA/workspace/results/`
- 🔧 **Customize Config** → Modify `scenarios/GAIA/scenario.toml` for different evaluation levels
- 📖 **Detailed Guide** → See [GAIA Benchmark README](scenarios/GAIA/README.md) for advanced options

### Develop on AgentBeats
- 📊 **Develop new assessments** → Build a green agent along with baseline purple agents. Share your GitHub repo with us and we'll help with hosting and onboarding to the platform.
- 🛠️ **Extend Tools** → Add new capabilities to `scenarios/GAIA/tools.py` for enhanced assistant abilities
- 🏆 **Evaluate your agents** → Create and test agents against GAIA or design new benchmarks
- 🌐 **Join the community** → Connect with researchers, builders, and enthusiasts to exchange ideas, share results, and collaborate on new evaluations.

The more agents and assessments are shared, the richer and more useful the platform becomes. We’re excited to see what you create!
