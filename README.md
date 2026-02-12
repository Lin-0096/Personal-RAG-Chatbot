# Support Copilot RAG

This project implements a Retrieval-Augmented Generation (RAG) based Support Copilot that utilizes the OpenAI API for embeddings and answering. It features a local vector database using Chroma and loads knowledge from a specified local folder. The project includes a simple command-line interface (CLI) for user interaction, allowing users to ask questions and receive structured answers along with the sources of the information.

## Project Structure

```
support-copilot-rag
├── src
│   └── support_copilot
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── ingest.py
│       ├── rag.py
│       ├── retriever.py
│       ├── llm.py
│       ├── embeddings.py
│       ├── sources.py
│       ├── utils.py
│       └── types
│           └── __init__.py
├── knowledge
│   └── .gitkeep
├── tests
│   ├── __init__.py
│   └── test_rag.py
├── pyproject.toml
├── README.md
└── .env.example
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd support-copilot-rag
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables by copying `.env.example` to `.env` and filling in the necessary API keys.

## Usage

This project uses a `src/` layout. If you have not installed it as a package yet,
run the CLI by executing the script directly.

To start the Support Copilot, run the following command in your terminal (from the `support-copilot-rag` folder):

```
python src/support_copilot/cli.py
```

One-shot question (non-interactive):

```
python src/support_copilot/cli.py --ask "Standard shipping takes how long?"
```

Alternatively, if you prefer module execution, add `src` to `PYTHONPATH`:

```
PYTHONPATH=src python -m support_copilot.cli --ask "Standard shipping takes how long?"
```

Note: if you copy commands from VS Code, make sure you're copying the raw command.
Text like `[python](http://_vscodecontentref_/...)` is a clickable link, not a shell command.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.