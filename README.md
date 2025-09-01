# AI with knowledge graph
Local ai with knowledge graph

## Project Setup: Ollama and Neo4j for a Local AI Knowledge Graph

This `README.md` provides a step-by-step guide to setting up a local development environment for building an AI-powered knowledge graph. We will use Ollama to run large language models locally and Neo4j to manage our graph database.

### Prerequisites

  - A compatible operating system (macOS, Windows, or Linux).
  - Ollama and Neo4j Desktop as standalone applications.

### Step 1: Install Ollama

1.  Go to the official Ollama website: `https://ollama.com/`
2.  Click on the "Download" button and select the installer for your operating system (macOS, Windows, or Linux).
3.  Run the installer and follow the on-screen instructions.
4.  Once installed, open a terminal or command prompt and run `ollama -v` to verify the installation.

### Step 2: Install and Set Up Neo4j

1.  Go to the Neo4j download page: `https://neo4j.com/download/`
2.  Download and install the Neo4j Desktop application for your operating system.
3.  Launch Neo4j Desktop. You will be prompted to create a new project and a local database instance.
4.  Create a new database instance. Make sure to note the username (`neo4j`) and the password you set.
5.  Start the database instance and open the Neo4j Browser.

## Python Project Setup and Execution

This guide provides the necessary terminal commands to set up a virtual environment, install dependencies, and run main Python script.

-----

### 1\. Create a Virtual Environment

First, run the following command to create a new virtual environment in project directory.

```bash
python -m venv <environment name>
```

### 2\. Activate the Virtual Environment

Before installing packages, you need to activate the virtual environment. The command depends on your operating system.

**On macOS or Linux:**

```bash
source <environment name>/bin/activate
```

**On Windows:**

```bash
<environment name>\Scripts\activate
```

-----

### 3\. Install Required Packages

With the virtual environment activated, you can now install all the packages listed in your `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4\. Run the Main Script

Once all the dependencies are installed, you can run your `main.py` script.

```bash
python main.py
```

### View file structure

```bash
tree /A /F > structure.txt
python main.py
```