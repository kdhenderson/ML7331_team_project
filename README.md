# ML7331_team_project
A semester-long group project for a Master of Data Science course in Machine Learning.

## Overview

This repository contains resources and initial exploratory tools to assist in selecting and analyzing a dataset for the project. The goal of this phase is to explore multiple datasets and determine the most suitable one for further analysis and modeling.

## Structure

- **`datasetProfiling.ipynb`**: A Jupyter notebook for exploring and profiling datasets.
- **HTML Profiling Reports**:
  - `airbnb.html`: Profiling report for the Airbnb dataset.
  - `bank-additional-full.html`: Profiling report for the Bank Marketing dataset.
  - `default.html`: Profiling report for the Credit Card Default dataset.
  - `diabetes.html`: Profiling report for the Diabetes dataset.
- **`requirements.txt`**: Lists the dependencies needed to run the Jupyter notebook and generate reports.
- **`data/`**: Directory containing raw datasets for initial exploration (if applicable).

## Setup Instructions

To set up the project locally and run the tools:

### 1. Clone the Repository
Clone the repository to your local machine:
```bash
git clone <repository_url>
cd ML7331_team_project
```

### 2. Set Up a Virtual Environment
Create and activate a virtual environment to manage dependencies:
- **On macOS/Linux**:
  ```bash
  python3 -m venv env
  source env/bin/activate
  ```
- **On Windows**:
  ```bash
  python -m venv env
  env\Scripts\activate
  ```

### 3. Install Dependencies
Install the required Python packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Open the Jupyter Notebook
Start Jupyter Notebook to explore datasets interactively:
```bash
jupyter notebook datasetProfiling.ipynb
```

### 5. View the HTML Reports
Open any of the HTML profiling reports in your browser to view pre-generated summaries for the datasets:
- `airbnb.html`
- `bank-additional-full.html`
- `default.html`
- `diabetes.html`

---

## Collaboration and Contributing

### Workflow
1. **Create a Branch for Your Work**:
   Create a new branch to keep your changes separate from the main branch:
   ```bash
   git checkout -b your-branch-name
   ```
   - Replace `your-branch-name` with a descriptive name, e.g., `dataset-exploration` or `update-readme`.

2. **Make Changes**:
   Edit files or add new ones as needed.

3. **Commit Your Changes**:
   Stage and commit your changes with a descriptive message:
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   ```

4. **Push Your Branch**:
   Push your branch to the repository on GitHub:
   ```bash
   git push origin your-branch-name
   ```

5. **Create a Pull Request**:
   Submit your changes for review:
   - Go to the repository on GitHub.
   - Click **Pull Requests** > **New Pull Request**.
   - Select your branch and describe the changes.
