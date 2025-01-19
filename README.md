# ML7331_team_project
A semester-long group project for a Master of Data Science course in Machine Learning.

## Overview

This repository contains jupyter notebooks and data files for a project aimed at predicting the hospital re-admittance of diabetes patients. The goal of each phase are as follows:
- Intro Phase: Identify a suitable dataset with good documentation, 30K+ records, and a combination of 10+ continuous and categorical features.  
- Phase One: Define the objective of the dataset, provide summary statistics, identify missing and duplicate data, identify outliers, and provide exploratory visualizations of the variables.  

## Structure

- **`LabOne.ipynb`**: A Jupyter notebook for identifying project goals and summarizing the data.
- **`data/`**: Directory containing the raw data.

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
Install the required Python packages (numpy, pandas, matplotlib, sklearn).

### 4. Open the Jupyter Notebook
Start Jupyter Notebook to explore datasets interactively:
```bash
jupyter notebook LabOne.ipynb
```

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
   git add <file>
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
