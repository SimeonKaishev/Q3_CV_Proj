# Album Art generation
## Project Setup Instructions

This guide will walk you through the steps to set up the environment, install dependencies, and run the Streamlit app for the CV project.

### Prerequisites

Ensure that you have the following installed on your machine:
- **[Conda](https://docs.conda.io/projects/conda/en/latest/index.html)** (either via Anaconda or Miniconda)


### 1. Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone https://github.com/SimeonKaishev/Q3_CV_Proj.git
cd cv-project
```

### 2. Create the Conda Environment

The project uses a Conda environment. To create the environment and install dependencies, use the provided environment.yml file:

Open the project in VS Code and run the following command int the terminal:
```bash
conda env create -f environment.yml
```
This will create a new Conda environment with the name album-cover-generator and install all the required dependencies (e.g., Streamlit, OpenCV, numpy, etc.).

### 3. Activate the Conda Environment
Once the environment is created, activate it using the following command:
```bash
conda activate album-cover-generator
```
### 4. Start the application
Now that the environment is activated, you should be able to run the app with the following command:
```bash
streamlit run GUI/app.py
```
This will launch the Streamlit app, and you'll be provided with a local URL (usually http://localhost:8501) where you can view the application in your browser.

### Updating the Conda Environment
If you need to add new libraries or update existing dependencies in the Conda environment, follow these steps:
1. First install the library you'd like to use:

    To install a new library using Conda (for example, pandas), run:
    ```bash
    conda install pandas
    ```
    Alternatively you can (and prob should) use pip to install it by running:
    ```bash
    pip install some-library
    ```
2.  Export the Updated Environment
    
    After adding or updating libraries, you need to update the environment.yml file to reflect the changes.

    To export the updated environment (including the new dependencies) into the environment.yml file, run:
     ```bash
    conda env export > environment.yml
    ```

3. Share the Updated environment.yml File:
    
    Finally push the updataed environment file to the git, other can now update their environments by running:
    ```bash
    conda env update -f environment.yml
    ```