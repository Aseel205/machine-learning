# Midterm Project: Instructions for Students

## Your Goal

Your objective is to build the best possible machine learning model to classify data **provided specifically for you** (each student gets a unique dataset). You will train your model using a given training set `(X_train, y_train)` and then use it to make predictions on a test set `(X_test)`.

## Getting Started

1.  **Set Up Your Environment**:
    *   Make sure you have Python 3 installed.
    *   We recommend using a Python virtual environment to avoid conflicts with other projects.
        *   Create one: `python -m venv my_midterm_env`
        *   Activate it:
            *   Windows: `my_midterm_env\Scripts\activate`
            *   macOS/Linux: `source my_midterm_env/bin/activate`
    *   Install the necessary Python packages using the provided file:
        ```bash
        pip install -r requirements.txt
        ```

2.  **Find Your Data**:
    *   Inside the `datasets` folder, you will find a file named `{your_student_id}_train.npz`. 
    *   The folder will be published at the start of the exam via Moodle or a shared Drive link.
    *   Your  file contains: 

        *   `X_train`: Your training data features.
        *   `y_train`: Your training data labels.
        *   `X_test`: The test data features you need to make predictions on.
    *   Use **only this file** for your work.

## Your Task
Your task will involve modifying `midterm.py`

1.  **Set Your Student ID**:
    *   Open `midterm.py`.
    *   Find the line near the bottom that says `student_id = 12345`.
    *   Change `12345` to your **full student ID**.

2.  **Implement Your Model**:
    *   Locate the function `train_predict(X_train, y_train, X_test)`.
    *   **This is the ONLY part of the script you should modify.** 
	* Add your code between the `# --- Start of your code ---` and `# --- End of your code ---` comments.
    *   **Inside this function, you should**:
        *   Import any additional libraries you need at the top of the file.
        *   Implement your model training logic. This might include:
            *   Data preprocessing.
            *   Trying different classification models and methods.
        *   Train your chosen model using `X_train` and `y_train`.
        *   Use your trained model to make predictions on `X_test`.
        *   The function **must return** a **NumPy array** containing these predictions. The example code shows how to do this.
    *   **DO NOT** change any code outside this function or the student ID line. The surrounding code handles loading your data and saving your predictions correctly.

## Running Your Code & Submitting

1.  **Run the Script**:
    *   Once you have implemented your model in the `train_predict` function, open your terminal (make sure your virtual environment is active).
    *   Navigate to the project directory.
    *   Run the script:
        ```bash
        python midterm.py
        ```
    *   The script will:
        *   Load your data based on the `student_id` you set.
        *   Call your `train_predict` function.
        *   Save the predictions returned by your function.

2.  **Find Your Submission File**:
    *   If the script runs without errors, it will create a new directory named `test`.
    *   Inside the `test` directory, it will save a file named `{your_student_id}_test_predictions.npz`.

3.  **Submit**:
    *   **Submit ONLY the `{your_student_id}_test_predictions.npz` file.** Do not submit the `.py` files or anything else.

## How You will be graded

*   The exact grading details and criteria will be determined and published later. 
* The grade will be based on improvement over baseline Logistic Regression: `from sklearn.linear_model import LogisticRegression`

Good luck! Focus on understanding your data and experimenting with different modeling techniques within the `train_predict` function. 