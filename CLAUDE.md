# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## High-Level Code Architecture and Structure

This project is a Flask web application (`app.py`) designed for an election system with a face recognition component. It allows users to register with their faces, log in via face verification, and then cast a vote. An admin interface is also provided for managing candidates, resetting votes, normalizing images, and retraining the face recognition model.

The application uses the following key technologies and components:

-   **Flask**: The web framework for handling routes, requests, and rendering HTML templates.
-   **OpenCV (`cv2`)**: Used for face detection (using Haar Cascades and optionally YOLOv8-face), image processing, and normalization.
-   **TensorFlow/Keras**: For building, training, and loading the Convolutional Neural Network (CNN) model used for face recognition.
-   **Pandas**: For managing user registration data (`users.csv`) and voting data (`votes.csv`).
-   **`ultralytics` (YOLOv8-face)**: An optional, more advanced face detection model that the application attempts to load first, falling back to Haar Cascades if not available or if loading fails.
-   **`python-dotenv`**: For loading environment variables (e.g., `YOLO_FACE_MODEL_PATH`).
-   **`pyngrok`**: (Commented out) Potentially used for exposing the local development server to the internet for testing.

**Key Directories and Files:**

-   `app.py`: The main Flask application file containing all routes, business logic, and machine learning model interactions.
-   `Face_Recog_App/`: This directory appears to be the root for application-specific assets.
    -   `Face_Recog_App/static/uploads/UserImages/`: Stores user-uploaded images, organized into subfolders per user.
    -   `Face_Recog_App/model/`: Stores the trained face recognition model (`face_cnn_model.keras`), label mappings (`label_map.json`), and training logs (`train.log`). The YOLO face model (`yolov8n-face.pt`) is also expected here.
-   `templates/`: Contains HTML templates for the web interface (e.g., `index.html`, `register.html`, `vote.html`, `results.html`, `admin_login.html`, `admin_dashboard.html`).
-   `requirements.txt`: Lists all Python dependencies.
-   `haarcascade_frontalface_default.xml`: The XML file for OpenCV's Haar Cascade face detector.

**Data Flow and Core Logic:**

1.  **User Registration (`/register`):**
    *   Users submit their name, surname, student ID, and multiple facial images.
    *   Images are processed: faces are detected, cropped, resized to 100x100 pixels, and normalized for camera variations (using `normalize_image_for_camera_variation`).
    *   Processed images are saved in `Face_Recog_App/static/uploads/UserImages/{name}_{surname}/`.
    *   User metadata (name, surname, student ID, folder path) is saved to `Face_Recog_App/static/uploads/UserImages/users.csv`.
    *   After registration, the CNN face recognition model is immediately retrained using all available user images.
2.  **Login/Face Verification (`/`):**
    *   Users submit their student ID and a live camera image.
    *   The system loads the trained `face_cnn_model.keras` and `label_map.json`. If the model cannot be loaded, it attempts to retrain it from existing images.
    *   The submitted image's face is detected, cropped, resized, normalized, and then fed into the CNN model for prediction.
    *   If the predicted face matches the registered student ID, the user is authenticated.
3.  **Voting (`/vote`, `/submit_vote`):**
    *   Authenticated users can view candidates and cast a single vote.
    *   Voting status is tracked in `Face_Recog_App/static/uploads/UserImages/users.csv` (`has_voted` column).
    *   Vote counts are updated in `Face_Recog_App/static/uploads/UserImages/votes.csv`.
4.  **Admin Functions (`/admin`, `/admin/dashboard`, etc.):**
    *   Requires a hardcoded password (`ADMIN_PASSWORD`).
    *   Admins can add/delete candidates, reset all votes, trigger image normalization for all existing images, and manually retrain the face recognition model.

## Commands

### Setup and Installation

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Run the Flask development server:**
    ```bash
    python app.py
    ```
    The application will typically run on `http://127.0.0.1:5000/`.

### Development and Maintenance

1.  **Retrain the face recognition model (via admin interface):**
    Navigate to `/admin/dashboard` in the browser and use the "Retrain Model" button. This triggers the `retrain_face_model_from_existing_images()` function.
2.  **Normalize all existing user images (via admin interface):**
    Navigate to `/admin/dashboard` in the browser and use the "Normalize Images" button. This triggers the `normalize_all_existing_images()` function.
3.  **Reset votes and user voting status (via admin interface):**
    Navigate to `/admin/dashboard` in the browser and use the "Reset Votes" button.
