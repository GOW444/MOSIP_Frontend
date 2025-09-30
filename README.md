# MOSIP OCR: Text Extraction and Verification

This repository contains a powerful Optical Character Recognition (OCR) solution designed to extract, analyze, and verify text from document images. It includes a robust API for backend processing and a user-friendly Streamlit application for easy interaction and testing.

## Table of Contents

  - [Problem Statement](https://www.google.com/search?q=%23problem-statement)
  - [Our Solution](https://www.google.com/search?q=%23our-solution)
  - [Features](https://www.google.com/search?q=%23features)
  - [Getting Started](https://www.google.com/search?q=%23getting-started)
      - [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      - [Installation](https://www.google.com/search?q=%23installation)
  - [Usage](https://www.google.com/search?q=%23usage)
      - [Running the Application](https://www.google.com/search?q=%23running-the-application)
      - [Using the API](https://www.google.com/search?q=%23using-the-api)
  - [API Endpoints](https://www.google.com/search?q=%23api-endpoints)
      - [Health Check](https://www.google.com/search?q=%23health-check)
      - [Extract Text](https://www.google.com/search?q=%23extract-text)
      - [Verify Text](https://www.google.com/search?q=%23verify-text)
  - [Technology Stack](https://www.google.com/search?q=%23technology-stack)
  - [Project Structure](https://www.google.com/search?q=%23project-structure)

## Problem Statement

The core task is to develop an innovative OCR-driven solution that seamlessly extracts text from scanned documents, intelligently auto-fills digital forms, and accurately verifies the extracted data against the original source. This enhances both the reliability and efficiency of data entry and validation processes.

The mandatory requirements for this project included:

  - **Two distinct APIs:** One for text extraction and another for data verification.
  - **Support for English:** The OCR must handle at least one Latin-based language.
  - **Compatibility with sample documents:** The solution needs to work with various document types like ID cards and forms.

## Our Solution

This project successfully addresses the problem statement by providing a comprehensive suite of tools for OCR-based text extraction and verification. The solution is built around a powerful backend API and an intuitive Streamlit application, offering a seamless user experience for document processing.

The key components of our solution are:

  - A **FastAPI-powered server** that exposes robust endpoints for text extraction and verification, supporting both printed and handwritten text.
  - A **Streamlit web application** that provides an interactive interface for uploading documents, visualizing results, and testing the API's capabilities in real-time.
  - Advanced OCR processing using **EasyOCR and TrOCR** to ensure high accuracy for both printed and handwritten text.

## Features

  - **Dual OCR Engines:** Utilizes **EasyOCR** for printed text and **TrOCR** for handwritten text, ensuring high accuracy across different document types.
  - **Text Extraction & Verification:** Offers separate functionalities for extracting text and verifying it against expected output.
  - **Confidence Scoring:** Provides confidence levels for each text region, allowing for more reliable data processing.
  - **Interactive UI:** A user-friendly Streamlit application for easy document uploading, result visualization, and real-time verification.
  - **Comprehensive API:** A well-documented API with clear endpoints for integration into other systems.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

Make sure you have Python 3.8+ and pip installed.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Application

To start the Streamlit application, run the following command in your terminal:

```bash
streamlit run app.py
```

This will open the application in your default web browser, where you can start uploading documents and testing the OCR functionalities.

### Using the API

The API provides a programmatic way to interact with the OCR service. For detailed information on how to use the API, please refer to the `API Documentation` page in the Streamlit application, which corresponds to the `pages/02_API_Documentation.py` file.

## API Endpoints

The base URL for the API is `http://localhost:8000`.

### Health Check

  - **Endpoint:** `GET /api/health`
  - **Description:** Checks the status of the API and loaded models.

### Extract Text

  - **Endpoint:** `POST /api/extract`
  - **Description:** Extracts text from an uploaded image.
  - **Parameters:**
      - `file` (file): The image file.
      - `text_type` (string): "printed" or "handwritten".
      - `min_confidence` (float): Minimum confidence threshold.

### Verify Text

  - **Endpoint:** `POST /api/verify`
  - **Description:** Extracts text and compares it against an expected string.
  - **Parameters:**
      - `file` (file): The image file.
      - `expected_text` (string): The text to verify against.
      - `text_type` (string): "printed" or "handwritten".
      - `min_confidence` (float): Minimum confidence threshold.

## Technology Stack

  - **Backend:** FastAPI, EasyOCR, TrOCR (Transformers)
  - **Frontend:** Streamlit
  - **Core Libraries:** PyTorch, Pillow, OpenCV

## Project Structure

```
├── api/
│   ├── __init__.py
│   ├── ocr_api.py
│   └── requirements_api.text
├── pages/
│   └── 02_API_Documentation.py
├── .devcontainer/
│   └── devcontainer.json
├── app.py
└── requirements.txt
```
