# Data Analysis API

A RESTful API that uses AI to analyze data through natural conversation. Upload your CSV or Excel files and ask questions about your data to get AI-powered insights.

## Features

- File upload support (CSV and Excel)
- AI-powered data analysis using Hugging Face's Mistral model
- RESTful API endpoints
- Detailed dataset information
- Comprehensive analysis metadata

## API Endpoints

### GET /
Get API information and status.

**Response:**
```json
{
    "name": "Data Analysis API",
    "version": "1.0.0",
    "status": "active",
    "endpoints": {
        "POST /upload": "Upload a CSV or Excel file for analysis",
        "POST /analyze": "Ask questions about the uploaded data",
        "GET /dataset/info": "Get information about the currently loaded dataset"
    }
}
```

### POST /upload
Upload a CSV or Excel file for analysis.

**Request:**
- Content-Type: multipart/form-data
- Body: file (CSV or Excel file)

**Response:**
```json
{
    "columns": ["column1", "column2", ...],
    "row_count": 1000,
    "column_types": {
        "column1": "int64",
        "column2": "object",
        ...
    },
    "sample_data": [
        {"column1": "value1", "column2": "value2"},
        ...
    ]
}
```

### POST /analyze
Ask questions about the uploaded data.

**Request:**
```json
{
    "question": "What are the highest values in the dataset?"
}
```

**Response:**
```json
{
    "response": "Based on the analysis...",
    "metadata": {
        "dataset_info": {
            "rows": 1000,
            "columns": 5,
            "memory_usage": 40000
        },
        "analysis_info": {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "token_count": 150
        }
    }
}
```

### GET /dataset/info
Get information about the currently loaded dataset.

**Response:**
```json
{
    "columns": ["column1", "column2", ...],
    "row_count": 1000,
    "column_types": {
        "column1": "int64",
        "column2": "object",
        ...
    },
    "sample_data": [
        {"column1": "value1", "column2": "value2"},
        ...
    ]
}
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/venkatacharan22/analytics.git
cd analytics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file and add your Hugging Face API key:
```
HUGGINGFACE_API_KEY=your_token_here
```

4. Run the application:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## Example Usage with cURL

1. Upload a file:
```bash
curl -X POST http://localhost:8000/upload \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data.csv"
```

2. Analyze data:
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the trends in this data?"}'
```

3. Get dataset info:
```bash
curl http://localhost:8000/dataset/info
```

## Deployment

This application is configured for deployment on Render. The `render.yaml` file contains the necessary configuration.

Remember to set the `HUGGINGFACE_API_KEY` environment variable in your Render dashboard.
