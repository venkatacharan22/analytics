from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from dotenv import load_dotenv
import json
from typing import Optional, List, Dict
import io
from pydantic import BaseModel
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Verify API key exists
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")

app = FastAPI(
    title="Data Analysis API",
    description="API for analyzing data using AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class AnalysisResponse(BaseModel):
    response: str
    metadata: Dict

class DatasetInfo(BaseModel):
    columns: List[str]
    row_count: int
    column_types: Dict[str, str]
    sample_data: List[Dict]

class DataAnalyzer:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.client = InferenceClient(token=HUGGINGFACE_API_KEY)
        self.model = "mistralai/Mistral-7B-Instruct-v0.2"
    
    def load_data(self, file_content: bytes, filename: str) -> DatasetInfo:
        try:
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension == '.csv':
                self.df = pd.read_csv(io.BytesIO(file_content))
            elif file_extension in ['.xlsx', '.xls']:
                self.df = pd.read_excel(io.BytesIO(file_content))
            else:
                raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
            
            # Create dataset info
            info = DatasetInfo(
                columns=self.df.columns.tolist(),
                row_count=len(self.df),
                column_types={col: str(dtype) for col, dtype in self.df.dtypes.items()},
                sample_data=self.df.head(5).to_dict(orient='records')
            )
            
            return info
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    def analyze_data(self, question: str) -> AnalysisResponse:
        if self.df is None:
            raise HTTPException(status_code=400, detail="No data loaded. Please upload a file first.")
        
        try:
            # Get basic statistics
            stats = self.df.describe().to_dict()
            columns = ", ".join(self.df.columns.tolist())
            sample = self.df.head(5).to_dict(orient='records')
            
            # Create prompt for the model
            prompt = f"""<s>[INST] I have a dataset with the following columns: {columns}

Here's a sample of the data:
{json.dumps(sample, indent=2)}

And here are some basic statistics:
{json.dumps(stats, indent=2)}

Question: {question}

Please analyze this data and provide a clear, concise answer. Focus on the specific question asked and use numbers from the data when relevant.[/INST]</s>"""

            # Get response from Hugging Face model
            response = self.client.text_generation(
                prompt,
                model=self.model,
                max_new_tokens=500,
                temperature=0.7,
                repetition_penalty=1.2,
                do_sample=True,
                return_full_text=False
            )
            
            if not response or not isinstance(response, str):
                raise ValueError("Invalid response from model")
                
            # Clean up the response
            response = response.strip()
            if response.startswith("[/INST]"):
                response = response[len("[/INST]"):].strip()
            
            # Create metadata
            metadata = {
                "dataset_info": {
                    "rows": len(self.df),
                    "columns": len(self.df.columns),
                    "memory_usage": self.df.memory_usage(deep=True).sum(),
                },
                "analysis_info": {
                    "model": self.model,
                    "token_count": len(response.split())
                }
            }
            
            return AnalysisResponse(response=response, metadata=metadata)

        except Exception as e:
            print(f"Error in analyze_data: {str(e)}")  # Debug log
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Initialize analyzer
analyzer = DataAnalyzer()

@app.get("/")
async def root():
    """Get API information and status"""
    return {
        "name": "Data Analysis API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "POST /upload": "Upload a CSV or Excel file for analysis",
            "POST /analyze": "Ask questions about the uploaded data",
            "GET /dataset/info": "Get information about the currently loaded dataset"
        }
    }

@app.post("/upload", response_model=DatasetInfo)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV or Excel file for analysis
    
    - Accepts .csv, .xlsx, or .xls files
    - Returns information about the uploaded dataset
    - The file will be loaded into memory for subsequent analysis
    """
    content = await file.read()
    return analyzer.load_data(content, file.filename)

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(question_request: QuestionRequest):
    """
    Analyze the uploaded data by asking questions
    
    - Requires a dataset to be uploaded first
    - Returns AI-generated analysis and metadata
    - Uses Mistral-7B model for analysis
    """
    try:
        return analyzer.analyze_data(question_request.question)
    except Exception as e:
        print(f"Error in analyze endpoint: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/info", response_model=Optional[DatasetInfo])
async def get_dataset_info():
    """
    Get information about the currently loaded dataset
    
    - Returns None if no dataset is loaded
    - Includes column names, types, and sample data
    """
    if analyzer.df is None:
        return None
    
    return DatasetInfo(
        columns=analyzer.df.columns.tolist(),
        row_count=len(analyzer.df),
        column_types={col: str(dtype) for col, dtype in analyzer.df.dtypes.items()},
        sample_data=analyzer.df.head(5).to_dict(orient='records')
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
