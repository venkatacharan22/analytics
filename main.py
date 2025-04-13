from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import json
from typing import Optional, List, Dict
import io
from pydantic import BaseModel

app = FastAPI(
    title="Data Analysis API",
    description="API for analyzing data using AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
            # Calculate basic statistics
            stats = {}
            for column in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[column]):
                    stats[column] = {
                        'mean': float(self.df[column].mean()),
                        'min': float(self.df[column].min()),
                        'max': float(self.df[column].max()),
                        'sum': float(self.df[column].sum())
                    }
            
            # Generate response based on the question and statistics
            response = self.generate_analysis_response(question, stats)
            
            # Create metadata
            metadata = {
                "dataset_info": {
                    "rows": len(self.df),
                    "columns": len(self.df.columns),
                    "memory_usage": self.df.memory_usage(deep=True).sum(),
                },
                "statistics": stats
            }
            
            return AnalysisResponse(response=response, metadata=metadata)

        except Exception as e:
            print(f"Error in analyze_data: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def generate_analysis_response(self, question: str, stats: Dict) -> str:
        """Generate analysis response based on the question and statistics"""
        question = question.lower()
        
        if 'average' in question or 'mean' in question:
            response_parts = []
            for col, col_stats in stats.items():
                if 'mean' in col_stats:
                    response_parts.append(f"The average {col} is {col_stats['mean']:.2f}")
            
            if response_parts:
                response = ". ".join(response_parts) + "."
            else:
                response = "No numeric columns found to calculate averages."
                
        elif 'total' in question or 'sum' in question:
            response_parts = []
            for col, col_stats in stats.items():
                if 'sum' in col_stats:
                    response_parts.append(f"The total {col} is {col_stats['sum']:.2f}")
            
            if response_parts:
                response = ". ".join(response_parts) + "."
            else:
                response = "No numeric columns found to calculate totals."
                
        elif 'maximum' in question or 'highest' in question:
            response_parts = []
            for col, col_stats in stats.items():
                if 'max' in col_stats:
                    response_parts.append(f"The maximum {col} is {col_stats['max']:.2f}")
            
            if response_parts:
                response = ". ".join(response_parts) + "."
            else:
                response = "No numeric columns found to find maximum values."
                
        elif 'minimum' in question or 'lowest' in question:
            response_parts = []
            for col, col_stats in stats.items():
                if 'min' in col_stats:
                    response_parts.append(f"The minimum {col} is {col_stats['min']:.2f}")
            
            if response_parts:
                response = ". ".join(response_parts) + "."
            else:
                response = "No numeric columns found to find minimum values."
                
        else:
            # Default response with all statistics
            response_parts = []
            for col, col_stats in stats.items():
                stats_str = ", ".join([f"{k}: {v:.2f}" for k, v in col_stats.items()])
                response_parts.append(f"Statistics for {col}: {stats_str}")
            
            if response_parts:
                response = ". ".join(response_parts) + "."
            else:
                response = "No numeric data found in the dataset to analyze."
        
        return response

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
    """
    content = await file.read()
    return analyzer.load_data(content, file.filename)

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(question_request: QuestionRequest):
    """
    Analyze the uploaded data by asking questions
    """
    try:
        return analyzer.analyze_data(question_request.question)
    except Exception as e:
        print(f"Error in analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/info", response_model=Optional[DatasetInfo])
async def get_dataset_info():
    """
    Get information about the currently loaded dataset
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
