import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(__file__).resolve().parent / "static" / "model"

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, not on every request
    with open(MODEL_DIR / "model.pickle", "rb") as f:
        state["model"] = pickle.load(f)
    with open(MODEL_DIR / "feature_columns.pickle", "rb") as f:
        state["feature_columns"] = pickle.load(f)

    cols = state["feature_columns"]
    state["job_titles"] = sorted(c.removeprefix("Job_Title_") for c in cols if c.startswith("Job_Title_"))
    state["degree_programs"] = sorted(c.removeprefix("Degree_Program_") for c in cols if c.startswith("Degree_Program_"))
    yield
    state.clear()


app = FastAPI(title="Career Match Model API", lifespan=lifespan)


class MatchRequest(BaseModel):
    Academic_Performance: float = Field(ge=0)
    Certifications_Count: int = Field(ge=0)
    Internship_Experience: float = Field(ge=0)
    Skill_Programming: float = Field(ge=0)
    Skill_Database_SQL: float = Field(ge=0)
    Skill_Version_Control_Git: float = Field(ge=0)
    Skill_Problem_Solving: float = Field(ge=0)
    Skill_Communication: float = Field(ge=0)
    Required_Programming: float = Field(ge=0)
    Required_Database_SQL: float = Field(ge=0)
    Required_Version_Control_Git: float = Field(ge=0)
    Required_Problem_Solving: float = Field(ge=0)
    Required_Communication: float = Field(ge=0)
    Min_Experience_Months: float = Field(ge=0)
    Job_Title: str
    Degree_Program: str


class MatchResponse(BaseModel):
    result: str
    prediction: int
    match_percentage: float


def preprocess_input(new_input: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    new_input = new_input.copy()
    new_input = new_input.drop(
        columns=["Student_ID", "Job_ID", "Gender", "Age", "Location", "Job_Match"],
        errors="ignore",
    )
    categorical = [c for c in ["Job_Title", "Degree_Program"] if c in new_input.columns]
    new_input = pd.get_dummies(new_input, columns=categorical, dtype=int)
    return new_input.reindex(columns=feature_columns, fill_value=0)


def predict_job_match(new_input: pd.DataFrame) -> dict:
    model = state["model"]
    processed = preprocess_input(new_input, state["feature_columns"])

    prediction = model.predict(processed)[0]
    idx = list(model.classes_).index(1)
    probability = model.predict_proba(processed)[0, idx]

    return {
        "result": "Job Match" if prediction == 1 else "Job Not Match",
        "prediction": int(prediction),
        "match_percentage": round(float(probability * 100), 2),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/options")
def options():
    """Valid category values, handy for React dropdowns."""
    return {"job_titles": state["job_titles"], "degree_programs": state["degree_programs"]}


@app.post("/predict", response_model=MatchResponse)
def predict(req: MatchRequest):
    # An unseen category would silently become all-zero dummies, so reject it
    if req.Job_Title not in state["job_titles"]:
        raise HTTPException(422, f"Unknown Job_Title '{req.Job_Title}'. See GET /options.")
    if req.Degree_Program not in state["degree_programs"]:
        raise HTTPException(422, f"Unknown Degree_Program '{req.Degree_Program}'. See GET /options.")

    df = pd.DataFrame([req.model_dump()])
    return predict_job_match(df)