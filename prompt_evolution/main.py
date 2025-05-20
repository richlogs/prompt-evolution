import pandas as pd
from pydantic import BaseModel
from models import Answer, Solver

DATA_DIR = "data/math_questions"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"

class ProblemRow(BaseModel):
    problem: str
    level: str
    type: str
    solution: str
    answer: int

class ExperimentResults(BaseModel):
    problems: list[ProblemRow]
    answers: list[Answer]

def batch_read_csv(file_path, batch_size=1):
    for chunk in pd.read_csv(file_path, chunksize=batch_size):
        models = []
        for _, row in chunk.iterrows():
            try:
                model = ProblemRow(
                    problem=row['problem'],
                    level=row['level'],
                    type=row['type'],
                    solution=row['solution'],
                    answer=row['answer']
                )
                models.append(model)
            except Exception as e:
                print(f"Row skipped due to error: {e}")
        yield models


original_developer_message = "<EMPTY_DEVELOPER_MESSAGE>"
experiment_results = []
for batch in batch_read_csv(TRAIN_PATH, batch_size=4):
    solver = Solver(developer_message=original_developer_message)
    answers = [solver.solve(row.problem) for row in batch]
    experiment_results.append(ExperimentResults(problems=batch, answers=answers))
    print(experiment_results)
    break


