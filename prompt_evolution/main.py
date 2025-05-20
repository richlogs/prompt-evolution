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

class Homework(BaseModel):
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
                pass
                # print(f"Row skipped due to error: {e}")
        yield models

def marker(homework: Homework):
    n = len(homework.problems)
    n_correct = 0
    for problem, answer in zip(homework.problems, homework.answers):
        if problem.answer == answer.answer:
            n_correct += 1

    if n == 0:
        return 0
    return n_correct / n

original_developer_message = "<EMPTY_DEVELOPER_MESSAGE>"
homework = []
for batch in batch_read_csv(TRAIN_PATH, batch_size=100):
    solver = Solver(developer_message=original_developer_message)
    answers = [solver.solve(row.problem) for row in batch]
    homework = Homework(problems=batch, answers=answers)

    marked_homework = marker(homework)
    print(f"Percentage correct: {round(marked_homework*100, 2)}%")
    break


