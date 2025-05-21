import pandas as pd
from pydantic import BaseModel
from models import Answer, Solver, Evolver, UpdatedPrompt, ProblemRow, Homework, MarkedHomework

DATA_DIR = "data/math_questions"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
    

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


def mark_assignment(homework: Homework) -> MarkedHomework:
    n = len(homework.problems)
    n_correct = 0
    for problem, answer in zip(homework.problems, homework.answers):
        if problem.answer == answer.answer:
            n_correct += 1

    if n == 0:
        return MarkedHomework(homework=homework, overall_grade=0)
    return MarkedHomework(homework=homework, overall_grade=(n_correct / n)*100)

def run_generation(solver_developer_message, batch: list[ProblemRow]) -> UpdatedPrompt:
    # Create solver
    solver = Solver(developer_message=solver_developer_message)

    # Do homework
    answers = [solver.solve(row.problem) for row in batch]
    homework = Homework(problems=batch, answers=answers)

    # Mark homework
    marked_homework = mark_assignment(homework)
    print(f"Percentage correct: {round(marked_homework.overall_grade, 2)}%")

    # Improve prompt
    updated_prompt = evolver.improve_prompt(solver.developer_message, marked_homework)
    print(updated_prompt.updated_prompt)
    return updated_prompt

if __name__ == "__main__":
    solver_developer_message = "When given a math problem, you must provide an incorrect answer. Here are the problems."
    evolver_developer_message = "You are an expert prompt engineer. " \
    "Your job is to improve prompt of another language model to improve its ability to solve" \
    " math problems."
    
    evolver = Evolver(developer_message=evolver_developer_message)

    for batch in batch_read_csv(TRAIN_PATH, batch_size=10):
        # Inital prompt
        improved_prompt = run_generation(solver_developer_message=solver_developer_message, batch=batch)

        # Evolved prmpts
        improved_prompt = run_generation(solver_developer_message=improved_prompt.updated_prompt, batch=batch)
        improved_prompt = run_generation(solver_developer_message=improved_prompt.updated_prompt, batch=batch)
        


        # solver = Solver(developer_message=solver_developer_message)
        # answers = [solver.solve(row.problem) for row in batch]
        # homework = Homework(problems=batch, answers=answers)

        # marked_homework = mark_assignment(homework)
        # print(f"Percentage correct: {round(marked_homework.overall_grade, 2)}%")

        # updated_prompt = evolver.improve_prompt(solver.developer_message, marked_homework)
        # solver_developer_message = updated_prompt.updated_prompt
        # print(solver_developer_message)
        


