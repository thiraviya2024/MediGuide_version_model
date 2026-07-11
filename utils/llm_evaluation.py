import ollama
import time
import csv


models = [
    "qwen2.5:3b",
    "deepseek-r1:1.5b"
]


def ask_model(model, question):

    start = time.time()

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        options={
            "temperature":0.3
        }
    )

    end = time.time()

    answer = response["message"]["content"]

    return (
        answer,
        round(end-start,2),
        len(answer)
    )


# Read questions

with open("evaluation_questions.txt","r") as file:
    questions = file.readlines()


results=[]


for model in models:

    print("\nTesting:",model)

    for q in questions:

        question=q.strip()

        answer,time_taken,length = ask_model(
            model,
            question
        )

        print(question)
        print("Time:",time_taken)

        results.append([
            model,
            question,
            time_taken,
            length,
            answer
        ])


# Save results

with open(
    "llm_evaluation_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer=csv.writer(file)

    writer.writerow(
        [
            "Model",
            "Question",
            "Response Time",
            "Answer Length",
            "Response"
        ]
    )

    writer.writerows(results)


print("\nEvaluation completed!")
print("Saved: llm_evaluation_results.csv")
