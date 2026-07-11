import ollama
import time


def test_model(model_name):

    print("\n==============================")
    print(f"Testing Model: {model_name}")
    print("==============================")

    try:
        start_time = time.time()

        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": "Explain diabetes in simple words."
                }
            ],
            options={
                "temperature": 0.3
            }
        )

        end_time = time.time()

        answer = response["message"]["content"]

        print("\nResponse:")
        print(answer)

        print("\nPerformance Metrics:")
        print("------------------------------")
        print(f"Model: {model_name}")
        print(f"Response Time: {end_time-start_time:.2f} seconds")
        print(f"Answer Length: {len(answer)} characters")


    except Exception as e:
        print("\n❌ Error:")
        print(e)



# Test Qwen
test_model("qwen2.5:3b")


# Test DeepSeek
test_model("deepseek-r1:1.5b")
