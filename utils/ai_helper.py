# utils/ai_helper.py
from langchain_community.llms import Ollama
from groq import Groq
import os

SYSTEM_PROMPT = """You are a helpful medical AI assistant. 
Answer ONLY based on the provided medical report context.
- Explain medical terms in simple language.
- Do NOT diagnose any disease.
- Do NOT prescribe medicines.
- Do NOT give medical advice.
- If information is not in the report, clearly say so.
- Always recommend consulting a qualified doctor.
Be accurate, helpful, and safe."""

def ask_ai(vectorstore, question: str, model_name: str):
    try:
        # FIXED: Use correct retriever method
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(question)   # ← This is the fix
        
        context = "\n\n".join([doc.page_content for doc in docs])

        full_prompt = f"{SYSTEM_PROMPT}\n\nContext from medical report:\n{context}\n\nQuestion: {question}"

        if model_name == "Qwen 2.5:3B":
            try:
                llm = Ollama(model="qwen2.5:3b", temperature=0.3)
                return llm.invoke(full_prompt).strip()
            except Exception as e:
                return f"⚠️ Ollama (Qwen) Error: {str(e)}"

        elif model_name == "DeepSeek R1:1.5B":
            try:
                llm = Ollama(model="deepseek-r1:1.5b", temperature=0.3)
                return llm.invoke(full_prompt).strip()
            except Exception as e:
                return f"⚠️ Ollama (DeepSeek) Error: {str(e)}"

        elif model_name == "Groq Llama 3.3 70B":
            try:
                client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": full_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=1024
                )
                return chat_completion.choices[0].message.content.strip()
            except Exception as e:
                return f"⚠️ Groq API Error: {str(e)}"

        return "Model not supported."

    except Exception as e:
        return f"Error: {str(e)}"
