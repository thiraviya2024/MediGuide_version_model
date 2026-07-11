import pandas as pd
from datetime import datetime

def log_performance(logs_list, model, question, response_time_ms, answer=""):
    """Log performance metrics for a query"""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "question": question[:100] + "..." if len(question) > 100 else question,
        "response_time_ms": response_time_ms,
        "answer_length": len(answer) if answer else 0,
        "status": "Success" if answer and "Please upload" not in str(answer) else "Error"
    }
    logs_list.append(entry)
    return entry


def get_metrics_df(logs_list):
    """Convert logs to DataFrame"""
    if not logs_list:
        return pd.DataFrame()
    return pd.DataFrame(logs_list)


def get_performance_summary(df):
    """Generate summary statistics"""
    if df.empty:
        return {
            "avg_time": 0,
            "total_queries": 0,
            "fastest_model": "N/A",
            "avg_answer_length": 0
        }
    
    avg_time = round(df["response_time_ms"].mean(), 2)
    total_queries = len(df)
    
    model_avg = df.groupby("model")["response_time_ms"].mean()
    fastest_model = model_avg.idxmin() if not model_avg.empty else "N/A"
    
    avg_answer_length = int(df["answer_length"].mean()) if "answer_length" in df.columns else 0
    
    return {
        "avg_time": avg_time,
        "total_queries": total_queries,
        "fastest_model": fastest_model,
        "avg_answer_length": avg_answer_length
    }