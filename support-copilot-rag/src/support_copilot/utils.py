def format_answer(answer, sources):
    formatted_answer = f"Answer: {answer}\n\nSources:\n"
    for source in sources:
        formatted_answer += f"- {source}\n"
    return formatted_answer

def handle_error(error_message):
    return f"Error: {error_message}"