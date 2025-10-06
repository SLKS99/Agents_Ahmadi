from tools.socratic import clarify_question, socratic_pass, tot_generation

MAX_RUN_ATTEMPTS = 3

welcome = """
Welcome to Socratic-Thinking Hypothesis Generator!
Type 'q' anytime to stop the program. 
"""

def normalize(text):
    return text.lower().strip()

def prompt_initial_question():
    question = normalize(input("Ask a question: "))
    return question

def prompt_user_choice(first_question: str, second_question: str, third_question: str):
    choice = input(f"""
    Choose a reasoning that you best align with: 
    1. {first_question}
    2. {second_question}
    3. {third_question}
    """)
    return choice

print(welcome)
for attempt in range(1, MAX_RUN_ATTEMPTS + 1):
    question = prompt_initial_question()
    if question == "q":
        break
    else:
        clarified_question = clarify_question(question)
        print(clarified_question)
        socratic_questions = socratic_pass(clarified_question)
        questions = tot_generation(socratic_questions, clarified_question)
        first_question, second_question, third_question = questions
        prompt_user_choice(first_question, second_question, third_question)

# while (user_input := prompt_initial_question()) != "q":
#     question = clarify_question(user_input)
#     print(question)
