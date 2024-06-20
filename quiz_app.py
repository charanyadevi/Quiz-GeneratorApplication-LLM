import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.runnables.base import RunnableLambda
import time

openai_api_key = os.getenv("OPENAI_API_KEY")


def create_the_quiz_prompt_template():
    template = """
    You are an expert quiz maker for {level} level. Let's think step by step and
    create a quiz with {num_questions} questions about the following concept/content: {quiz_context}.

    The format of the quiz could be one of the following:
    - Multiple-choice: 
    - Questions:
        <Question1>: <a. Answer 1>, <b. Answer 2>, <c. Answer 3>, <d. Answer 4>
        <Question2>: <a. Answer 1>, <b. Answer 2>, <c. Answer 3>, <d. Answer 4>
        ....
    - Answers:
        <Answer1>: <a|b|c|d>
        <Answer2>: <a|b|c|d>
        ....
        Example:
        - Questions:
        - 1. What is the time complexity of a binary search tree?
            a. O(n)
            b. O(log n)
            c. O(n^2)
            d. O(1)
        - Answers: 
            b
    """
    prompt = PromptTemplate.from_template(template)
    return prompt

def create_quiz_chain(prompt_template, llm):
    def create_formatted_prompt(data):
        level = data.get("level")
        num_questions = data.get("num_questions")
        quiz_type = data.get("quiz_type")
        quiz_context = data.get("quiz_context")
        return prompt_template.format(level=level, num_questions=num_questions, quiz_type=quiz_type, quiz_context=quiz_context)

    return RunnableLambda(create_formatted_prompt) | llm | StrOutputParser()

def split_questions_answers(quiz_response):
    questions, answers = quiz_response.split("Answers:")
    return questions.strip(), answers.strip()

def main():
    st.title("Interactive Quiz App")
    st.write("This app generates a quiz based on the user-selected topic and level.")

    prompt_template = create_the_quiz_prompt_template()
    llm = ChatOpenAI()
    chain = create_quiz_chain(prompt_template, llm)

    level = st.selectbox("Select the knowledge level", ["Basic", "Intermediate", "Advanced"])
    context = st.text_area("Enter context for the quiz ")
    num_questions = st.number_input("Enter the number of questions", min_value=1, max_value=50)
    quiz_type = st.selectbox("Select the quiz type", ["Multiple-Choice"])

    time_limit = st.number_input("Enter time limit (in seconds) for the quiz", min_value=1)

    if st.button("Generate Quiz"):
        quiz_response = chain.invoke({"level": level, "quiz_type": quiz_type, "num_questions": num_questions, "quiz_context": context})
        st.write("Quiz Generated!")
        questions, answers = split_questions_answers(quiz_response)
        st.session_state.answers = answers
        st.session_state.questions = questions
        st.session_state.start_time = time.time()

    # Display questions unconditionally
    if "questions" in st.session_state:
        st.write(st.session_state.questions)

    if "questions" in st.session_state:
        user_answers = {}
        for i in range(1, int(num_questions) + 1):
            if quiz_type == "Multiple-Choice":
                user_answers[i] = st.radio(f"Question {i}", ("a", "b", "c", "d"))

        st.session_state.user_answers = user_answers

    if st.button("Submit Answers"):
        if "user_answers" not in st.session_state:
            st.warning("Please generate a quiz and answer the questions first.")
        else:
            user_answers = st.session_state.user_answers
            correct_answers = st.session_state.answers.strip().split("\n")
            score = 0
            for i, answer in user_answers.items():
                correct_index = i - 1
                if correct_index < len(correct_answers): 
                    correct_answer_parts = correct_answers[correct_index].split(":")
                    if len(correct_answer_parts) > 1:
                        correct_answer = correct_answer_parts[1].strip()
                        if answer == correct_answer:
                            score += 1
                    else:
                        st.warning(f"Correct answer format is invalid for question {i}")
                else:
                    st.warning(f"Answer not found for question {i}")

            st.write(f"Your score: {score} out of {num_questions}")

    if st.button("Show Answers"):
        st.markdown(st.session_state.questions)
        st.write("----")
        st.markdown(st.session_state.answers)

    # Check time limit
    if "start_time" in st.session_state:
        elapsed_time = time.time() - st.session_state.start_time
        if elapsed_time > time_limit:
            st.warning("Time limit exceeded!")

if __name__ == "__main__":
    main()
