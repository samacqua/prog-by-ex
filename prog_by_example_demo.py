import ast
import os
import streamlit as st
from openai import OpenAI

OPENAI_KEY = os.environ.get("OPENAI_KEY")
if not OPENAI_KEY:
    raise ValueError("OPENAI_KEY is not set. Set it in api_keys.py")

client = OpenAI(api_key=OPENAI_KEY)

def is_valid_python(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def evaluate_assertions(code, assertions):
    try:
        exec(code)
        for assertion in assertions:
            exec(assertion)
        return True
    except AssertionError:
        return False
    except Exception:
        return False

def clean_llm_response(response):
    # Remove ```python from the start and ``` from the end
    cleaned = response.strip().removeprefix("```python").removesuffix("```")
    # Remove any remaining leading or trailing whitespace
    return cleaned.strip()

def generate_code(assertions, model, max_attempts, description=None, test_assertions=None, update_callback=None):
    generated_codes = []
    attempts = 0
    n = 1

    while attempts < max_attempts:
        if description:
            prompt = f"Write a Python function based on this description:\n{description}\n\n"
            prompt += "It should pass the following assertions:\n"
        else:
            prompt = "Write a Python function that passes these assertions:\n"
        prompt += "\n".join(assertions)
        
        prompt += "\n\nProvide only the function code without any explanations."

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a skilled Python programmer."},
                {"role": "user", "content": prompt}
            ],
            n=n
        )

        for choice in response.choices:
            code = choice.message.content.strip()
            cleaned_code = clean_llm_response(code)
            generated_codes.append(cleaned_code)

            if is_valid_python(cleaned_code) and evaluate_assertions(cleaned_code, assertions):
                if test_assertions and not evaluate_assertions(cleaned_code, test_assertions):
                    continue
                return cleaned_code, generated_codes

        attempts += n
        if update_callback:
            update_callback(attempts)
        n = min(n * 2, max_attempts - attempts)

    return None, generated_codes

DEFAULT_DESC = None
DEFAULT_ASSERTIONS = "\n".join([
    "assert function([0]) == 0",
    "assert function([5, 8, 2, 1]) == 5"
])
DEFAULT_TEST_ASSERTIONS = "\n".join([
    "assert function([]) is None",
    "assert function([7, 7, 2, 4 , 2, 2, 4, 4, 2]) == 7"
])

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SAMPLE_BUDGET = 20

def run_streamlit():
    st.title("LLM Code Synthesis")

    # Initialize session state
    if 'generated_codes' not in st.session_state:
        st.session_state.generated_codes = None

    # Pre-populate with CLI defaults

    description = st.text_area("Function Description (optional)", value=DEFAULT_DESC or "")
    assertions = st.text_area("Assertions (one per line)", value=DEFAULT_ASSERTIONS)
    test_assertions = st.text_area("Test Assertions (one per line, not included in prompt)", value=DEFAULT_TEST_ASSERTIONS)
    model = st.selectbox("OpenAI Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"], index=["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"].index(DEFAULT_MODEL))
    sample_budget = st.number_input("Sample Budget", min_value=1, value=DEFAULT_SAMPLE_BUDGET)

    assertions_list = [a.strip() for a in assertions.split('\n') if a.strip()]
    test_assertions_list = [a.strip() for a in test_assertions.split('\n') if a.strip()]

    if st.button("Generate Code"):
        if not assertions:
            st.error("Please provide at least one assertion.")
        else:
            with st.spinner("Generating code..."):
                
                sample_count = st.empty()
                
                def update_sample_count(n):
                    sample_count.text(f"Used {n} of {sample_budget} attempts")
                
                result, generated_codes = generate_code(assertions_list, model, sample_budget, description=description if description else None, test_assertions=test_assertions_list, update_callback=update_sample_count)
                
                # Store generated codes in session state
                st.session_state.generated_codes = generated_codes
                
                if result:
                    st.success("Code generated successfully!")
                    st.code(result, language="python")
                else:
                    st.error("Could not synthesize valid program")

    # Display pager for sampled programs
    if st.session_state.generated_codes:
        st.subheader("Sampled Programs")
        
        # Create a list of program options with pass/fail status
        program_options = []
        for i, code in enumerate(st.session_state.generated_codes, 1):
            passes_assertions = is_valid_python(code) and evaluate_assertions(code, assertions_list)
            passes_test_assertions = not test_assertions_list or evaluate_assertions(code, test_assertions_list)
            status = "✅" if passes_assertions and passes_test_assertions else "❌"
            program_options.append(f"{status} Program {i}")
        
        program_index = st.selectbox("Select program", range(len(program_options)), format_func=lambda x: program_options[x])
        st.code(st.session_state.generated_codes[program_index], language="python")
        
        # Display failed assertions for incorrect programs
        code = st.session_state.generated_codes[program_index]
        if not (is_valid_python(code) and evaluate_assertions(code, assertions_list) and (not test_assertions_list or evaluate_assertions(code, test_assertions_list))):
            st.subheader("Failed Assertions:")
            failed_assertions = []
            
            if not is_valid_python(code):
                failed_assertions.append("Invalid Python syntax")
            else:
                for assertion in assertions_list:
                    if not evaluate_assertions(code, [assertion]):
                        failed_assertions.append(assertion)
                for assertion in test_assertions_list:
                    if not evaluate_assertions(code, [assertion]):
                        failed_assertions.append(f"(Test) {assertion}")
            
            for assertion in failed_assertions:
                st.error(assertion)

def run_cli():
    result, generated_codes = generate_code(DEFAULT_ASSERTIONS, DEFAULT_MODEL, DEFAULT_SAMPLE_BUDGET, description=DEFAULT_DESC, test_assertions=DEFAULT_TEST_ASSERTIONS)

    if result:
        print("Code generated successfully:")
        print(result)
    else:
        print("Could not synthesize valid program")

    print("\nAll Generated Programs:")
    for i, code in enumerate(generated_codes, 1):
        print(f"\nAttempt {i}:")
        print(code)
        print("-" * 40)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ui", action="store_true")
    args = parser.parse_args()

    if args.no_ui:
        run_cli()
    else:
        run_streamlit()
