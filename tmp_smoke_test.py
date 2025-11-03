import openvino as ov
import openvino_genai as ov_genai
import numpy as np
import logging

# --- Setup ---
# Configure logging for clear, professional output.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Define the model path and device for the test.
MODEL_PATH = r"C:\Users\Administrator\Downloads\openvino.genai\tools\llm_bench\models\gpu_models\google_gemma-2b-it_int4_cw"
DEVICE = "GPU"

def analyze_sequence_log_likelihood(model, tokenizer, prompt, choice):
    """
    Demonstrates the failure of the "Complete Processing" (full log-likelihood) method.
    This test verifies if the backend returns a score for each token in the input sequence.
    """
    logging.info("\n" + "="*70)
    logging.info("     ANALYSIS 1: Sequence Log-Likelihood Method Verification")
    logging.info("="*70)
    
    full_text = prompt + choice
    raw_tokens = tokenizer.encode(full_text).input_ids.data.tolist()[0]
    seq_len = len(raw_tokens)
    
    # Prepare the inputs in the format required by the OpenVINO LLMPipeline.
    input_ids_np = np.array([raw_tokens], dtype=np.int64)
    tokenized_inputs = ov_genai.TokenizedInputs(
        input_ids=ov.Tensor(input_ids_np),
        attention_mask=ov.Tensor(np.ones_like(input_ids_np))
    )
    
    # Request log probabilities from the backend.
    config = ov_genai.GenerationConfig(max_new_tokens=1, logprobs=1)
    result = model(inputs=tokenized_inputs, generation_config=config)
    
    logging.info(f"Input Sequence: '{full_text}'")
    logging.info(f"Expected Number of Score Entries (Sequence Length): {seq_len}")
    
    if hasattr(result, 'scores') and result.scores:
        scores = result.scores
        logging.info(f"Actual Number of Score Entries Returned: {len(scores)}")
        logging.info(f"Raw Content of '.scores' Attribute: {scores}")

        print("\n--- Per-Token Score Availability Analysis ---")
        for i, token_id in enumerate(raw_tokens):
            token_str = tokenizer.decode([token_id])
            if i < len(scores):
                score_value = scores[i]
                print(f"Token {i+1:>2} ('{token_str}'):  Score Data: AVAILABLE ({score_value})")
            else:
                print(f"Token {i+1:>2} ('{token_str}'):  Score Data: MISSING")
        
        logging.error("\n[CONCLUSION] The '.scores' attribute is incomplete. It provides only one entry, not one for each token in the sequence. This makes calculating the full sequence log-likelihood impossible.")
    else:
        logging.error("[CONCLUSION] The '.scores' attribute was not found on the result object.")

def analyze_next_token_probability(model, tokenizer, prompt, choices):
    """
    Demonstrates the failure of the "Individual Token" (first-token) method.
    This test verifies if the backend returns a usable probability distribution.
    """
    logging.info("\n" + "="*70)
    logging.info("     ANALYSIS 2: Next-Token Probability Method Verification")
    logging.info("="*70)
    
    formatted_prompt = f"{prompt}\n" + "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)]) + "\nAnswer:"
    
    raw_tokens = tokenizer.encode(formatted_prompt).input_ids.data.tolist()[0]
    input_ids_np = np.array([raw_tokens], dtype=np.int64)
    tokenized_inputs = ov_genai.TokenizedInputs(
        input_ids=ov.Tensor(input_ids_np),
        attention_mask=ov.Tensor(np.ones_like(input_ids_np))
    )
    
    config = ov_genai.GenerationConfig(max_new_tokens=1, logprobs=1)
    result = model(inputs=tokenized_inputs, generation_config=config)

    logging.info("Requirement: To compare choices ('A', 'B', 'C', etc.), a probability distribution over the vocabulary for the next token is needed.")
    
    if hasattr(result, 'scores') and result.scores:
        logging.info(f"Observed Backend Output: A single float value in '.scores': {result.scores[0]}")
        logging.critical("[CONCLUSION] This method is not feasible. The backend returns a single aggregate score, not a probability distribution that would allow for the comparison of individual next-token probabilities.")
    else:
        logging.error("[CONCLUSION] The '.scores' attribute was not found on the result object.")

if __name__ == "__main__":
    logging.info("--- Initializing OpenVINO Log-Probability Verification Script ---")
    model = ov_genai.LLMPipeline(MODEL_PATH, DEVICE.upper())
    tokenizer = model.get_tokenizer()

    # Define the test case
    question_prompt = "The capital of Italy is"
    answer_choices = [" Rome.", " Paris.", " Madrid.", " Athens."]
    
    # Run the analysis for the "Complete Processing" method on one example
    analyze_sequence_log_likelihood(model, tokenizer, question_prompt, answer_choices[0])
    
    # Run the analysis for the "Individual Token" comparison method
    analyze_next_token_probability(model, tokenizer, question_prompt, answer_choices)

    logging.info("\n--- Verification Script Complete ---")




