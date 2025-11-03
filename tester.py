import openvino as ov
import openvino_genai as ov_genai
import numpy as np
import logging

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MODEL_PATH = r"C:\Users\Administrator\Downloads\openvino.genai\tools\llm_bench\models\gpu_models\google_gemma-2b-it_int4_cw"
DEVICE = "GPU"

def get_sequence_score(model, tokenizer, prompt, choice):
    """
    Processes a single full sentence and returns the single, incomplete score
    provided by the backend, logging the data mismatch.
    """

    full_text = prompt + choice
    try:
        # Robustly encode the text to get a list of token IDs
        raw_tokens = tokenizer.encode(full_text).input_ids.data.tolist()[0]
        seq_len = len(raw_tokens)
    except Exception as e:
        logging.error(f"  -> TOKENIZATION FAILED for '{full_text}'. Error: {e}")
        return -float('inf')

    # Prepare inputs for the model
    input_ids_np = np.array([raw_tokens], dtype=np.int64)
    tokenized_inputs = ov_genai.TokenizedInputs(
        input_ids=ov.Tensor(input_ids_np),
        attention_mask=ov.Tensor(np.ones_like(input_ids_np))
    )
    
    config = ov_genai.GenerationConfig(max_new_tokens=1, logprobs=1)
    result = model(inputs=tokenized_inputs, generation_config=config)
    
    logging.info(f"Processing sequence: '{full_text}' (Length: {seq_len})")
    
    if hasattr(result, 'scores') and result.scores:
        actual_len = len(result.scores)
        if actual_len < seq_len:
            logging.warning(f"  -> INCOMPLETE DATA: Expected {seq_len} scores, but received {actual_len}.")
        return result.scores[0]
    else:
        logging.error("  -> NO DATA: '.scores' attribute was not found.")
        return -float('inf')

if __name__ == "__main__":
    # --- Define the Comprehensive Test Suite ---
    TEST_SUITE = [
        {"prompt": "The capital of Italy is", "choices": [" Rome.", " Paris.", " Madrid.", " Athens."]},
        {"prompt": "The country famous for its pyramids is", "choices": [" Egypt.", " Mexico.", " Greece.", " Sudan."]},
        {"prompt": "Solid water is also known as", "choices": [" ice.", " steam.", " vapor.", " gas."]},
        {"prompt": "A common pet that barks is a", "choices": [" dog.", " cat.", " fish.", " bird."]},
        {"prompt": "The planet closest to the Sun is", "choices": [" Mercury.", " Venus.", " Mars.", " Earth."]},
        {"prompt": "To write with a pen, you need a piece of", "choices": [" paper.", " glass.", " metal.", " wood."]},
        {"prompt": "The opposite of 'hot' is", "choices": [" cold.", " wet.", " bright.", " loud."]},
        {"prompt": "The sun rises in the", "choices": [" east.", " west.", " north.", " south."]},
        {"prompt": "A year has twelve", "choices": [" months.", " days.", " weeks.", " hours."]},
        {"prompt": "The chemical symbol for water is", "choices": [" H2O.", " CO2.", " NaCl.", " O2."]},
    ]
    
    logging.info("--- Initializing Comprehensive Bug Demonstration  ---")
    model = ov_genai.LLMPipeline(MODEL_PATH, DEVICE.upper())
    tokenizer = model.get_tokenizer()

    # --- Main Test Loop ---
    for i, test_case in enumerate(TEST_SUITE):
        question_num = i + 1
        prompt = test_case["prompt"]
        choices = test_case["choices"]
        all_scores = {}

        print("\n" + "#"*70)
        print(f"     RUNNING TEST {question_num}/{len(TEST_SUITE)}: '{prompt}'")
        print("#"*70)
        
        # Loop through each choice for the current question
        for choice in choices:
            score = get_sequence_score(model, tokenizer, prompt, choice)
            all_scores[choice.strip()] = score

        # Print the results for the current question
        print("\n" + "-"*70)
        print(f"               Final (Faulty) Scores for Test {question_num}")
        print("-"*70)
        for choice, score in all_scores.items():
            print(f"Choice '{choice}': Received Score = {score:.4f}")
        
        if all_scores:
            winner = max(all_scores, key=all_scores.get)
            logging.info(f"\nModel's choice for Test {question_num} based on incomplete data: '{winner}'")
        
        print("#"*70)

    logging.critical("\n[OVERALL CONCLUSION] The test suite consistently demonstrates that the backend fails to provide complete, per-token scores, rendering log-likelihood evaluation methods invalid.")