# DSA-Llama-3.2-FineTuning

This project is a complete pipeline for fine-tuning a `Llama-3.2-1B` model on a custom dataset of LeetCode problems and their C++ solutions. The process involves scraping data, cleaning it, and then using it for training.

---

### `datascraping.py`

This script kicks off the data gathering. It uses Selenium to navigate a list of LeetCode URLs, scrapes the core problem description from each page, and saves the results incrementally to a JSON file to prevent data loss.

### `questiondatahandling.py`

This script cleans the raw scraped text. It finds any image URLs in the problem descriptions, uses the Google Gemini API to get a text description of the image, and replaces the link with the text, making the dataset model-friendly.

### `finaldataprocessing.py`

This is the final assembly step for the dataset. It pairs the cleaned problem descriptions with their corresponding C++ solution files from a local directory, creating the final Question/Answer dataset and skipping any problems that are missing a solution file.

### `data/leetcodeinstruction.json`

This is the final, polished dataset ready for training. It's a clean JSON array where each entry contains a "Question" field with the problem text and an "Answer" field with the C++ code solution.

### `train.py`

#### Description
This script handles the main event: fine-tuning the Llama 3.2 model. It takes our custom-prepared LeetCode dataset and uses it to teach the model how to solve programming problems, creating a specialized version with enhanced coding logic capabilities.

#### How It's Done:
* The final JSON dataset is loaded using the Hugging Face `datasets` library.
* A custom function formats each Question/Answer pair into the specific chat template required by Llama 3, including `system`, `user`, and `assistant` roles.
* It uses **4-bit quantization** (via `bitsandbytes`) to significantly reduce the model's memory usage, allowing it to be trained on consumer-grade hardware.
* A **LoRA (Low-Rank Adaptation)** configuration is applied, which freezes the base model and only trains a small "adapter" layer, making the fine-tuning process highly efficient.
* The `SFTTrainer` from the TRL library manages the entire supervised fine-tuning loop.
* Finally, the trained LoRA adapter is saved, which can be merged with the base model later for inference.

## How to Run This Project

1.  **Setup**:
    * Create a Python virtual environment and install the required packages (you'd list them in a `requirements.txt` file).
    * Create a `.env` file and add your Google Gemini API keys.
    * Make sure you have a `leetcode_links.csv` file with the URLs you want to scrape.
    * Place all your C++ solution files in the `C++/` directory, ensuring their filenames match the LeetCode problem titles (e.g., `two-sum.cpp`).

2.  **Execution Order**:
    * Run `python datascraping.py` to get the initial raw data.
    * Run `python questiondatahandling.py` to clean the text and handle images.
    * Run `python finaldataprocessing.py` to create the final Q&A dataset.
    * Run `python train.py` to start the fine-tuning process.

And that's it! You'll have your very own LeetCode-savvy language model.