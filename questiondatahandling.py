import os
import json
import re
import requests
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm import tqdm

# --- Configuration ---

# Load environment variables from the .env file
load_dotenv()

# List of API keys to rotate through when quota is exceeded
API_KEYS = [
    "xxx",
    "xx",
    "x-x",
    "x-x",
]

# Track current API key index
current_key_index = 0

# Set the path for your input and output JSON files
INPUT_JSON_PATH = "/home/deadsec/Desktop/ML/FIne tuning/leetcode_problems_selenium.json"
OUTPUT_JSON_PATH = "/home/deadsec/Desktop/ML/FIne tuning/leetcode_problems_processed.json"

# Tuple of common image file extensions to check against
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg')

# The specific prompt for Gemini to get a clean, insertable description
GEMINI_PROMPT = """Analyze the attached image, which is part of a programming problem description. Provide a concise, plain-text description or an ASCII art representation suitable for replacing the image in the problem text. Do not include any introductory or concluding phrases like "Here is the description:". Your output should be directly insertable into the text."""

# --- Helper Functions ---

def get_next_api_key():
    """
    Get the next API key from the list, cycling back to the start if needed.
    """
    global current_key_index
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    return API_KEYS[current_key_index]

def create_client_with_current_key():
    """
    Create a new client with the current API key.
    """
    return genai.Client(api_key=API_KEYS[current_key_index])

def fetch_image_data(url: str) -> dict | None:
    """
    Downloads an image and returns its content and MIME type.
    Returns None if the download fails.
    """
    try:
        print(f"\nFetching image from: {url}")
        response = requests.get(url, timeout=20)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        
        # Get content type from headers, default to jpeg if not found
        content_type = response.headers.get('Content-Type', 'image/jpeg')
        
        return {
            "data": response.content,
            "mime_type": content_type
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from {url}: {e}")
        return None

def get_image_description_from_gemini(image_info: dict, client) -> str:
    """
    Sends image data to the Gemini API using the requested streaming pattern
    and returns a text description. Handles quota errors by switching API keys.
    """
    if not image_info:
        return "[Image download failed]"

    # Construct the request using the correct pattern
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type=image_info["mime_type"],
                    data=image_info["data"]
                ),
                types.Part.from_text(text=GEMINI_PROMPT),
            ],
        ),
    ]
    
    # Configure tools
    tools = [
        types.Tool(googleSearch=types.GoogleSearch()),
    ]
    
    # Configure generation settings
    generate_content_config = types.GenerateContentConfig(
        tools=tools,
    )

    max_retries = len(API_KEYS)
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Generate content using the client's streaming method
            full_response_text = ""
            for chunk in client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    full_response_text += chunk.text
            
            # If the response is empty after streaming, it likely means it was blocked.
            if not full_response_text.strip():
                print(f"\nWarning: AI generation returned empty response. May be blocked by safety filters.")
                return "[AI content generation was blocked or returned empty]"

            return full_response_text.strip()
                
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a quota error
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                retry_count += 1
                
                # Extract retry delay if available
                retry_delay = 7  # default delay
                if "retryDelay" in error_str:
                    try:
                        import re
                        delay_match = re.search(r"'retryDelay': '(\d+)s'", error_str)
                        if delay_match:
                            retry_delay = int(delay_match.group(1))
                    except:
                        pass
                
                print(f"\nQuota exceeded on API key {current_key_index + 1}/{len(API_KEYS)}")
                
                if retry_count < max_retries:
                    # Switch to next API key
                    next_key = get_next_api_key()
                    client = create_client_with_current_key()
                    print(f"Switching to API key {current_key_index + 1}/{len(API_KEYS)}")
                    print(f"Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                else:
                    print(f"\nAll API keys exhausted. Cannot process this image.")
                    return "[All API keys quota exceeded]"
            else:
                # Non-quota error
                print(f"\nError calling Gemini API: {e}")
                return "[AI generation failed due to an API error]"
    
    return "[Failed to generate description after all retries]"

def process_question_text(text: str, client) -> tuple[str, object]:
    """
    Finds all bracketed URLs in a string and processes them.
    Returns both the processed text and the potentially updated client.
    """
    # Regex to find any URL starting with http/https inside square brackets
    pattern = re.compile(r'\[(https?://[^\]]+)\]')
    
    # Find all matches before starting to modify the string
    matches = list(pattern.finditer(text))
    
    # Iterate backwards through the matches to avoid messing up character indices
    # during replacement.
    for match in reversed(matches):
        url = match.group(1)          # The URL captured from the parentheses
        full_match = match.group(0)   # The full matched string, e.g., "[https://...]"
        
        # Check if the URL points to an image
        if url.lower().endswith(IMAGE_EXTENSIONS):
            image_info = fetch_image_data(url)
            description = get_image_description_from_gemini(image_info, client)
            
            # Check if we need to update the client (it might have been switched during API call)
            client = create_client_with_current_key()
            
            text = text[:match.start()] + f"\n{description}\n" + text[match.end():]
        else:
            # It's a regular link, so remove the full match
            print(f"\nFound and removed non-image link: {url}")
            # We also remove one preceding space if it exists, to clean up formatting.
            start_index = match.start()
            if start_index > 0 and text[start_index - 1].isspace():
                start_index -= 1
            text = text[:start_index] + text[match.end():]
            
    return text, client

def safe_append_to_json_file(item: dict, file_path: str, is_first: bool):
    """
    Safely append a single item to a JSON file using atomic operations.
    """
    temp_path = file_path + '.tmp'
    
    try:
        if is_first or not os.path.exists(file_path):
            # Create new file
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump([item], f, indent=4, ensure_ascii=False)
        else:
            # Read existing data
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # Append new item
            existing_data.append(item)
            
            # Write to temp file
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4, ensure_ascii=False)
        
        # Atomic rename
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rename(temp_path, file_path)
        
    except Exception as e:
        print(f"Error in safe_append_to_json_file: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    """
    Append a single item to a JSON file in a way that maintains valid JSON structure.
    """
    # If file doesn't exist or is_first is True, start fresh
    if is_first or not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('[\n')
            json.dump(item, f, indent=4, ensure_ascii=False)
            f.write('\n]')
    else:
        # Append to existing file using a safer approach
        try:
            # Read the entire file content
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Decode to string
            content_str = content.decode('utf-8')
            
            # Find the last ']' character
            last_bracket_pos = content_str.rfind(']')
            
            if last_bracket_pos == -1:
                # No closing bracket found, append at the end
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(',\n')
                    json.dump(item, f, indent=4, ensure_ascii=False)
                    f.write('\n]')
            else:
                # Split the content at the last bracket
                before_bracket = content_str[:last_bracket_pos]
                
                # Write the updated content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(before_bracket)
                    if before_bracket.rstrip().endswith('['):
                        # First item after opening bracket
                        f.write('\n')
                    else:
                        # Not the first item, add comma
                        f.write(',\n')
                    json.dump(item, f, indent=4, ensure_ascii=False)
                    f.write('\n]')
                    
        except Exception as e:
            print(f"Error appending to file: {e}")
            print("Attempting simple append...")
            # Fallback: simple append
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(',\n')
                json.dump(item, f, indent=4, ensure_ascii=False)
                f.write('\n]')

# --- Main Execution ---

def main():
    """
    Main function to run the entire process.
    """
    if not API_KEYS or not API_KEYS[0]:
        print("FATAL ERROR: No API keys configured.")
        print("Please add API keys to the API_KEYS list.")
        return

    # Create the initial Gemini client
    client = create_client_with_current_key()

    # --- Read Input File ---
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: Input file not found at '{INPUT_JSON_PATH}'")
        return
    except json.JSONDecodeError:
        print(f"FATAL ERROR: The file at '{INPUT_JSON_PATH}' is not valid JSON.")
        return

    # Check if output file exists and has content to determine resume position
    start_index = 0
    if os.path.exists(OUTPUT_JSON_PATH):
        try:
            # Try to read the file more carefully
            with open(OUTPUT_JSON_PATH, 'rb') as f:
                content = f.read()
            
            # Check if file is empty
            if not content.strip():
                print("Output file is empty. Starting fresh.")
                start_index = 0
            else:
                # Decode and parse JSON
                content_str = content.decode('utf-8', errors='ignore')
                existing_data = json.loads(content_str)
                start_index = len(existing_data)
                print(f"Resuming from index {start_index} (found {start_index} already processed items)")
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse existing output file: {e}")
            print("Starting fresh with a new file.")
            start_index = 0
            # Backup the corrupted file
            if os.path.exists(OUTPUT_JSON_PATH):
                backup_path = OUTPUT_JSON_PATH + '.backup'
                os.rename(OUTPUT_JSON_PATH, backup_path)
                print(f"Corrupted file backed up to: {backup_path}")
        except Exception as e:
            print(f"Warning: Error reading existing output file: {e}")
            print("Starting fresh.")
            start_index = 0

    # --- Process Data ---
    total_items = len(data)
    items_to_process = total_items - start_index
    
    if items_to_process <= 0:
        print("All items have already been processed!")
        return
    
    print(f"Starting to process {items_to_process} items (indices {start_index} to {total_items-1})...")
    
    # Use tqdm for a nice progress bar
    for i, item in enumerate(tqdm(data[start_index:], desc="Processing Questions", initial=start_index, total=total_items)):
        actual_index = start_index + i
        new_item = item.copy()
        original_question = new_item.get("Question", "")
        
        # Only process questions that contain a potential link
        if "http" in original_question and "[" in original_question:
            processed_question, client = process_question_text(original_question, client)
            new_item["Question"] = processed_question
        
        # Append to file immediately after processing
        # Use the safer method for better reliability
        try:
            safe_append_to_json_file(new_item, OUTPUT_JSON_PATH, is_first=(actual_index == 0))
        except Exception as e:
            print(f"\nError with safe append, trying regular append: {e}")
            append_to_json_file(new_item, OUTPUT_JSON_PATH, is_first=(actual_index == 0))
        
        # Small delay to avoid hitting rate limits too quickly
        time.sleep(0.1)

    print(f"\n✅ Processing complete!")
    print(f"Updated data has been saved to '{OUTPUT_JSON_PATH}'")
    print(f"Total items processed: {items_to_process}")
    print(f"Current API key index: {current_key_index + 1}/{len(API_KEYS)}")

if __name__ == "__main__":
    main()