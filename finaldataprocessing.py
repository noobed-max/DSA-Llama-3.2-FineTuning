import json
import os

def create_new_json_with_answers():
    """
    Reads a JSON file of LeetCode problems, populates the 'Answer' field
    with content from corresponding C++ files, and saves it to a new JSON file.
    It skips any entries for which a C++ file is not found.
    """
    # --- Configuration ---
    input_json_path = '/home/deadsec/Desktop/ML/FIne tuning/leetcode_problems_processed.json'
    cpp_source_dir = '/home/deadsec/Desktop/ML/FIne tuning/C++/'
    output_json_path = '/home/deadsec/Desktop/ML/FIne tuning/leetcode_with_answers_filtered.json'

    # --- Main Logic ---
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            problems_data = json.load(f)
        print(f"✅ Successfully loaded {len(problems_data)} problems from {input_json_path}")
    except FileNotFoundError:
        print(f"❌ Error: Input JSON file not found at '{input_json_path}'")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Could not parse JSON from '{input_json_path}'. Please check if it's a valid JSON.")
        return

    processed_problems = []
    skipped_count = 0

    # Step 2: Iterate through each problem
    for problem in problems_data:
        title = problem.get('title')
        if not title:
            print("⚠️ Warning: Skipping an entry because it has no 'title' key.")
            skipped_count += 1
            continue

        cpp_file_path = os.path.join(cpp_source_dir, f"{title}.cpp")

        # Step 3: Try to read the C++ file. If it doesn't exist, skip this entry.
        try:
            with open(cpp_file_path, 'r', encoding='utf-8') as cpp_file:
                raw_content = cpp_file.read()
                answer_content = raw_content.replace('\n', '\\n')

            # **Entry creation is now INSIDE the 'try' block.**
            # This ensures it only happens if the file is successfully read.
            new_problem_entry = {
                "Question": problem["Question"],
                "Answer": answer_content
            }
            processed_problems.append(new_problem_entry)

        except FileNotFoundError:
            # **MODIFIED BEHAVIOR**
            # If the file is not found, print a message and skip to the next problem.
            print(f"⏭️ Skipping entry for title '{title}': C++ file not found at '{cpp_file_path}'.")
            skipped_count += 1
            continue # This jumps to the next iteration of the loop.

    # Step 4: Write the processed data to a new JSON file
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_problems, f, indent=4, ensure_ascii=False)
        print("\n---")
        print(f"✅ Success! Processing complete.")
        print(f"📝 {len(processed_problems)} entries written to the new file.")
        if skipped_count > 0:
            print(f"🚫 {skipped_count} entries were skipped due to missing files or titles.")
        print(f"📄 New JSON file created at: {output_json_path}")
    except IOError as e:
        print(f"❌ Error: Could not write the output file to '{output_json_path}'. Reason: {e}")

# --- Run the script ---
if __name__ == "__main__":
    create_new_json_with_answers()