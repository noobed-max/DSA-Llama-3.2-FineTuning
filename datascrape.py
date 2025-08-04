import requests
from bs4 import BeautifulSoup
import json
import time
import csv
import html
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def load_urls_from_csv(file_path):
    """
    Loads URLs from a single-column CSV file.
    """
    urls = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            url_reader = csv.reader(csvfile)
            for row in url_reader:
                if row:
                    urls.append(row[0].strip())
        print(f"✅ Successfully loaded {len(urls)} URLs from {file_path}")
    except FileNotFoundError:
        print(f"❌ ERROR: The file at '{file_path}' was not found.")
    except Exception as e:
        print(f"❌ ERROR: An error occurred while reading the CSV file: {e}")
    return urls

def scrape_leetcode_problems_with_selenium(urls):
    """
    Scrapes LeetCode problem descriptions from a list of URLs using Selenium
    and saves them incrementally to a JSON file.
    """
    if not urls:
        print("No URLs to process. Exiting.")
        return

    scraped_data = []
    output_filename = 'leetcode_problems_selenium.json'
    
    # --- Selenium Setup ---
    # Set up Chrome options for headless Browse (optional)
    chrome_options = Options()
    # To see the browser window, comment out the next line
    # chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    # Initialize the Chrome driver
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        print("✅ Selenium WebDriver initialized successfully.")
    except Exception as e:
        print(f"❌ ERROR: Could not initialize Selenium WebDriver: {e}")
        print("Please ensure you have Google Chrome installed.")
        return
    # --- End Selenium Setup ---

    for i, url in enumerate(urls):
        original_url = url.strip().rstrip('/')
        
        # --- Extract problem name from the URL ---
        problem_title = original_url.replace("https://leetcode.com/problems/", "")
        
        # Append /description/ to get to the right page
        final_url = original_url + '/description/'
        
        print(f"Processing URL {i+1}/{len(urls)}: {final_url}")
        print(f"  -> Extracted Title: {problem_title}")
        try:
            # Use Selenium to get the page
            driver.get(final_url)
            time.sleep(2) # Wait for the page to load

            # Get the page source
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')

            # Find the meta description tag
            meta_tag = soup.find('meta', {'name': 'description'})

            if meta_tag and meta_tag.has_attr('content'):
                question_content = html.unescape(meta_tag['content'].strip())
                print(f"  -> ✅ Successfully extracted description.")
            else:
                print(f"  -> ❌ Could not find meta description for {final_url}. Skipping.")
                continue

            # --- Format, Store, and Write Data Incrementally ---
            problem_entry = {
                "Question": question_content,
                "Answer": "",
                "title": problem_title
            }
            scraped_data.append(problem_entry)
            
            # Write the entire list to the file after each addition
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(scraped_data, f, indent=4, ensure_ascii=False)
            print(f"  -> ✅ Data saved to {output_filename}. Total items: {len(scraped_data)}")
            
        except Exception as e:
            print(f"  -> An unexpected error occurred for {final_url}: {e}")

    # --- Close the browser ---
    driver.quit()

    # --- Final Summary ---
    if scraped_data:
        print(f"\n✅ Scraping complete. A total of {len(scraped_data)} problems were saved to '{output_filename}'.")
    else:
        print("\nNo data was scraped.")

if __name__ == '__main__':
    # Make sure to update this path to the actual location of your CSV file
    csv_file_path = '/home/deadsec/Downloads/leetcode_links.csv'  
    leetcode_urls = load_urls_from_csv(csv_file_path)
    scrape_leetcode_problems_with_selenium(leetcode_urls)
