import os
import time
import requests
import PyPDF2
from io import BytesIO
import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
import unicodedata
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def stream_pdf_from_url(url):
    """Stream a PDF file from a given URL."""
    logging.info("Streaming the PDF file from URL.")
    response = requests.get(url, stream=True)
    response.raise_for_status() 
    return BytesIO(response.content)

def extract_text_from_pdf(pdf_stream):
    """Extract text from a PDF stream."""
    logging.info("Extracting text from the PDF.")
    pdf_reader = PyPDF2.PdfReader(pdf_stream)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text() or ""
        text += page_text
    return text

def normalize_text(text):
    """Normalize text to handle special characters."""
    return unicodedata.normalize('NFKC', text)

def chunk_text(text, chunk_size=5000):
    """Chunk text into smaller pieces."""
    text = normalize_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Ensure we do not split surrogate pairs
        if end < len(text):
            while end < len(text) and text[end] in ('\uD800', '\uDBFF'):
                end += 1
        chunks.append(text[start:end])
        start = end
    return chunks

def summarize_chunk(llm, chunk, retries=3, delay=1):
    """Summarize a text chunk using a language model."""
    prompt = f"Please provide a well-defined summary of the following text:\n TEXT: {chunk}"
    for attempt in range(retries):
        try:
            # Encode and decode to handle special characters
            encoded_chunk = chunk.encode('utf-8', 'replace').decode('utf-8')
            response = llm.generate_content(prompt)
            return response.text
        except ResourceExhausted as e:
            logging.warning(f"Error processing chunk: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
        except UnicodeEncodeError as e:
            logging.error(f"Encoding error: {e}. Skipping chunk.")
            return None
        except Exception as e:
            logging.error(f"Unexpected error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
    logging.error("Max retries exceeded for summarizing the chunk.")
    raise Exception("Max retries exceeded")


def main(file_url, chunk_size=5000):
    """Main function to process PDF and summarize text."""
    try:
        pdf_stream = stream_pdf_from_url(file_url)
        text = extract_text_from_pdf(pdf_stream)
        chunks = chunk_text(text, chunk_size)
        
        llm = genai.GenerativeModel("gemini-pro")
        summaries = []

        logging.info("Starting the summarization process.")
        for i, chunk in enumerate(chunks):
            try:
                summary = summarize_chunk(llm, chunk)
                if summary:  # Only append if summarization was successful
                    summaries.append(summary)
                    logging.info(f"Chunk {i+1}/{len(chunks)} summarized successfully.")
                else:
                    logging.warning(f"Chunk {i+1} could not be summarized. Skipping.")
                time.sleep(1)  # Add delay between requests to avoid rate limit
            except Exception as e:
                logging.error(f"Error processing chunk {i+1}: {e}")
                continue

        # Ensure no None values are in summaries
        final_summary = " ".join(filter(None, summaries))
        logging.info("\nFinal Summary:\n")
        logging.info(final_summary)

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    file_url = 'https://www.bu.edu/lernet/artemis/years/2011/slides/python.pdf'
    main(file_url)



# without api key----------------------------------------------------------------------------------------------------------------


# import requests
# from PyPDF2 import PdfReader
# from transformers import pipeline
# from io import BytesIO

# # URL of the PDF
# pdf_url = "https://www.bu.edu/lernet/artemis/years/2011/slides/python.pdf"

# # Fetch the PDF content
# response = requests.get(pdf_url)
# response.raise_for_status()  # Ensure the request was successful

# # Read the PDF content with PyPDF2
# pdf_file = BytesIO(response.content)
# pdf_reader = PdfReader(pdf_file)

# # Extract text from each page
# text_content = ""
# for page in pdf_reader.pages:
#     text_content += page.extract_text()

# # Load the summarization pipeline
# summarizer = pipeline("summarization")

# # Define a function to chunk the text
# def chunk_text(text, chunk_size=512):
#     words = text.split()
#     return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# # Summarize the text content
# chunks = chunk_text(text_content)
# summary = []

# for chunk in chunks:
#     # Summarize each chunk individually
#     summary_chunk = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
#     summary.append(summary_chunk[0]['summary_text'])

# # Join the summary chunks
# summary_text = " ".join(summary)

# # Print the summary
# print("Summary: ", summary_text)





