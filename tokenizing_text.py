import urllib.request

#Step 1: Import the text
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

#load the text using Pythons standard file reading utiliites
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of characters: ", len(raw_text))
print(raw_text[:99])

#Step 2: Split the text for better tokenization
#Keep the puncuation marks with the words, and refrain from removing the capitialization beacsue LLMs need it. 
import re
text = raw_text[:99]
result = re.split(r'(\s)', text)
print(result)

#Modify the split to remove whitespaces
result = re.split(r'([,.] |\s)', text)
print(f"Result with words and punctuation split: {result}")

result = [item for item in result if item.strip()]
print(f"Result with whitespaces removed: {result}")

#Removing whitespaces reduces memory requirements but whitespaces are important fo tthe models to understand the exact structure of the text.


#Improving it to handle multiple punctuation marks
result2 = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result2 = [item.strip() for item in result2 if item.strip()]
print(f"Improved results to handle multplie punctuation marks: {result2}")

#Now apply the final split version to the entire raw_text
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(f"Total number of tokens: {len(preprocessed)}")
print(f"First 30 tokens: {preprocessed[:30]}")