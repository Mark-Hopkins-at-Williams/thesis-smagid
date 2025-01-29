import re
import os
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

ALL_HTML = "html-files/all-fonts.html"
HTML_DIR = 'html-files/attributed'

def count(html_dir, searchTerm):
    for filePath in os.listdir(html_dir):
        count = 0
        with open(f'{HTML_DIR}/{filePath}', 'r') as file:
            for line in file:
                start = 0
                while True:
                    start = line.find(searchTerm, start)
                    if start == -1:
                        break
                    indexFound = start
                    count += 1
                    start += len(searchTerm)
        print(f'{filePath}: {count}')

def buildDataset(all_html, html_dir):
    # build a dataframe with all the google fonts
    allFonts = []
    with open(all_html, 'r') as file:
        for line in file:
            soup = BeautifulSoup(line, "html.parser")
            for a_tag in soup.find_all("a"):
                h1_tag = a_tag.find("h1")
                if h1_tag and h1_tag.find("span"):
                    fontFound = h1_tag.get_text(strip=True)
                    allFonts.append(fontFound)
    df = pd.DataFrame({'font': allFonts})

    # populate attributes
    for filePath in tqdm(os.listdir(html_dir)):
        attributeName = filePath.split('.')[0]
        df[attributeName] = 0
        fontsFound = []
        with open(f'{HTML_DIR}/{filePath}', 'r') as file:
            for line in file:
                soup = BeautifulSoup(line, "html.parser")
                for a_tag in soup.find_all("a"):
                    h1_tag = a_tag.find("h1")
                    if h1_tag and h1_tag.find("span"):
                        fontFound = h1_tag.get_text(strip=True)
                        fontsFound.append(fontFound)
        for fontFound in fontsFound:
            df.loc[df['font'] == fontFound, attributeName] = 1

    df.to_csv("fonts-attributes.csv", index=False)

if __name__ == "__main__":
    pattern = r"(.*?)\s*</span></h1></a>"
    # count(HTML_DIR, '</span></h1></a>')
    buildDataset(ALL_HTML, HTML_DIR)