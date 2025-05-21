# %% 
import requests
from bs4 import BeautifulSoup
import shutil
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import time
import json
import multiprocessing as mp

# webpages
HEADERS = {'accept': '"text/html', 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'}

# %%
def transcript_scraper(html_file_path):
        
    with open(html_file_path, 'rb') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    # -----Get "Shape Media" image for this species-----
    sub_tag = soup.find('div', {"id":"subtitles-container", "class":"show-sub"})
    span_tags = sub_tag.find_all('span')

    eng_subs = []
    vi_subs = []

    for i, span in enumerate(span_tags):
        # if i % 2 != 0:
        #     continue
        eng_sub_tag =  span.find_all('span', {"class":"js-textEn"})
        vi_sub_tag = span.find_all('small', {"class":"js-textVi"}) 
        
        if eng_sub_tag == [] or vi_sub_tag == []:
            continue
        eng_text = eng_sub_tag[0].text
        vi_text = vi_sub_tag[0].text
        # print(eng_text)
        # print(vi_text)
        # print('----')

        eng_subs.append(eng_text)
        vi_subs.append(vi_text)

    # save to file
    with open(f"{html_file_path[:-5]}.txt", "w") as f:
        for eng, vi in zip(eng_subs, vi_subs):
            f.write(f"{eng}\n{vi}\n")
    
    

# %%
transcript_scraper('/Users/tinnguyen/Downloads/ss1_5.html')

# %%
