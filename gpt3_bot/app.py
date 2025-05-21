from flask import Flask, render_template, request
import os
import json
import openai
from time import time,sleep


app = Flask(__name__)


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Create an openaiapikey.txt file and save your api key.
openai.api_key = open_file('openaiapikey.txt')


def bot(prompt, engine='text-davinci-003', temp=0.7, top_p=1.0, tokens=100, freq_pen=0.0, pres_pen=0.5, stop=['<<END>>']):
    max_retry = 1
    retry = 0
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=[" User:", " AI:"])
            text = response['choices'][0]['text'].strip()
            print(text)
            # filename = '%s_gpt3.txt' % time()
            # with open('gpt3_logs/%s' % filename, 'w') as outfile:
            #    outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    # userText = request.args.get('msg')
    with open('gpt_imagenet_prompts.json') as f:
        labels = json.load(f)
    
    new_labels = {}
    # command = f"What are useful features for distinguishing a {label} sign in Germany in a photo?"
    # command = f"Check the grammar of this sentence {template} "

    # This sentence is grammatically correct.
    # The sentence is grammatically correct.
    # The grammar of this sentence is correct.
    # The sentence is grammatically correct.
    # The grammar of this sentence is correct

    for i, label in enumerate(labels):
        new_labels[label] = []
        for j, template in enumerate(labels[label]):
            command = f"Check the grammar of this sentence {template} "
            botresponse = bot(prompt = command)
            new_labels[label].append(botresponse)

    with open('gpt_imagenet_prompts_2.json', 'w') as f:
       json.dump(new_labels, f, indent=4)
    
    # return str(botresponse)
    return 'haha'

if __name__ == "__main__":
    app.run(debug = True)
