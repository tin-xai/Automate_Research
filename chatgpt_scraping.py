from chatgpt_wrapper import ChatGPT
import json
import os
import pandas as pd

def get_definition():
    bot = ChatGPT()

    class_name_path = "./365Places/filelist_places365-standard/categories_places365.txt"
    f = open(class_name_path, 'r')
    lines = f.readlines()

    def remove_underscore(text):
        new_text = text.replace("_", " ")
        return new_text
        
    label2idx = {}
    idx2def = {} # {idx: {class_name: class_name, definitions: [def1,...]}}

    prompts_4_single = ["what is {}", "what appears in the {}"]
    prompts_4_double = ["what is the {} of the {}", "what appears in the {} of the {}"]

    num_test = 1
    resume_idx = 353

    for line in lines:
        if '\n' in line:
            line = line[:-1]
        components = line.split(' ')
        label_idx = int(components[1])
        if label_idx < resume_idx or label_idx > resume_idx + 30:
            continue
        label2idx[components[0]] = label_idx

        names = components[0].split('/')[2:]
        
        list_definitions = []
        if len(names) == 1:
            name = remove_underscore(names[0])
            for p in prompts_4_single:
                p = p.format(name)
                try:
                    response = bot.ask(p)
                except:
                    response = 'None'
                list_definitions.append(response)
                
        elif len(names) == 2:
            name1 = remove_underscore(names[0])
            name2 = remove_underscore(names[1])
            for p in prompts_4_double:
                p = p.format(name1, name2)
                try:
                    response = bot.ask(p)
                except:
                    response = 'None'
                list_definitions.append(response)

        idx2def[label_idx] = {}
        idx2def[label_idx]['class_name'] = components[0]
        idx2def[label_idx]['definitions'] = list_definitions

    # Serializing json
    json_object = json.dumps(idx2def, indent=4)
    with open(f"365_definitions_{resume_idx}.json", "w") as outfile:
        outfile.write(json_object)

def concat_def():
    files = []
    for file in os.listdir("./"):
        if file.startswith("365_definitions_"):
            files.append(file)
    def func(elem):
        index = int(elem[:-5].split("_")[-1])
        return index

    files.sort(key=func)

    full_data = {}
    for file in files:
        f = open(file)
        data = json.load(f)
        for k in data:
            full_data[k] = data[k]
    
    print(len(full_data))
    # save json
    json_object = json.dumps(full_data, indent=4)
 
    # Writing to sample.json
    with open("full_365_definitions.json", "w") as outfile:
        outfile.write(json_object)

def json2csv():
    f = open("full_365_definitions.json")
    json_def = json.load(f)

    data = {}
    indexes = []
    class_names = []
    defs = []
    for ind in json_def:
        indexes.append(ind)
        class_names.append(json_def[ind]['class_name'])
        defs.append(json_def[ind]['definitions'])
    data['Index'] = indexes
    data['Class Name'] = class_names
    data['Definitions'] = defs

    df = pd.DataFrame(data, columns=['Index', "Class Name","Definitions"])
    df.to_csv("full_365_definitions.csv", index=False)
json2csv()
# concat_def()
