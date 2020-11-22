import logging
logging.basicConfig(level=logging.INFO)

path = 'train_text'
with open(path,'r') as fid:
    for line in fid:
        char_text = []
        parts = line.strip().split(' ')
        utt = parts[0]
        text = parts[1:]
        text = "".join(text)
        char = " ".join(text)
        char = char.split(' ')
        char_text.append(utt)
        for item in char:
            char_text.append(item)

        with open('char_' + path, 'a', encoding='utf8') as fid:
            fid.write(" ".join(char_text) + '\n')






