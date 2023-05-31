chunk_size = 1000000 # read 1 MB at a time
with open('./data/lscp-0.5-fa-derivation-tree.txt', 'r',encoding='utf-8') as file:
    while True:
        data = file.read(chunk_size)
        if not data:
            break
        # process the data chunk here
        print(data)
