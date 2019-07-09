def concatenate_texts(path, pattern):
    file_paths = list(path.glob("{}*.txt".format(pattern)))

    input_texts = []
    target_texts = []

    for file_path in file_paths:
        with open(str(file_path), "r", encoding="utf-8") as f:
            lines = f.read().split("\n")[:-1]

        input_texts.extend(lines[0::2])
        target_texts.extend(["\t" + target_text + "\n" for target_text in lines[1::2]])

    return input_texts, target_texts