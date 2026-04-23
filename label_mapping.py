import os

LABEL_ROOT = "labels"

unique_labels = set()

for file in os.listdir(LABEL_ROOT):
    with open(os.path.join(LABEL_ROOT, file)) as f:
        for line in f:
            unique_labels.add(line.strip())

unique_labels = sorted(list(unique_labels))

label2id = {label: i for i, label in enumerate(unique_labels)}

print("Label Mapping:")
for k, v in label2id.items():
    print(k, "→", v)