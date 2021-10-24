import csv

with open('data/jester/annotations/jester-v1-labels.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter='\n', quotechar='|')
    jester_cls2label = list(reader)
for idx in range(len(jester_cls2label)):
    # print(jester_cls2label[idx])
    assert len(jester_cls2label[idx]) == 1
    jester_cls2label[idx] = jester_cls2label[idx][0]
# print(jester_cls2label)

ucf_cls2label = []
with open('data/ucf101/annotations/classInd.txt', 'r') as txtfile:
    for line in txtfile.readlines():
        ucf_cls2label.append(line.strip().split(' ')[-1])
# print(ucf_cls2label)
