import json
with open('./results/6fd63984-28a0-11eb-9fd0-0242ac1c0002.json', 'r') as fp:
    data = json.load(fp)

print(max(data['accuracies']))
print(data['strategy'])
print(data['did_it_train'])
