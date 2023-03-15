import json


white_list = [132, 135, 139, 150, 156, 157, 220, 221, 223, 226, 228, 242, 244, 245, 248, 250, 251, 255]
grey_list = []
black_list = []

white_dict = {}
grey_dict = {}
black_dict = {}

lags = []
# Opening JSON file
with open('shift2.json') as json_file:
    data = json.load(json_file)
    for id in white_list:
        lags.append(data[str(id)])
        white_dict[id] = data[str(id)]


with open('shift3.json') as json_file:
    data = json.load(json_file)
    for id in data:
        if int(id) not in white_list:
            if -20 < data[str(id)] < 0:
                grey_list.append(int(id))
            else:
                black_list.append(int(id))

    for id in grey_list:
        grey_dict[id] = data[str(id)]
    for id in black_list:
        black_dict[id] = data[str(id)]

json.dump(white_dict, open("white_list.json", 'w'))
print(f"{len(white_list)} ids whitelisted")
json.dump(grey_dict, open("grey_list.json", 'w'))
print(f"{len(grey_list)} ids greylisted")
json.dump(black_dict, open("black_list.json", 'w'))
print(f"{len(black_list)} ids blacklisted")



