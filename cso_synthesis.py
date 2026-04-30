def remove_elements_by_indices(input_list, indices_to_remove):
    indices_to_remove.sort(reverse=True)

    for index in indices_to_remove:
        if 0 <= index < len(input_list):
            del input_list[index]

    return input_list

path_to_netlist = input("Enter the path for input netlist: ")
path_to_netlist_out = input("Enter the path for output netlist: ")

# read Block-Cipher netlist
with open(path_to_netlist) as f:
    content_list = f.readlines()

# remove new line characters
content_list = [x.strip() for x in content_list]

result = []

for x in content_list:
    temp = x.split()
    result.append(temp)

print("\nNetlist size before: ", len(result))

# finding AND output used as input to AND (replace)

count_dep = 0
remove_list = []
for id in range(len(result)):
  if result[id][-1] == 'AND':
    w1 = result[id][2]
    w2 = result[id][3]
    wo = result[id][4]

    count = 0

    for j in range(id, len(result)):
      if ((result[j][2] == wo) or (result[j][3] == wo)) and (result[j][-1] == 'AND'):
        if (result[j][2] == wo):
          tmp_idx = result[j][3]
        if (result[j][3] == wo):
          tmp_idx = result[j][2]

        for k in range(len(result)):
          if wo in result[k]:
            count = count + 1
        if count == 2:
          count_dep = count_dep + 1
          tmp_list = ['3', '1', result[id][2], result[id][3], tmp_idx, result[j][4], 'AND3']
          result[j] = tmp_list
          result[id] = ["replaced"]
          remove_list.append(id)

result = remove_elements_by_indices(result, remove_list)
print("Netlist size after AND3 gates: ", len(result))

# finding XOR output used as input to XOR (replace)

count_dep = 0
remove_list = []
for id in range(len(result)):
  if result[id][-1] == 'XOR':
    w1 = result[id][2]
    w2 = result[id][3]
    wo = result[id][4]

    count = 0

    for j in range(id, len(result)):
      if ((result[j][2] == wo) or (result[j][3] == wo)) and (result[j][-1] == 'XOR'):
        if (result[j][2] == wo):
          tmp_idx = result[j][3]
        if (result[j][3] == wo):
          tmp_idx = result[j][2]

        for k in range(len(result)):
          if wo in result[k]:
            count = count + 1
        if count == 2:
          count_dep = count_dep + 1
          tmp_list = ['3', '1', result[id][2], result[id][3], tmp_idx, result[j][4], 'XOR3']
          result[j] = tmp_list
          result[id] = ["replaced"]
          remove_list.append(id)

result = remove_elements_by_indices(result, remove_list)
print("Netlist size after XOR3 gates: ", len(result))

# finding AND output used as input to XOR (replace)

count_dep = 0
remove_list = []
for id in range(len(result)):
  if result[id][-1] == 'AND':
    w1 = result[id][2]
    w2 = result[id][3]
    wo = result[id][4]

    count = 0

    for j in range(id, len(result)):
      if ((result[j][2] == wo) or (result[j][3] == wo)) and (result[j][-1] == 'XOR'):
        if (result[j][2] == wo):
          tmp_idx = result[j][3]
        if (result[j][3] == wo):
          tmp_idx = result[j][2]

        for k in range(len(result)):
          if wo in result[k]:
            count = count + 1
        if count == 2:
          count_dep = count_dep + 1
          # print("start index: ", id, result[id], " -> dependent index: ", j, result[j])
          tmp_list = ['3', '1', result[id][2], result[id][3], tmp_idx, result[j][4], 'AND-XOR']
          result[j] = tmp_list
          result[id] = ["replaced"]
          remove_list.append(id)

result = remove_elements_by_indices(result, remove_list)
print("Netlist size after AND-XOR gates: ", len(result))

print("\n\nFinal netlist size: ", len(result))

# Open the file in write mode
with open(path_to_netlist_out, 'w') as file:
    for index, sublist in enumerate(result):
        file.write(' '.join(map(str, sublist)))
        if index < len(result) - 1:
            file.write('\n')
