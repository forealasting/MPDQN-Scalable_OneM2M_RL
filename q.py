# Convert the list to a string
data_str = '\n'.join(str(x) for x in data)

# Write the string to a text file
path = "reward.txt"
f = open(path, 'w')
f.write(data_str)