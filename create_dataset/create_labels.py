import os

path = './example'
label = 'example_labels.txt'

file_name = os.listdir(path)

label_txt = open(label, 'w')
for i in range(len(file_name)):
    cur_label = file_name[i].split('.jpg')[0]
    label_txt.writelines(cur_label + '   ' + str(i) + '\n')
    os.rename(path + '/' + file_name[i], path + '/' + str(i) + '.jpg')

print('over')

        
        
        
