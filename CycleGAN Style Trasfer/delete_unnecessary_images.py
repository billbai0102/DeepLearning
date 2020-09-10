import os

path = './progress_images/'
count = 0
for file in os.listdir(path):
    if count == 1 or count ==2:
        os.remove(os.path.join(path, file))
    if count == 2:
        count = 0
    else:
        count += 1