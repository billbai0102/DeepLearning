import time
from tqdm import tqdm


dl = list(range(200))
dl_tqdm = tqdm(dl)

for epoch in range(200):
    print(epoch)
    for i in enumerate(dl_tqdm):
        time.sleep(0.0001)
        tqdm.write('Hi', end = '')