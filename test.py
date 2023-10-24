import torch 



a = [[None, None], [None, None]]

a[0][0] = '00'
a[0][1] = '01'
a[1][0] = '10'
a[1][1] = '11'

for r in range(2):
    for d, px_z in enumerate(a[r]):
        print(px_z)
