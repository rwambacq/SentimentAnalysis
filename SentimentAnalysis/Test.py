import sys

orig_stdout = sys.stdout
f = open('testbestand.txt', 'w')
sys.stdout = f

for i in range(2):
    print('i = ', i)

sys.stdout = orig_stdout
f.close()