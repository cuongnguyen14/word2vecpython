'''
Created on May 13, 2017

@author: CuongNguyen
'''
import re
from os import listdir
from os.path import isfile, join
import glob

start = 0
step = 1000
dirin = 'joininput'
dirout = 'joininput'

lines = ''

onlyfiles = (glob.glob(dirin + "/*.txt"))
text_file = open(dirout + "/join_" + str(start) + ".txt", "w")

for filename in onlyfiles :
    print "processing " + filename
    with open(filename) as f:
        start = 0
        for line in f:
            if line.startswith('< <'):
                continue
            if line.startswith('/ a') or line.startswith('/ t') or line.startswith('/ c'):
                continue
            start += 1
            text_file.write("\n" + line)

            if (start % 100000 == 0) :
                print filename + " processing " + str(start)
text_file.close()

print "END"