'''
Created on Apr 26, 2017

@author: CuongNguyen
'''
import sys
import copy

reload(sys)
sys.setdefaultencoding('utf8')

def listprinter(l):
    if isinstance(l, list):
        return u'\n'.join([listprinter(i) for i in l])
    elif isinstance(l, (str, unicode)):
        return unicode(l)

lines = [line.rstrip('\n') for line in open('rawinput')]
mdict = {}
mdata = []
for line in lines :
    if line.find(':') == 0 :
        mdata = []
        mdict[line] = mdata
    else:
        mdata.append(line)

newDict = {}
for key in mdict.keys() :
    data = mdict[key]
    newdata = []
    for token in data:
        ignoreData = copy.copy(data)
        ignoreData.remove(token)
        for subToken in ignoreData:
            newString = token + ' ' + subToken
            newdata.append(newString)
    newDict[key] = newdata

# print mdict.keys()
# [':GRAM3-COMPARATIVE', ':GRAM7-PAST-TENSE', ':GRAM2-OPPOSITE', ':FAMILY', ':GRAM4-SUPERLATIVE', ':GRAM5-PRESENT-PARTICIPLE']

for key in newDict.keys():
    print key
    print listprinter(newDict[key])
    
# print ': FAMILY'
# print listprinter(newDict[': FAMILY'])
# print ': GRAM2-OPPOSITE'
# print listprinter(newDict[': GRAM2-OPPOSITE'])
# print ': GRAM3-COMPARATIVE'
# print listprinter(newDict[': GRAM3-COMPARATIVE'])
# print ': GRAM4-SUPERLATIVE-SUFIX'
# print listprinter(newDict[': GRAM4-SUPERLATIVE-SUFIX'])
# print ': GRAM5-PRESENT-PARTICIPLE'
# print listprinter(newDict[': GRAM5-PRESENT-PARTICIPLE'])
# print ': FAMILY'
# print listprinter(newDict[': FAMILY'])
# print ': GRAM2-OPPOSITE'
# print listprinter(newDict[': GRAM2-OPPOSITE'])
# print ': GRAM3-COMPARATIVE'
# print listprinter(newDict[': GRAM3-COMPARATIVE'])
# print ': GRAM4-SUPERLATIVE'
# print listprinter(newDict[': GRAM4-SUPERLATIVE'])
# print ': GRAM5-PRESENT-PARTICIPLE'
# print listprinter(newDict[': GRAM5-PRESENT-PARTICIPLE'])
