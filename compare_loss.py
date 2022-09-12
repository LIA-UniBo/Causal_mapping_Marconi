import json
import statistics

l = {}
l_s = {}
with open('./temporal_evaluation/loss_shift_epoch30.json', 'r') as fp:
    l_s = json.load(fp)

with open('./temporal_evaluation/loss_without_shift_epoch30.json', 'r') as fp:
    l = json.load(fp)

for f in l:
    print(f)
    print(l_s[f][-1], l[f][-1], bool(l_s[f][-1] < l[f][-1]))
    #print('Start point')
    #print(l_s[f][0], l[f][0], bool(l_s[f][0] < l[f][0]))
    #print('Mean')
    #print(statistics.mean(l_s[f]), statistics.mean(l[f]), bool(statistics.mean(l_s[f]) < statistics.mean(l[f])))
    #print('######################')

