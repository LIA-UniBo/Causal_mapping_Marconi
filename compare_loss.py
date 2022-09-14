import json
import statistics
import matplotlib.pyplot as plt
import numpy as np

def draw_result(epochs, lst_loss_shift, lst_loss, title):
    plt.plot(epochs, lst_loss_shift, '-b', label='loss_shift')
    plt.plot(epochs, lst_loss, '-r', label='loss_no_shift')

    plt.xlabel("epochs")
    plt.legend(loc='upper left')
    plt.title(title+': '+str(causes[title])+' direct causes')

    # save image
    plt.savefig('./temporal_evaluation/plots/no_drop_nodistance_'+title+".png") 
    # show
    plt.show()

l = {}
l_s = {}
causes = {}
with open('./temporal_evaluation/loss_shift_epoch30_nodistance.json', 'r') as fp:
    l_s = json.load(fp)

with open('./temporal_evaluation/loss_without_shift_epoch30_nodistance.json', 'r') as fp:
    l = json.load(fp)

with open('./temporal_evaluation/feature_causes.json', 'r') as fp:
    causes = json.load(fp)

for f in l:
    print(f)
    print(l_s[f][-1], l[f][-1], bool(l_s[f][-1] < l[f][-1]))
    #print('Start point')
    #print(l_s[f][0], l[f][0], bool(l_s[f][0] < l[f][0]))
    #print('Mean')
    #print(statistics.mean(l_s[f]), statistics.mean(l[f]), bool(statistics.mean(l_s[f]) < statistics.mean(l[f])))
    #print('######################')
    draw_result(range(len(l_s[f])), l_s[f], l[f], f)
    

