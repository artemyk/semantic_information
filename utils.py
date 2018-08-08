from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def get_marginals(probs, s_to_xy_map):
    num_Xstates = max(v[0] for v in s_to_xy_map.values())+1
    num_Ystates = max(v[1] for v in s_to_xy_map.values())+1
    pX = np.zeros(num_Xstates)
    for ndx, p in enumerate(probs):
        pX[s_to_xy_map[ndx][0]] += p
    pY = np.zeros(num_Ystates)
    for ndx, p in enumerate(probs):
        pY[s_to_xy_map[ndx][1]] += p
    return pX, pY

def get_mi(probs, s_to_xy_map):
    pX, pY = get_marginals(probs, s_to_xy_map)
    return entropy(pX, base=2) + entropy(pY, base=2) - entropy(probs, base=2)

def reverse_dict(d):
    return { v:k for k, v in d.items() }

def normalize_mx(mxbase):
    mx = mxbase.copy()
    np.fill_diagonal(mx, 0)
    mx/=mx.sum(axis=1).max()
    mx += np.diag(1-mx.sum(axis=1))
    return mx

# Return all partitions of items into k groups
def partitions(items, k):
    def split(indices):
        i=0
        for j in indices:
            yield items[i:j]
            i = j
        yield items[i:]

    for indices in combinations(range(1, len(items)), k-1):
        yield list(split(indices))

def compare_runs(init1, mx1, init2, mx2, viabilityfunc, iters):
    l1=[]
    l2=[]
    curp1 = init1.copy()
    curp2 = init2.copy()
    for citer in range(iters):
        l1.append(viabilityfunc(curp1))
        l2.append(viabilityfunc(curp2))
        curp1 = curp1.dot(mx1)
        curp2 = curp2.dot(mx2)

        if not np.isclose(np.sum(curp2) , 1.0):
            raise Exception('warning %d %0.5f'%(citer, np.sum(curp2)))

    l1, l2 = map(np.array, [l1, l2])
    plt.plot(l1, color='blue', label='Actual')
    plt.plot(l2, color='red', ls='--', label='Intervened')
    plt.legend()
    return l1, l2


# Function that returns a set of all possible interventions on the initial mutual information
# Returns list of interventions , where each item in list has form
#   (intervened_initial_distribution,
#    coarse_graining_function)
def get_all_interventions(num_locations, initp, state2id_dict, id2xy_id_dict, verbose=False):
    pX, pY = get_marginals(initp, id2xy_id_dict)
    items = list(range(num_locations)) + ['-',]
    savedfuncs = []
    all_partitions = []
    for k in range(1,num_locations+2):
        for part in list(partitions(items, k)):
            all_partitions.append(part)
            
    for partndx, part in enumerate(all_partitions):
        mapdict = { v:modndx for modndx, mod in enumerate(part) for v in mod}
        ixdict = {}
        func = {}
        for (agentloc, agenttarget, agentlevel, foodloc), ndx in state2id_dict.items():
            cgstate = (agentloc, agenttarget, agentlevel, mapdict[foodloc])
            if cgstate not in ixdict:
                ixdict[cgstate] = len(ixdict)
            func[ndx] = ixdict[cgstate]

        cgprobs = np.zeros(len(ixdict)+1)
        cgprobs2 = np.zeros(len(mapdict)+1)
        for (agentloc, agenttarget, agentlevel, foodloc), ndx in state2id_dict.items():
            cgstate = (agentloc, agenttarget, agentlevel, mapdict[foodloc])
            cgprobs[ixdict[cgstate]] += initp[ndx]
            cgprobs2[mapdict[foodloc]] += initp[ndx]

        newp = np.zeros(len(initp))
        for (agentloc, agenttarget, agentlevel, foodloc), ndx in state2id_dict.items():
            cgstate = (agentloc, agenttarget, agentlevel, mapdict[foodloc])
            a, b, c = cgprobs[ixdict[cgstate]],  pY[id2xy_id_dict[ndx][1]], (cgprobs2[mapdict[foodloc]])
            if np.isclose(a,0) or np.isclose(b,0):
                newp[ndx] = 0
            else:
                newp[ndx] = a*b/c

        savedfuncs.append((newp, part))
    return savedfuncs