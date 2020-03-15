def get_constraint_edge(device, target, lc=30, rc=30):
    all_constraint = []
    for i in range(len(target)):
        constraint = []
        for j in range(1, len(target[i][1:]), 2):
            l=max(target[i][1+j]-lc,0)
            r=max(target[i][1+j]+rc,l)
            constraint.append([l, r])
        all_constraint.append(constraint)
#    print(all_constraint)
    return all_constraint

def get_constraint_std(device, target, lc=0, rc=0):
    all_constraint = []
    for i in range(len(target)):
        constraint = []
        for j in range(1, len(target[i][1:]), 2):
            l=max(target[i][j]-lc,0)
            r=max(target[i][1+j]+rc,l)
            constraint.append([l, r])
        all_constraint.append(constraint)
#    print(all_constraint)
    return all_constraint
