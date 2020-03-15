import torch
# Now 
def OHEM(index, k):
    if (k<=0):
        return index
    lenght = len(index)
    available_index = torch.tensor([1]*(lenght))
    reserve=[]
    for i in range(lenght):
        if 1 == available_index[index[i]]:
            reserve.append(index[i])
            rm_s = max(index[i]-k, 0)
            rm_e = min(index[i]+k, lenght)
            available_index[rm_s:rm_e+1] = 0
        else:
            continue

        if torch.sum(available_index) <= 0:
            break

    return torch.tensor(reserve).long()
        
if __name__=="__main__":
    index=torch.tensor([3, 2, 0, 7, 5, 8, 1, 4, 6])
    print OHEM(index, 0) # [3, 2, 0, 7, 5, 8, 1, 4, 6]
    print OHEM(index, 1) # [3, 0, 7, 5 ]
    print OHEM(index, 2) # [3, 0, 7]
    print OHEM(index, 3) # [3, 7]
    print OHEM(index, 100) # [3]
