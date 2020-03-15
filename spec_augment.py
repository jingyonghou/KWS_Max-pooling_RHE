import random

def spec_augment(inputs, num_samples, specaug_t, specaug_f):
    B, T, D = inputs.size()
    prob = random.random()
    for i in range(0, B):
        end_idx = num_samples[i]
        t = random.randint(1, min(specaug_t, end_idx-1))
        t0 = random.randint(0, end_idx - t - 1)
    
        f = random.randint(1, specaug_f)
        f0 = random.randint(0, D - f - 1)
        if prob <= 0.33:
            inputs[i, t0:t0+t, :] = 0.0
        elif prob <= 0.66:
            inputs[i, :, f0:f0+f] = 0.0
        else:
            inputs[i, t0:t0+t, f0:f0+f] = 0.0
#        print("%d %d %d %d\n"%(t0,t,f0,f))
    return inputs
