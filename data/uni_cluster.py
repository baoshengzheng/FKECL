import pickle

all_list = []
thresh = 0.85
for i in range(10):
    begin = i * 25000
    with open(f'cluster_result/{begin}_{thresh}.pkl', 'rb') as f:
        all_list.extend(pickle.load(f))

with open(f'cluster_{thresh}.pkl', 'wb') as f:
    pickle.dump(all_list, f)
