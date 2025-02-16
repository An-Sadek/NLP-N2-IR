class TesetDataset:

    def __init__(self):
        pass 

    def __len__(self):
        return 10
    
    def __getitem__(self, idx):
        arr = [i for i in range(10)]
        return arr[idx]
    

test = TesetDataset()
print(len(test))
print(test[5])