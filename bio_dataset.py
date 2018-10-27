from torch.utils.data import Dataset
import numpy as np

class BioDataset(Dataset):
    def __init__(self, csv_file, split='train'):
        self.split = split
        
        if self.split not in ['train', 'test']:
            raise ValueError('Wrong split entered! Please use split="train", or split="test"')
    
        data_f = open(csv_file+split+'.csv', 'r')
        contents = data_f.read().splitlines()[1:]
        data_f.close()
        contents = [content.split(',')[2:] for content in contents]
        contents = [[float(entry) for entry in content] for content in contents]
#        if self.split == 'train':
#            contents = contents[787:]
#        elif self.split == 'test':
#            contents = contents[:787]
            
        self.contents = np.array(contents, 'float32')
        
    def __len__(self):
        return len(self.contents)
        
    def __getitem__(self, idx):
        return self.contents[idx]
        
class BioMixDataset(Dataset):
    def __init__(self, csv_file, gen_file, split='train'):
        self.split = split
        
        if self.split not in ['train', 'test']:
            raise ValueError('Wrong split entered! Please use split="train", or split="test"')
    
        data_f = open(csv_file+split+'.csv', 'r')
        contents = data_f.read().splitlines()[1:]
        data_f.close()
        contents = [content.split(',')[2:] for content in contents]
        contents = [[float(entry) for entry in content] for content in contents]
#        if self.split == 'train':
#            contents = contents[787:]
#        elif self.split == 'test':
#            contents = contents[:787]
            
        self.contents = np.array(contents, 'float32')
        if self.split == 'train':
            self.gen_data = np.load(gen_file)
            self.contents = np.concatenate((self.contents, self.gen_data), axis=0)
        
    def __len__(self):
        return len(self.contents)
        
    def __getitem__(self, idx):
        return self.contents[idx, :-1], self.contents[idx, -1]