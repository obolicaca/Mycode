""" 为模型准备输入的数据集类 """
import os
from torch.utils.data import Dataset, DataLoader


class DataSet(Dataset):
    def __init__(self, file_name='NCBI-disease', train=True):
        if file_name == 'NCBI-disease':
            file_path = r'./NERdata/NCBI-disease'
        elif file_name == 'BC4CHEMD':
            file_path = r'./NERdata/BC4CHEMD'
        else:
            file_path = r'./NERdata/BC5CDR-chem'

        if train == True:
            with open(os.path.join(file_path, 'train_dev.tsv'), 'r') as f:
                data = f.readlines()
        else:
            with open(os.path.join(file_path, 'test.tsv'), 'r') as f:
                data = f.readlines()

        self.content = []
        words = []
        tags = []
        for line in data:
            line_content = [i.strip().upper() for i in line.split()]
            if line_content == []:
                temp = list(zip(words, tags))
                self.content.append(temp)
                words = []
                tags = []
            else:
                words.append(line_content[0])
                tags.append(line_content[-1])

    def __len__(self):
        return len(self.content)

    def __getitem__(self, item):
        contents, target = zip(*self.content[item])
        return list(contents), list(target)


def collate_fn(batch):
    sentences, tags = zip(*batch)
    return sentences, tags


def DataLoad(file_name ='NCBI-disease',train=True):
    if train == True:
        batch_size = 32
    else:
        batch_size = 64
    dataset = DataSet(file_name,train)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    return dataloader


if __name__ == '__main__':
    dataset = DataLoad('BC5CDR-chem',True)             # 'NCBI-disease','BC4CHEMD','BC5CDR-chem'
    for index, (sentences, tags) in enumerate(dataset):
        print(index)
        print(sentences)
        print(tags)
        break
