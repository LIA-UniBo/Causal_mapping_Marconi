
class DatasetMarconi:
    def __init__(self):
        self.x = []
        self.y = []

    def append(self,x_data,y_data):
        self.x.append(x_data)
        self.y.append(y_data)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]