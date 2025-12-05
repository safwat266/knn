import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier



image = torch.tensor([
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1]
], dtype=torch.float32)

image_4d = image.unsqueeze(0).unsqueeze(0)    # 1×1×5×5


print("=== Image Input Tensor ===")
print(image_4d)
print("\n---------------------------------------\n")




class SimpleKNN(nn.Module):
    def __init__(self, n_neighbors=3):
        super(SimpleKNN, self).__init__()

        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        X_train = np.random.randint(0, 2, (20, 25))   # 20 samples × 25 features
        y_train = np.random.randint(0, 10, 20)        # 10 classes

        self.knn.fit(X_train, y_train)

    def forward(self, x):
        x = x.view(x.size(0), -1).numpy()

        out = self.knn.predict(x)

        return torch.tensor(out)



#  Test the KNN

model = SimpleKNN()

out = model(image_4d)

print("=== Output of KNN Model (10 classes) ===")
print(out)
