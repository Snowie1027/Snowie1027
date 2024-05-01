# 导入所需的库：`numpy` 用于数值计算，`matplotlib.pyplot` 用于绘图，`os` 用于处理文件路径
import numpy as np
import matplotlib.pyplot as plt
import os


# 设置数据文件夹路径
data_folder = '/Users/suxueqing/Desktop/计算机视觉第一次作业/fashion_mnist_data'

# 定义函数 `load_mnist_images`，用于加载 MNIST 数据集的图像数据
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)
    return data

# 定义函数 `load_mnist_labels`，用于加载 MNIST 数据集的标签数据
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

# 加载训练集和测试集的图像数据和标签数据
train_images = load_mnist_images(os.path.join(data_folder, 'train-images-idx3-ubyte'))
train_labels = load_mnist_labels(os.path.join(data_folder, 'train-labels-idx1-ubyte'))
test_images = load_mnist_images(os.path.join(data_folder, 't10k-images-idx3-ubyte'))
test_labels = load_mnist_labels(os.path.join(data_folder, 't10k-labels-idx1-ubyte'))

# 定义数据集类 `FMNISTDataset`，用于封装图像数据和标签数据，并提供获取数据样本和数据集长度的方法
class FMNISTDataset:
    def __init__(self, x, y):
        self.x = x.astype('float32')  # 将输入数据转换为float32类型
        self.y = y.astype('int64')    # 将标签数据转换为int64类型
        
    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x, y
    
    def __len__(self):
        return len(self.x)

# 定义数据加载器类 `DataLoader`，用于从数据集中按批次加载数据样本，并提供迭代功能
class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        
    def __iter__(self):
        indices = list(range(self.num_samples))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            batch_x = [self.dataset[ix][0] for ix in batch_indices]
            batch_y = [self.dataset[ix][1] for ix in batch_indices]
            yield np.array(batch_x), np.array(batch_y)

# 定义函数 `get_data`，用于创建训练数据加载器
def get_data(tr_images, tr_targets):
    train = FMNISTDataset(tr_images, tr_targets)
    trn_dl = DataLoader(train, batch_size=32, shuffle=True)
    return trn_dl

# 定义函数 `get_model`，用于创建模型。这里创建了一个包含两个全连接层的模型，每个层都有权重和偏置项
def get_model():
    model = {
        'layer1': {'weights': np.random.randn(28*28, 1000), 'bias': np.zeros(1000)},
        'layer2': {'weights': np.random.randn(1000, 10), 'bias': np.zeros(10)}
    }
    return model

# 定义函数 `forward`，用于执行前向传播过程。该函数接受输入数据 `x` 和模型 `model` 作为参数。在前向传播过程中，输入数据首先被展平为二维矩阵，并与第一个隐藏层的权重矩阵相乘，再加上偏置项，并通过 ReLU 激活函数进行非线性变换。得到的隐藏层输出作为输入传递给第二个输出层，与其权重矩阵相乘，再加上偏置项，得到模型的输出。最后，通过 softmax 函数将输出转换为概率分布。
def forward(x, model):
    flattened_x = x.reshape(x.shape[0], -1)  # 将输入 x 展平
    layer1_output = np.maximum(0, np.dot(flattened_x, model['layer1']['weights']) + model['layer1']['bias'])
    hidden_layer = layer1_output
    output = np.dot(hidden_layer, model['layer2']['weights']) + model['layer2']['bias']
    probs = softmax(output)
    return hidden_layer, probs

# 定义 softmax 函数，用于将模型的输出转换为概率分布
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# 定义计算损失函数 `calculate_loss`，用于计算模型的损失值。根据交叉熵损失函数的定义，首先从概率分布中选择正确类别的对数概率，然后对其求和并取平均
def calculate_loss(probs, targets):
    num_samples = probs.shape[0]
    corect_logprobs = -np.log(probs[range(num_samples), targets])
    loss = np.sum(corect_logprobs) / num_samples
    return loss

# 定义反向传播函数 `backward`，用于计算梯度。首先计算输出层的梯度，然后根据梯度和权重矩阵的转置计算隐藏层的梯度。最后，计算权重和偏置项的梯度，并返回梯度字典
def backward(hidden_layer, probs, model, targets):
    num_samples = hidden_layer.shape[0]
    delta = probs
    delta[range(num_samples), targets] -= 1
    delta /= num_samples
    dhidden = np.dot(delta, model['layer2']['weights'].T)
    dhidden[hidden_layer <= 0] = 0
    grads = {}
    grads['layer2_weights'] = np.dot(hidden_layer.T, delta)
    grads['layer2_bias'] = np.sum(delta, axis=0)
    grads['layer1_weights'] = np.dot(flattened_x.T, dhidden)
    grads['layer1_bias'] = np.sum(dhidden, axis=0)
    return grads

# 定义参数更新函数 `update_parameters`，用于根据梯度和学习率更新模型的参数
def update_parameters(model, grads, learning_rate):
    model['layer1']['weights'] -= learning_rate * grads['layer1_weights']
    model['layer1']['bias'] -= learning_rate * grads['layer1_bias']
    model['layer2']['weights'] -= learning_rate * grads['layer2_weights']
    model['layer2']['bias'] -= learning_rate * grads['layer2_bias']


# 定义训练单个批次的函数 `train_batch`，用于执行前向传播、计算损失、执行反向传播和更新模型参数
def train_batch(model, x, y, learning_rate):
    hidden_layer, probs = forward(x, model)
    loss = calculate_loss(probs, y)
    grads = backward(hidden_layer, probs, model, y)
    update_parameters(model, grads, learning_rate)
    return loss

# 定义计算准确率的函数 `calculate_accuracy`，用于在给定的数据集加载器上计算模型的准确率。通过前向传播获取预测结果，并与真实标签进行比较，统计正确预测的数量
def calculate_accuracy(model, dataset_loader):
    num_correct = 0
    num_samples = 0
    
    for batch_x, batch_y in dataset_loader:
        _, probs = forward(batch_x, model)
        predictions = np.argmax(probs, axis=1)
        num_correct += np.sum(predictions == batch_y)
        num_samples += batch_x.shape[0]
    
    accuracy = num_correct / num_samples
    return accuracy

# 定义训练模型函数
def train(model, learning_rate, num_epochs):
    batch_size = 32
    num_batches = train_images.shape[0] // batch_size
    losses, accuracies = [], []

    for epoch in range(num_epochs):
        loss = 0
        total_accuracy = 0

        for batch in range(num_batches):
            start_index = batch * batch_size
            end_index = (batch + 1) * batch_size
            x = train_images[start_index:end_index]
            y = train_labels[start_index:end_index]

            hidden_layer, probs = train_batch(x, y, model, learning_rate)

            loss += np.mean(-np.log(np.maximum(probs[range(x.shape[0]), y], 1e-7)))
            total_accuracy += calculate_accuracy(x, y, model)

        loss /= num_batches
        total_accuracy /= num_batches

        losses.append(loss)
        accuracies.append(total_accuracy)

        val_accuracy = calculate_accuracy(test_images, test_labels, model)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {total_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return losses, accuracies

# 在训练过程中，损失值和准确率被记录下来，并使用 Matplotlib 库进行可视化展示
num_epochs = 15
learning_rate = 0.001

losses, accuracies = train(get_model(), learning_rate, num_epochs)

epochs = np.arange(num_epochs) + 1
plt.figure(figsize=(20, 5))
plt.subplot(121)
plt.title('Loss value over increasing epochs')
plt.plot(epochs, losses, label='Training Loss')
plt.legend()
plt.subplot(122)
plt.title('Accuracy value over increasing epochs')
plt.plot(epochs, accuracies, label='Training Accuracy')
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.legend()

plt.show()
