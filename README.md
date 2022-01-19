# FoodVoiceRecognition
本文通用天池提供的20种不同种类的食物声音，通过对声音特征的提取，采用CNN架构，利用已标记的数据训练后，实现对食物声音种类进行预测

### 导入API

```python
import glob
import librosa.display
import librosa
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
import wget
```

### 数据准备：

#### 下载数据

```python
import wget
# 训练集数据
url_train = 'http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531887/train_sample.zip'
train_name = wget.download(url_train)
# 预测集数据
url_test = 'http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531887/test_a.zip'
test_name = wget.download(url_test)
```

找到下载目录，手动解压压缩包

#### 标签编码

```python
feature = []
label = []
# 建立类别标签，不同类别对应不同的数字。
label_dict = {'aloe': 0, 'burger': 1, 'cabbage': 2, 'candied_fruits': 3, 'carrots': 4, 'chips': 5, 'chocolate': 6, 'drinks': 7, 'fries': 8, 'grapes': 9, 'gummies': 10, 'ice-cream': 11, 'jelly': 12, 'noodles': 13, 'pickles': 14, 'pizza': 15, 'ribs': 16, 'salmon': 17, 'soup': 18, 'wings': 19}
# 将数字标签反相映射，对应声音类别，用于后面预测
label_dict_inv = {v: k for k, v in label_dict.items()}
```

#### 定义训练集信息提取函数

通过遍历文件，提取训练集中每个音频的梅尔谱以及对应的分类标签，梅尔谱包含音频的特征信息，将时间序列信息转换为二维图像信息，方便利用CNN训练

```python
def extract_features(parent_dir, sub_dirs, max_file=10, file_ext="*.wav"):
    c = 0
    label, feature = [], []
    for sub_dir in sub_dirs:
        # 遍历数据集的所有文件
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]):
            label_name = fn.split('\\')[0]
            label_name = label_name.split('/')[-1]
            label.extend([label_dict[label_name]])
            X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
            # 计算梅尔频谱(mel spectrogram),并把它作为特征
            mels = np.mean(librosa.feature.melspectrogram(
                y=X, sr=sample_rate).T, axis=0)
            feature.extend([mels])
    return [feature, label]
```

#### 提取训练集音频信息

```python
# 通过预览下载的数据文件夹与子文件夹，得到特征提取函数的重要路径参数
parent_dir = './train_sample/'
save_dir = "./"
folds = sub_dirs = np.array(['aloe', 'burger', 'cabbage', 'candied_fruits', 'carrots', 'chips', 'chocolate', 'drinks', 'fries', 'grapes', 'gummies', 'ice-cream', 'jelly', 'noodles', 'pickles', 'pizza', 'ribs', 'salmon', 'soup', 'wings'])
# 提取数据文件，并计算梅尔谱
temp = extract_features(parent_dir, sub_dirs, max_file=100)
```

![1.jpg](https://github.com/Cocytus-Leon/FoodVoiceRecognition_1/blob/main/20210413101648-aqxfuft-6.jpg)

#### 规整数据

```python
temp = np.array(temp)
data = temp.transpose()
```

#### 提取频谱信息

```python
X = np.vstack(data[:, 0])
```

#### 提取标签信息

```python
Y = np.array(data[:, 1])
```

#### 对标签值进行one-hot编码

```python
# 在Keras库中：to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示
Y = to_categorical(Y)
```

### 利用CNN进行训练

#### 训练集的划分

通过对训练集数据进行划分，一部分用于训练，另一部分用于评价CNN模型的训练效果

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, random_state=1, stratify=Y)
print('训练集的大小', len(X_train))
print('测试集的大小', len(X_test))
```

#### 数据规整

将频谱长度为128的数据序列规整为16×8的谱图

```python
X_train = X_train.reshape(-1, 16, 8, 1)
X_test = X_test.reshape(-1, 16, 8, 1)
```

#### CNN网络架构设计

搭建CNN网络架构，采用六层架构，即卷积层-池化层-卷积层-池化层-隐藏层-输出层，其中隐藏层神经元随机失活率设置为0.1，避免过拟合

模型训练参数的设置：优化方法为adam，采用多元交叉熵函数作为损失函数，将输出结果正确率作为评价指标

```python
model = Sequential()

# 输入的大小
input_dim = (16, 8, 1)

model.add(Conv2D(64, (3, 3), padding="same", activation="tanh", input_shape=input_dim))  # 卷积层
model.add(MaxPool2D(pool_size=(2, 2)))  # 最大池化
model.add(Conv2D(128, (3, 3), padding="same", activation="tanh"))  # 卷积层
model.add(MaxPool2D(pool_size=(2, 2)))  # 最大池化层
model.add(Dropout(0.1))
model.add(Flatten())  # 展开
model.add(Dense(1024, activation="tanh"))
model.add(Dense(20, activation="softmax"))  # 输出层：20个units输出20个类的概率

# 编译模型，设置损失函数，优化方法以及评价标准
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# %%
model.summary()
```

![2.jpg](https://github.com/Cocytus-Leon/FoodVoiceRecognition_1/blob/main/20210413103835-rggqh9s-7.jpg)

#### CNN的训练

设置20轮训练，每批次15组数据

```python
model.fit(X_train, Y_train, epochs=20, batch_size=15, validation_data=(X_test, Y_test))
```

![3.jpg](https://github.com/Cocytus-Leon/FoodVoiceRecognition_1/blob/main/20210413104219-8owurbv-8.jpg)

CNN网络训练好后，就可以用于分类预测了

由于各层相关参数是随手设置的，并没有去调整，故评价的正确率仅有0.38，读者可以手动调参进行优化

### 利用CNN进行预测

#### 定义预测集信息提取函数

由于预测集信息不包含标签信息，仅用于预测，故相比于训练集信息提取函数，预测集信息提取函数仅提取出音频的梅尔谱即可

```python
def extract_features(test_dir, file_ext="*.wav"):
    feature = []
    # 遍历数据集的所有文件
    for fn in tqdm(glob.glob(os.path.join(test_dir, file_ext))[:]):
        X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
        # 计算梅尔频谱(mel spectrogram),并把它作为特征
        mels = np.mean(librosa.feature.melspectrogram(
            y=X, sr=sample_rate).T, axis=0)
        feature.extend([mels])
    return feature
```

#### 提取预测集音频信息

```python
X_test = extract_features('./test_a/')
```

![4.jpg](https://github.com/Cocytus-Leon/FoodVoiceRecognition_1/blob/main/20210413110154-caf9rni-9.jpg)

#### 进行预测

```python
X_test = np.vstack(X_test)
predictions = model.predict(X_test.reshape(-1, 16, 8, 1))
```

#### 解码类别

```python
preds = np.argmax(predictions, axis=1)
preds = [label_dict_inv[x] for x in preds]
```

#### 写入文件

```python
path = glob.glob('./test_a/*.wav')
result = pd.DataFrame({'name': path, 'label': preds})

result['name'] = result['name'].apply(lambda x: x.split('/')[-1])
result.to_csv('submit.csv', index=None)
