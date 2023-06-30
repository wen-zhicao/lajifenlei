import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import math
import matplotlib.pyplot as plt
from PIL import Image

def createModel(input_shape,num_class):
    covn_base = tf.keras.applications.resnet.ResNet152(weights='imagenet',
                         include_top=False,
                         input_shape=input_shape)
    covn_base.trainable = True

    # 添加新的全连接层
    model = Sequential([
    covn_base,
    GlobalAveragePooling2D(),
    Dense(num_class, activation='softmax'),
    ])
    return model

# 定义数据增强器
train_datagen = ImageDataGenerator(rescale=1./255,#将像素值缩放到 0 到 1 之间。
                  rotation_range=20,#随机旋转图片的角度范围为 -20 到 20 度。
                  width_shift_range=0.2,#随机水平平移图片的宽度范围为 -20% 到 20%。
                  height_shift_range=0.2,#随机垂直平移图片的高度范围为 -20% 到 20%。
                  shear_range=0.2,#随机剪切变换的程度为 -20% 到 20%。
                  zoom_range=0.2,#随机缩放图片的尺寸为 80% 到 120%。
                  horizontal_flip=True,#随机水平翻转图片。
                  validation_split=0.3)

test_datagen = ImageDataGenerator(rescale=1./255)
# 生成训练数据和验证数据
train_generator = train_datagen.flow_from_directory('/content/drive/MyDrive/garbage/train',
                          target_size=(224, 224),
                          batch_size=32,
                          shuffle=True,
                          class_mode='categorical',
                          subset='training'
)

val_generator = train_datagen.flow_from_directory('/content/drive/MyDrive/garbage/train',
                         target_size=(224, 224),
                         batch_size=32,
                         shuffle=False,
                         class_mode='categorical',
                         subset='validation')

test_generator = test_datagen.flow_from_directory('/content/drive/MyDrive/garbage/test',
                          target_size=(224, 224),
                          batch_size=32,
                          shuffle=False,
                          class_mode='categorical')

# 训练模型
input_shape = (224, 224, 3)
num_class = 40
model=createModel(input_shape, num_class)
model.summary()
model.compile(optimizer=tf.optimizers.SGD(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator,
                steps_per_epoch=math.ceil(train_generator.n/train_generator.batch_size),
                epochs=6,
                callbacks=[keras.callbacks.EarlyStopping(patience=3)],
                validation_data=val_generator,
                verbose=1)

model.save("/content/drive/MyDrive/model.h5")

#损失函数下降曲线
plt.plot(history.history['loss'])

# 预测测试集
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

def plotcm(cm):
  classes = [str(i) for i in range(40)]
  labels = range(40)
  plt.matshow(cm, cmap=plt.cm.Blues)
  plt.title('混淆矩阵')
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes)
  plt.yticks(tick_marks, classes)
  for x in range(len(cm)):
    for y in range(len(cm)):
      plt.annotate(cm[x, y], xy=(x, y),horizontalalignment='center',verticalalignment='center')
  plt.grid(True, which='minor', linestyle='-')
  plt.show()
# 输出混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
plotcm(cm)

# 输出分类报告
target_names = list(test_generator.class_indices.keys())
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

# 输出准确率、精确率、召回率
accuracy = (y_pred == y_true).sum() / len(y_true)
precision = np.diag(cm) / cm.sum(axis=0)
recall = np.diag(cm) / cm.sum(axis=1)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# 输出错误分类的部分数据
misclassified_idx = np.where(y_pred != y_true)[0]
misclassified_samples = np.random.choice(misclassified_idx, size=5, replace=False)
print("Misclassified Samples:")
for idx in misclassified_samples:
    img_path = test_generator.filepaths[idx]
    true_label = target_names[y_true[idx]]
    pred_label = target_names[y_pred[idx]]
    print(f"True Label: {true_label}, Predicted Label: {pred_label}")
    display(Image.open(img_path))