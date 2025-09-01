import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Load model
model = load_model('flower_model.h5')
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Prepare test data
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_data = datagen.flow_from_directory(
    'dataset',
    target_size=(180, 180),
    batch_size=32,
    subset='validation',
    class_mode='categorical',
    shuffle=False
)

# Predict
y_pred_probs = model.predict(test_data)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_data.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Purples')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=class_names))

# Show some sample predictions
for i in range(5):
    img, label = test_data[i]
    pred = model.predict(img)
    plt.imshow(img[0])
    plt.title(f'True: {class_names[np.argmax(label[0])]}, Pred: {class_names[np.argmax(pred[0])]}')
    plt.axis('off')
    plt.show()
