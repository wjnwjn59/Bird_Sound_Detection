import matplotlib.pyplot as plt

def visualize_training(history):
    train_loss, train_acc = history.history['loss'], history.history['accuracy'] 
    val_loss, val_acc = history.history['val_loss'], history.history['val_accuracy'] 

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1) 
    plt.xlabel('Epochs') 
    plt.ylabel('Loss') 
    plt.title('Training loss') 
    plt.plot(train_loss, color='green') 

    plt.subplot(2, 2, 2) 
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training accuracy')
    plt.plot(train_acc, color='orange') 

    plt.subplot(2, 2, 3)
    plt.xlabel('Epochs') 
    plt.ylabel('Loss') 
    plt.title('Validation loss') 
    plt.plot(val_loss, color='green') 

    plt.subplot(2, 2, 4) 
    plt.xlabel('Epochs') 
    plt.ylabel('Accuracy')
    plt.title('Validation accuracy')
    plt.plot(val_acc, color='orange') 
    plt.show() 