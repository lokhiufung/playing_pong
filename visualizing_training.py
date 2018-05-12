import matplotlib.pyplot as plt
import pandas as pd

training_history = pd.read_csv('training_history.csv', header=0)

def plot_training_history(training_history):
    fig, axs = plt.subplots(1, 2, figsize=(16, 9))
    axs[0].plot(training_history['loss'], color='red')
    axs[0].set_title('Loss')
    axs[1].plot(training_history['Q'], color='blue')
    axs[1].set_title('Estimated Q value')

    plt.show()

def main():
    plot_training_history(training_history)

if __name__ == '__main__':
    main()