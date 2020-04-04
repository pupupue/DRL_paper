from ddpg_tf import Agent
import gym
import numpy as np



import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from IPython.display import clear_output

class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure(figsize=(20, 10))
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        ### here is a bit tricky part
        ### the original code is made for the metrics accuracy
        ### I used different one, e.g. binary_accuracy, categorical_accuracy, mae, mse
        ### than you need to have following things instead of 'val_acc':
        ### 'val_binary_accuracy', 'val_categorical_accuracy', 'val_mean_absolute_error', 'val_mean_squared_error'
        self.acc.append(np.sqrt(logs.get('acc')))
        self.val_acc.append(np.sqrt(logs.get('val_acc')))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(20, 10))
        
        clear_output(wait=True)        
        plt.ticklabel_format(useOffset=False, style='plain') ### I am annoyed with "offset-tick" plotting that is common in Python
        
#         ax1.set_yscale('log') ### original
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.tick_params(labelsize=10) ### I added control of the fontsize
        ax1.legend(fontsize=9)
        
#         ax2.set_yscale('log') ### if you like, choose the metrics to be in log-scale too :)
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.tick_params(labelsize=10)  
        ax2.legend(fontsize=9)
        
        plt.show();



if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    agent = Agent(alpha=0.0001, beta=0.01, input_dims=[3], tau = 0.001, env=env, batch_size=64, layer1_size=400, layer2_size = 300, n_actions=1)
    np.random.seed(1337)


    score_history = []
    for i in range(1000):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, done)
            agent.learn()
            score += reward
            obs = new_state
        score_history.append(score)
        print('episode ', i, 'score %.2f' % score, '100 game average %.2f' % np.mean(score_history[-100:]))

    env.render()
    filename = 'pendulum.png'
    plotLearning(score_history, filename, window=100)



