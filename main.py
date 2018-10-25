from model import DQN
from train import bot_play, get_copy_var_ops, replay_train, env
from train import input_size, output_size, dis, REPLAY_MEMORY

import tensorflow as tf
import numpy as np
from collections import deque
import random


def main():
    max_episodes = 50000
    # store the previous observations in replay memory
    replay_buffer = deque()
    total_count = 0

    with tf.name_scope("network"):
        mainDQN = DQN(input_size=input_size, output_size=output_size, name="main")
        targetDQN = DQN(input_size=input_size, output_size=output_size, name="target")

    with tf.name_scope("train"):
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")

    with tf.name_scope("miscellaneous"):
        init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        sess.run(copy_ops)
        mainDQN.set_session(sess)
        targetDQN.set_session(sess)

        for episode in range(max_episodes):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0
            state = env.reset()

            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state=state))

                next_state, reward, done, _ = env.step(action)
                if done:
                    reward = -100

                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > 10000:
                    bot_play(mainDQN=mainDQN)
                    break

            print("Episode: {} steps: {}".format(episode, step_count))
            total_count = total_count + step_count
            if step_count > 10000:
                pass

            if episode % 10 == 1:
                for _ in range(50):
                    mini_batch = random.sample(replay_buffer, 10)
                    loss, _ = replay_train(mainDQN=mainDQN, targetDQN=targetDQN, train_batch=mini_batch)

                print("Loss: ", loss)
                sess.run(copy_ops)

        bot_play(mainDQN=mainDQN)

if __name__ == "__main__":
    main()
