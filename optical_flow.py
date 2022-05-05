import cv2
import gym
from utils import *


class FlowEnv(object):
    """ Environment that expands an image environment with motion flow. """

    def __init__(self, env):
        self.env = env
        self.previousState = np.zeros((250, 160, 3))

    def reset(self):
        observation = cv2.cvtColor(self.env.reset(), cv2.COLOR_BGR2GRAY)
        out = self.combineImages(observation, observation)
        self.previousState = observation
        return out

    def step(self, action):
        new_observation, reward, game_over, info = self.env.step(action)
        new_observation = cv2.cvtColor(new_observation, cv2.COLOR_BGR2GRAY)
        merged = self.combineImages(self.previousState, new_observation)
        self.previousState = new_observation
        return merged.astype(np.float32), reward, game_over, info

    def combineImages(self, previousState, currentState):

        shape = currentState.shape[0:2]
        flow = np.ones(shape, dtype=currentState.dtype) * 255

        opticalflow = cv2.calcOpticalFlowFarneback(previousState, currentState, None,
                                                   0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(opticalflow[..., 0], opticalflow[..., 1])
        ang = ang * 90 / np.pi
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        merged = cv2.cvtColor(cv2.merge((ang.astype(currentState.dtype),
                              flow,
                              mag.astype(currentState.dtype))), cv2.COLOR_HSV2BGR)
        merged = cv2.merge((merged, currentState))
        return merged

    def __getattr__(self, name):
        return getattr(self.env, name)


if __name__ == "__main__":
    env = gym.make("Skiing-v0")
    opticalflow = FlowEnv(env)
    out = opticalflow.reset()
    for _ in range(10):
        merged, reward, gameover, info = opticalflow.step(0)
        print(merged.shape)

