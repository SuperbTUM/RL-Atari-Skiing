import cv2
import numpy as np


class FlowEnv(object):
    """ Environment that expands an image environment with motion flow. """

    def __init__(self, env, verbose=False):
        self.env = env
        self.previousState = np.zeros((250, 160, 3))
        self.verbose = verbose

    def reset(self):
        observation = self.env.reset()
        out = self.combineImages(observation, observation)
        self.previousState = observation
        return out

    def step(self, action):
        new_observation, reward, game_over, info = self.env.step(action)
        merged = self.combineImages(self.previousState, new_observation)
        self.previousState = new_observation
        return merged, reward, game_over, info

    def combineImages(self, previousState, currentState):
        pb, pg, pr = cv2.split(previousState)
        cb, cg, cr = cv2.split(currentState)

        shape = cb.shape + (2,)
        b = np.zeros(shape, dtype=cb.dtype)
        g = np.zeros(shape, dtype=cb.dtype)
        r = np.zeros(shape, dtype=cb.dtype)

        b = cv2.calcOpticalFlowFarneback(pb, cb, b, 0.5, 1, 5, 5, 5, 1.1, 0)
        g = cv2.calcOpticalFlowFarneback(pg, cg, g, 0.5, 1, 5, 5, 5, 1.1, 0)
        r = cv2.calcOpticalFlowFarneback(pr, cr, r, 0.5, 1, 5, 5, 5, 1.1, 0)

        b1, b2 = cv2.split(b)
        g1, g2 = cv2.split(g)
        r1, r2 = cv2.split(r)

        merged = cv2.merge((b1, g1, r1,
                            b2, g2, r2,
                            cb.astype('float32') / 255,
                            cg.astype('float32') / 255,
                            cr.astype('float32') / 255))

        if self.verbose:
            cv2.imshow("ax1", merged[:, :, :3])
            cv2.imshow("ax2", merged[:, :, 3:6])
            cv2.imshow("img", merged[:, :, 6:])
            cv2.waitKey(1)
        return merged

    def __getattr__(self, name):
        return getattr(self.env, name)
