from A3C import *

class Worker(object):
    def __init__(self, name, a3c_model, env):
        self.name = name
        self.env = env.unwrapped

        self.a3c_model = A3C(env, name, a3c_model)

        self.moving_rewards = []

        self.episode_count = 0
        self.max_episodes = 2000
        self.max_steps = 500
        self.update_iter = 32
        self.gamma = 0.9

    def work(self, session, coordinator):
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not coordinator.should_stop() and self.episode_count < self.max_episodes:
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(self.max_steps):
                # if self.name == 'W_0':
                #     self.env.render()
                a = self.a3c_model.choose_action(s, session)
                s_next, r, done, info = self.env.step(a)
                done = True if ep_t == self.max_steps - 1 else False

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r+8)/8)    # normalize

                if total_step % self.update_iter == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = session.run(self.a3c_model.v, {self.a3c_model.s: s_next[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + self.gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.a3c_model.s: buffer_s,
                        self.a3c_model.a_his: buffer_a,
                        self.a3c_model.v_target: buffer_v_target,
                    }
                    self.a3c_model.update_global(feed_dict, session)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.a3c_model.pull_global(session)

                s = s_next
                total_step += 1
                if done:
                    if len(self.moving_rewards) == 0:  # record running episode reward
                        self.moving_rewards.append(ep_r)
                    else:
                        self.moving_rewards.append(0.9 * self.moving_rewards[-1] + 0.1 * ep_r)
                    if self.episode_count%10==0:
                        print(
                            "Worker: ", self.name,
                            "| Episode:", self.episode_count,
                            "| Rewards: %i" % self.moving_rewards[-1],
                              )
                    self.episode_count += 1
                    break
