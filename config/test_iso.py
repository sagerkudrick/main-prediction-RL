import time
from stable_baselines3 import PPO
from simple_train import SimpleIsotopeEnv  # replace with your filename

env = SimpleIsotopeEnv(render=True)
model = PPO.load("isotope_upright_with_xyz_arrows", env=env)

obs, _ = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    time.sleep(1./240.)

env.close()
