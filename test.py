x=np.random.random_sample((7000, 7000))
y=np.random.random_sample((7000,))         
dd=deepnetwork(x, y, learning_rate=0.01, bias=0.5)
dd.train(x, y, epochs=25)
