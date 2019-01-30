x=np.random.randnom_sample((7000, 7000))
y=np.random.random((7000,))         
dd=deepnetwork(x, y)
dd.train(x, y, 20)
