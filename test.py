x=np.random.randn(70, 70)
y=np.random.randn(70,)          
dd=deepnetwork(x, y)
dd.train(x, y, 50)
