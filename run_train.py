from train import CryptoResearch

cs = CryptoResearch()

cs.train();

cs.get_scores(cs.model, cs.X_train, cs.y_train)
