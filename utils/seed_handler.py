import json


def save_seed(rand=246, np_rand=4812, np_rng=42, filename="utils/seed.json"):
    """ saves seed into json. Called whenever a new seed is needed."""
    dic = {"random.seed": rand, "np.random.seed": np_rand,
           "np.random.default_rng": np_rng}
    with open(filename, "wb") as f:
        json.dump(dic, f)


def load_seed(filename="utils/seed.json"):
    """ loads seed json file. Called by all scripts that need the shared seed value """
    with open(filename, "rb") as f:
        # change datatype accordingly (numpy.random.random() returns a float)
        data = json.load(f)
        return data
        
