import json


def save_seed(inpdict, filename="utils/seed.json"):
    """ saves seed into json. Called whenever a new seed is needed."""
    seed_list = ["random.seed", "np.random.seed", "np.random.default_rng", "rfi.sample.seed"]
    dic = {x: inpdict[x] for x in seed_list}
    with open(filename, "w") as f:
        json.dump(dic, f)


def load_seed(filename="utils/seed.json"):
    """ loads seed json file. Called by all scripts that need the shared seed value """
    with open(filename, "r") as f:
        data = json.load(f)
        return data
        
