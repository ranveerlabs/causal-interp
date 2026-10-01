class Scrub:
    def __init__(self, parents, functions, important, features, population):
        self.parents = parents
        self.functions = functions
        self.important = important
        self.features = features
        self.population = tuple(population)
        self.classes = {}
        for node, feature in features.items():
            groups = {}
            for x in self.population:
                groups.setdefault(feature(x), []).append(x)
            self.classes[node] = groups

    def ordinary(self, node, ref):
        if not self.parents[node]:
            return self.functions[node](ref)
        return self.functions[node]([self.ordinary(p, ref) for p in self.parents[node]])

    def run(self, node, ref, rng):
        if not self.parents[node]:
            return self.functions[node](ref)
        donor = rng.choice(self.population)
        values = []
        for parent in self.parents[node]:
            if parent in self.important.get(node, ()):
                group = self.classes[parent][self.features[parent](ref)]
                values.append(self.run(parent, rng.choice(group), rng))
            else:
                values.append(self.ordinary(parent, donor))
        return self.functions[node](values)
