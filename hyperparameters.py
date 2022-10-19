import random


class AbstractParameterGenerator:
    def name(self):
        raise NotImplemented()

    def generate(self, meta_params):
        raise NotImplemented()


class RandomParameterGenerator(AbstractParameterGenerator):

    def __init__(self) -> None:
        self.seed = 0

    def next_seed(self):
        self.seed += 1

    def name(self):
        return 'RandomParameterGenerator'

    def generate(self, meta_params):
        concrete_params = {}
        random.seed(self.seed)
        self.next_seed()
        for key, desc in meta_params.items():
            if type(desc) is list:
                concrete_params[key] = random.choice(desc)
            elif type(desc) is dict and desc['type'] == 'discrete':
                concrete_params[key] = random.randint(desc['min'], desc['max'])
            elif type(desc) is dict and desc['type'] == 'continuous':
                if desc['distribution'] == 'uniform':
                    concrete_params[key] = round(random.uniform(desc['min'], desc['max']),3)
                else:
                    raise NotImplemented("continuous distribution generator not implemented ")
            elif type(desc) is dict and desc['type'] == 'matrix':
                param_matrix = []
                layers = 0
                num_filter = 1
                filter_size = 1
                pool_window = 1
                for k, v in desc['in_params'].items():
                    if k == 'layers':
                        layers = random.randint(v['min'], v['max'])
                    elif k == 'filter_params':
                        for layer in range(layers):
                            for in_key, value in v.items():
                                if in_key == 'num_filters':
                                    num_filter = random.randint(value['min'], value['max'])
                                elif in_key == 'filter_size':
                                    filter_size = random.randrange(value['min'], value['max'], 2)
                                elif in_key == 'pool_window':
                                    pool_window = random.randint(value['min'], value['max'])
                            param_matrix.append([num_filter, filter_size, pool_window])
                concrete_params['param'] = param_matrix
            else:
                concrete_params[key] = desc
                # raise NotImplemented(f"Type of parameter ({desc['type']}) not implemented")
        return concrete_params
