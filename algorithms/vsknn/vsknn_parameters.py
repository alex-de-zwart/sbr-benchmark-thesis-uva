import random

class VsknnParamSearch:
    """"
    Create parameter space and take a sample of size 100
    from the space
    """

    @staticmethod
    def get_parameters() -> []:

        k = [50, 100, 500, 1000, 1500]
        sample_size = [500, 1000, 2500]
        weighting = ['same', 'div', 'linear', 'quadratic', 'log']
        weighting_score = ['same', 'div', 'linear', 'quadratic', 'log']
        idf_weighting = [False, 1, 2, 5, 10]

        random.seed(10)

        params = []
        for n_neighbours in k:
            for size in sample_size:
                for weight in weighting:
                    for weight_score in weighting_score:
                        for idf_weight in idf_weighting:
                            params.append({'k': n_neighbours, 'sample_size': size, 'weighting': weight,
                                          'weighting_score': weight_score, 'idf_weighting': idf_weight})
        param_sample = random.sample(params, 100)
        return param_sample
