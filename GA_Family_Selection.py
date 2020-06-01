from typing import List
import tensorflow as tf
import numpy as np
import os
from deap import base
from deap import creator
from deap import tools
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
from sklearn.metrics import roc_auc_score
import json
from operator import attrgetter
from keras.models import load_model
import random
import gc
import keras.backend as K
tqdm.pandas()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Do other imports now...
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

family_exp_weights = lambda x: (1 - np.exp(-x)) / (1 - np.exp(-1))



def create_output_folder(path: str) -> None:
    """
    Creates an output folder in the given path
    :param path:
    :return:
    """
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def read_dict(name: str):
    """
    :param name: path to a json dict
    :return: Dict
    """
    with open(name, 'r') as f:
        return json.load(f)


def single_point_mutation(individual):
    """
    Currently not used

    Flips a random bit in the given individual

    :param individual: GA Individual
    :return: Mutated Ind
    """
    size = len(individual)
    rand_index = random.randint(0, size - 1)

    # 0 if value was 1 , and 1 if value was 0
    rand_value = abs(individual[rand_index] - 1)

    individual[rand_index] = rand_value
    return individual,


def generate_child(ind1, ind2):
    """
    N point Crossover between two individuals
    :param ind1: GA Parent 1
    :param ind2: GA Parent 2
    :return: 2 new children
    """
    child = []

    # I assume that len(ind1)=len(ind2)

    # 50% chance to take a bit from Parent-1 and 50% to take a bit from Parent-2
    for i in range(len(ind1)):
        if random.random() < 0.5:
            child += [ind1[i]]
        else:
            child += [ind2[i]]

    return child


# Generating two children based on n_point_crossover
def n_point_crossover(ind1, ind2):
    return generate_child(ind1, ind2), generate_child(ind1, ind2)


def five_point_crossover(ind1, ind2):
    """
    5 Point crossover between two individuals
    :param ind1: Parent-1
    :param ind2: Parent-2
    :return: Two generated children
    """
    points = random.sample(range(1, len(ind1)), 4)
    points.sort()
    i1, i2, i3, i4 = points[0], points[1], points[2], points[3]
    ind1[i1:i2], ind1[i3:i4], ind2[i1:i2], ind2[i3:i4] = ind2[i1:i2], ind2[i3:i4], ind1[i1:i2], ind1[i3:i4]
    return ind1, ind2


def mixedCrossovers(ind1, ind2):
    """
    Crossover used in this Expr.
    50% chance for a 5-point crossover and 50% chance for n-point crossover.

    Notice that family-vector is not used in the crossover (stays as it was).
    :param ind1: Parent-1
    :param ind2: Parent-2
    :return:
    """
    features1, family1 = ind1
    features2, family2 = ind2

    crossoverSelectionThres = 0.5

    if random.random() < crossoverSelectionThres:
        new_features1, new_features2 = five_point_crossover(features1, features2)
    else:
        new_features1, new_features2 = n_point_crossover(features1, features2)

    new_ind1 = [new_features1, family1]
    new_ind2 = [new_features2, family2]
    return new_ind1, new_ind2


def save_list(lst, path):
    """
    Saves a list in text-form, so results can be read mid-expr.
    :param lst: Some list
    :param path: path to save
    :return: None
    """
    with open(path, "w") as f:
        for s in lst:
            f.write(str(s) + "\n")


def get_active_families(ind):
    """

    :param ind: feature-selection vector & family mask vector
    :return: Number of features that are active
    """
    _, family = ind

    return sum(family)


def set_fits_for_inds(inds, max_ones, max_family, auc_func):
    """
    Sets the fitness for the given generation of individuals

    :param inds: GA Individuals
    :param max_ones: Max number of 1's in the given generation
    :param max_family: Max number of families in the given generation
    :param auc_func: Function that maps inds to their AUC score.
    :return: None
    """

    # Weights for the fitness.
    AUC_WEIGHT = 0.8
    FAMILY_WEIGHT = 0.05
    ONES_WEIGHTS = 0.15

    auc_mapper = lambda indv: 1 - auc_func(indv)

    METRIC_INDS = list(map(auc_mapper, inds))
    max_metric_inds = max(METRIC_INDS)

    for ind, mse_ind in zip(inds, METRIC_INDS):
        fitness_score = AUC_WEIGHT * (mse_ind / max_metric_inds) + ONES_WEIGHTS * (sum(ind[0]) / max_ones)
        fitness_score += FAMILY_WEIGHT * family_exp_weights((sum(ind[1]) / max_family))
        ind.fitness.values = fitness_score,


def mutate_family_ind(ind, low, up, indpb_features, indpb_family):
    """
    Mutation is to redraw a bit with some given prob

    :param ind: GA individual
    :param low: Minimum value in vector (0 in a binary vector)
    :param up: Maximum value in vector (1 in a binary vector)
    :param indpb_features: Prob to mutate the feature-vector
    :param indpb_family: Prob to mutate the family-mask-vector
    :return: Mutated ind
    """
    features, family = ind

    for i in range(len(features)):
        if random.random() < indpb_features:
            features[i] = random.randint(low, up)

    for i in range(len(family)):
        if random.random() < indpb_family:
            family[i] = random.randint(low, up)

    return [features, family],


def get_ind_code(ind):
    """

    :param ind: GA Ind
    :return: Code that is used to save scores in the cache (dictionary key)
    """
    features, family = ind
    code = tuple(features), tuple(family)
    return code


def opt_tournament(individuals, k, auc_func, tournsize, fit_attr="fitness"):
    """
    :param individuals: Population
    :param k: How many individuals to select
    :param auc_func: function to calculate the AUC of every individual
    :param tournsize: How many individuals per tournament
    :param fit_attr: key for selection
    :return: k selected individuals
    """
    chosen = []
    sum_first = lambda x: sum(x[0])
    sum_second = lambda x: sum(x[1])

    max_ones = max(list(map(sum_first, individuals)))
    max_family = max(list(map(sum_second, individuals)))

    # getting k inds
    for _ in range(k):
        comps = tools.selRandom(individuals, tournsize)
        set_fits_for_inds(comps, max_ones, max_family, auc_func)
        chosen.append(max(comps, key=attrgetter(fit_attr)))

    return chosen


class FeatureFamilySelectionGA:

    def __init__(self, n_generations: int, population_size: int, mu_for_sampling: float, sigma_for_sampling: float,
                 crossover_prob: float, mutation_prob: float, ind_length, family_length, family_intervals,
                 random_state=42):
        """

        :param n_generations: Number of generations to run
        :param population_size: Population Size
        :param mu_for_sampling: Used to initialize a normal distribution
        :param sigma_for_sampling: Used to initialize a normal distribution
        :param crossover_prob: Crossover probability
        :param mutation_prob:  Mutation probability
        :param ind_length: Individual length
        :param random_state: Initial random seed

        """
        assert 0 <= crossover_prob <= 1, "ILLEGAL CROSSOVER PROBABILITY"
        assert 0 <= mutation_prob <= 1, "ILLEGAL MUTATION PROBABILITY"
        assert population_size > 0, "Population size must be a positive integer"
        assert n_generations > 0, "Number of generations must be a positive integer"
        assert mu_for_sampling > 0 and sigma_for_sampling > 0, 'Illegal selection params'

        # params
        self.n_generations = n_generations
        self.population_size = population_size
        self.mu_for_sampling = mu_for_sampling
        self.sigma_for_sampling = sigma_for_sampling
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.random_state = random_state
        self.family_length = family_length
        self.family_intervals = family_intervals

        # params

        # GA toolbox
        self.toolbox = base.Toolbox()

        # solving for minimum fitness
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # probability to generate '1' in a random selection
        self.one_prob = None

        self.fitness_dict = {}
        self.mask_dicts = {}
        self.pop_dict = {}
        self.std_gen = []
        self.mean_gen = []
        self.median_gen = []
        self.max_gen = []
        self.min_gen = []
        self.time_gen = []

        self.__set_ind_generator(max_ind_size=ind_length, family_size=self.family_length)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.gen_ind)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evalInd)
        self.toolbox.register("mate", mixedCrossovers)
        self.toolbox.register("mutate", mutate_family_ind, low=0, up=1,
                              indpb_features=0.005, indpb_family=0.01)
        self.toolbox.register("select", opt_tournament, tournsize=5)

    def reset_stats(self):
        """
        Resets saved-metrics.

        :return: None
        """
        self.fitness_dict = {}
        self.pop_dict = {}
        self.std_gen = []
        self.mean_gen = []
        self.median_gen = []
        self.max_gen = []
        self.min_gen = []
        self.time_gen = []

    def save_all_data(self, curr_generation, output_folder):
        """
        Saves metrics for the given generations in a given path.

        :param curr_generation: Current generation (int)
        :param output_folder: folder to save data in.
        :return: None
        """
        written_dict = {str(k): v for k, v in self.fitness_dict.items()}
        with open(output_folder + "fitness_dict.json", 'w') as file:
            json.dump(written_dict, file)

        with open(output_folder + "gens_dict.json", 'w') as file:
            json.dump(self.pop_dict, file)

        with open(output_folder + "am_alive", "w") as f:
            f.write("Still running at generation :" + str(curr_generation) + "\n")

        save_list(self.std_gen, output_folder + "std.txt")
        save_list(self.mean_gen, output_folder + "mean.txt")
        save_list(self.median_gen, output_folder + "median.txt")
        save_list(self.max_gen, output_folder + "max.txt")
        save_list(self.min_gen, output_folder + "min.txt")
        save_list(self.time_gen, output_folder + "time.txt")

    def __init_normal_dist(self, length_to_gen: int, family_len: int) -> List[list]:
        """
        Creates a random individual with a given length
        :param length_to_gen: How many '1/0' to generate
        :return: A random binary array of the given size
        """
        family = []

        feature_arr = []
        while len(feature_arr) < length_to_gen:
            feature_arr += [int(random.random() <= self.one_prob)]

        family_one_prob = 0.8

        while len(family) < family_len:
            family += [int(random.random() <= family_one_prob)]

        return [feature_arr, family]

    def __set_ind_generator(self, max_ind_size, family_size) -> None:
        """
        Sets the individual generator. This is done by sampling from a normal distribution (from the given __init__
        params) the probability of '1' to generate. For example:
        max_ind_size = 480
        mu = 120
        sigma = 40
        Assume randomly chose 130, then one_probability = 0.2708 = (130 / 480) = (normal_dist_sample / max_ind_size)

        :param max_ind_size: Size of the individual to generate
        :return: None
        """
        normal_dist_sample = np.random.normal(self.mu_for_sampling, self.sigma_for_sampling, 1000)
        self.one_prob = random.choice(normal_dist_sample) / max_ind_size
        self.toolbox.register("gen_ind", self.__init_normal_dist, max_ind_size, family_size)

    def __init_seeds(self) -> None:
        """
        Resets seeds back to the initial seed
        :return: None
        """
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        os.environ['PYTHONHASHSEED'] = str(self.random_state)

    def learn_only_pred(self, output_folder, context_val, y_val, classifier_path) -> None:
        """

        :param output_folder: Where to write the output to
        :param context_val: Validation Context [matrix]
        :param y_val: Target values for validation
        :param classifier_path: Path to a keras classifier that will be used
        used to load fresh weights for every fitness calculation (for retraining the network from scratch)

        :return:
        """
        self.reset_stats()
        self.__init_seeds()
        create_output_folder(output_folder)
        assert len(context_val) == len(y_val)

        classifier = load_model(classifier_path)
        get_auc_func = lambda ind: self.only_predict_get_auc(ind, context_val, y_val, classifier)
        self.__run_gens(output_folder, get_auc_func)

    def learn_only_pred_full_arch(self, output_folder: str, val_input, y_val, classifier_path: str) -> None:
        """
        Starts the learning process to the only-predict algorithm
        :param output_folder: Folder to save results in
        :param val_input: Validation Data
        :param y_val: Validation Labels
        :param classifier_path: Path to the Keras MLP classifier.
        :return: None
        """
        self.reset_stats()
        self.__init_seeds()
        create_output_folder(output_folder)
        input_size = 3

        # item, user and context (3)
        assert len(val_input) == input_size

        classifier = load_model(classifier_path)
        get_auc_func = lambda ind: self.only_predict_get_auc_full_arch(ind, val_input, y_val, classifier)
        self.__run_gens(output_folder, get_auc_func)

    def learn_full_train(self, output_folder, context_train, y_train, context_val, y_val, classifier_path,
                         fresh_weights_path):
        """
        Starts the learning process for the full-train algorithm
        :param output_folder: Folder to save results in
        :param context_train: Training Data
        :param y_train: Training labels
        :param context_val: Validation Data
        :param y_val: Validation Labels
        :param classifier_path: Path to the keras MLP classifier
        :param fresh_weights_path: Path to fresh-weights (non-trained) for the classifier.
        :return: None
        """
        self.reset_stats()
        self.__init_seeds()
        create_output_folder(output_folder)

        assert len(context_val) == len(y_val) and len(context_train) == len(y_train)
        # classifier = load_model(classifier_path)
        density = sum(y_train) / len(y_train)
        ones_weight = (1 - density) / density
        zero_weight = 1
        class_weights = {0: zero_weight, 1: ones_weight}
        get_auc_func = lambda ind: self.train_mlp_net_get_auc(ind, context_train, y_train, context_val, y_val,
                                                              classifier_path, fresh_weights_path, class_weights)
        self.__run_gens(output_folder, get_auc_func)

    def transform_array_by_ind_and_family(self, ind, arr):
        """
        Returns the input-arr (training/validation data) after applying the feature-selection via the feature-vector
        and the mask-vector.

        - Selected features are not modified.
        - Non selected features are replaced with 0 (Not removed to retain the architecture input dimensions)

        :param ind: Binary Vector for feature selection & family mask
        :param arr: Feature vector
        :return: The feature vector after performing feature-selection by the family mask and feature-selection vector
        """

        # saving masks, in order to not recalculate it again
        features, family = ind
        code = tuple(family)
        if code in self.mask_dicts:
            mask = self.mask_dicts[code]

        else:
            mask = np.array([0] * len(features))

            for family_mask_index in range(len(family)):
                family_interval = self.family_intervals[family_mask_index]
                for feature_index in family_interval:
                    mask[feature_index] = family[family_mask_index]

            self.mask_dicts[code] = mask

        return np.multiply(np.multiply(features, arr), mask)

    def train_mlp_net_get_auc(self, individual, context_train, y_train, context_val, y_val, classifier_path,
                              fresh_weights, class_weights):
        """

        :param individual: Ind to evaluate
        :param context_train: Training data
        :param y_train: Training labels
        :param context_val: Validation Data
        :param y_val: Validation Labels
        :param classifier_path: Path to a MLP classifier
        :param fresh_weights: Path to fresh weights
        :param class_weights: Class weights for training
        :return: AUC Score
        """

        if self.is_in_dict(individual):
            return self.get_fit(individual)

        best_model_path = '../data/expr_running_best_full_train.h5'
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min')
        mcp = ModelCheckpoint(best_model_path, save_best_only=True, mode='min')

        make_zero = lambda element: self.transform_array_by_ind_and_family(individual, element)
        x_val = np.array(list(map(make_zero, context_val)))
        x_train = np.array(list(map(make_zero, context_train)))

        classifier = load_model(classifier_path)
        classifier.load_weights(fresh_weights)

        classifier.fit(x_train, y_train, validation_data=(x_val, y_val),
                       epochs=100, batch_size=256, verbose=0, callbacks=[es, mcp], class_weight=class_weights)

        best_classifier = load_model(best_model_path)

        val_predict = best_classifier.predict(x_val, batch_size=256)
        res_auc = roc_auc_score(y_val, val_predict)
        self.save_fitness(individual, res_auc)

        # Clearing model, to avoid a memory leak.
        del best_classifier, es, mcp, x_val, x_train, classifier, val_predict
        gc.collect()
        K.clear_session()
        tf.compat.v1.reset_default_graph()

        return res_auc

    def __run_gens(self, outputFolder: str, get_auc_func) -> None:
        """
        :param outputFolder: Folder to save output data
        :param get_auc_func: Function that receives an individual and returns his AUC score
        :return: None
        """

        # Crossover probability & Mutation Probability
        CXPB, MUTPB = self.crossover_prob, self.mutation_prob
        pop = self.toolbox.population(n=self.population_size)

        for g in tqdm(range(0, self.n_generations)):
            start_time = time.time()

            self.pop_dict[g] = [list(x).copy() for x in pop]

            # A new generation

            print("-- Generation %i --" % g)

            # Note : The fitness is calculated inside of the select function, for better performance.
            # Meaning that the fitness is calculated only for the selected individuals in the tournament.

            # Select the next generation individuals

            offspring = self.toolbox.select(pop, len(pop), get_auc_func)

            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            fits = [ind.fitness.values[0] for ind in offspring]

            self.update_progress_arrays(offspring, fits, g, start_time, outputFolder)

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Changed to offsprings because fitness changes for the entire generation
            pop[:] = offspring

    def only_predict_get_auc_full_arch(self, individual, input_val, y_val, fully_trained_mlp) -> float:
        """
        Calculates the fitness of a given individual by predicting on the fully trained arch
        :param individual: Ind to predict on
        :param input_val: a tuple of 3 elements - [item_val, user_val, context_val]
        :param y_val: Validation Target
        :param fully_trained_mlp: Fully trained arch classifier
        :return: AUC of the prediction
        """
        if self.is_in_dict(individual):
            return self.get_fit(individual)

        item_val, user_val, con_val = input_val

        make_zero = lambda element: self.transform_array_by_ind_and_family(individual, element)
        x_cv = np.array(list(map(make_zero, con_val)))
        input_val_ind = [item_val, user_val, x_cv]

        val_predict = fully_trained_mlp.predict(input_val_ind)
        res_auc = roc_auc_score(y_val, val_predict)

        self.save_fitness(individual, res_auc)
        return res_auc

    def only_predict_get_auc(self, individual, context_val, y_val, fully_trained_mlp) -> float:
        """
        Calculates the fitness of a given individual by predicting on the fully trained mlp
        :param individual: Ind to predict on
        :param context_val: Validation context
        :param y_val: Validation Target
        :param fully_trained_mlp: Fully trained classifier
        :return: AUC of the prediction
        """

        if self.is_in_dict(individual):
            return self.get_fit(individual)

        make_zero = lambda element: self.transform_array_by_ind_and_family(individual, element)

        x_cv = np.array(list(map(make_zero, context_val)))

        val_predict = fully_trained_mlp.predict(x_cv)

        res_auc = roc_auc_score(y_val, val_predict)
        self.save_fitness(individual, res_auc)
        return res_auc

    def save_fitness(self, ind, val):
        """

        :param ind: GA individual to save
        :param val: metric to save
        :return: None
        """
        code = get_ind_code(ind)
        self.fitness_dict[code] = val

    def is_in_dict(self, ind) -> bool:
        """

        :param ind: GA individual
        :return: True iff the given individual has a value store in the cache (past metric)
        """
        code = get_ind_code(ind)
        return code in self.fitness_dict

    def get_fit(self, ind):
        """

        :param ind: GA individual
        :return: Saved metric
        """
        code = get_ind_code(ind)
        if code in self.fitness_dict:
            return self.fitness_dict[code]

    """
    Not used on purpose, evaluation of individuals is done inside the selection function (for performance reasons).
    Instead of evaluating the entire generation, we evaluate only the individuals that were randomly
    selected in the tournament selection, thus saving unnecessary training of the MLP network . 
    """

    def evalInd(self, individual, METRIC):
        pass

    def update_progress_arrays(self, pop, fits, g, gen_start, outputpath):
        """

        :param pop: current generation of GA individuals
        :param fits: list of fitness scores
        :param g: current generation number
        :param gen_start: time that this generation started
        :param outputpath: output to save the data
        :return: None
        """

        length = len(pop)

        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        min_feat = min(fits)
        max_feat = max(fits)
        median_feat = np.median(fits)

        self.min_gen += [min_feat]
        self.max_gen += [max_feat]
        self.mean_gen += [mean]
        self.std_gen += [std]
        self.median_gen += [median_feat]

        self.time_gen += [time.time() - gen_start]
        self.save_all_data(g, outputpath)


def get_train_test_val_split(arr):
    x_train, x_test = train_test_split(arr, test_size=0.3, shuffle=False)
    x_val, x_test = train_test_split(x_test, test_size=0.66, shuffle=False)
    return x_train, x_test, x_val


if __name__ == '__main__':

    # Family intervals for dataset
    family_intervals_ = read_dict('../data_hero_ds/family_intervals.json')

    index_to_name = read_dict('../data_hero_ds/Family_Indexes.json')
    index_to_interval = []
    for key in index_to_name.keys():
        index_to_interval += [family_intervals_[index_to_name[key]]]

    n_generations_ = 300
    n_population_size = 100
    sigma_for_init = 40
    CXPB_ = 0.65
    MUTPB_ = 1
    ind_length_ = 661
    number_of_families = len(family_intervals_.keys())
    mu_for_init = 0.25 * ind_length_

    fs = FeatureFamilySelectionGA(n_generations_, n_population_size, mu_for_init, sigma_for_init, CXPB_, MUTPB_,
                                  ind_length_, number_of_families, index_to_interval)

    context_train_, _, context_val_ = get_train_test_val_split(np.load('../data_hero_ds/X.npy'))
    y_train_, _, y_val_ = get_train_test_val_split(np.load('../data_hero_ds/y.npy'))

    assert len(context_train_) == len(y_train_) and len(context_val_) == len(y_val_)

    mlp_path = '../fresh_mlp_data/HEARO_fresh_mlp_200_100_50_1.h5'
    fresh_weights_ = '../fresh_mlp_data/HEARO_fresh_weights_200_100_50_1.h5'

    output_folder_ = '../RESULTS_HERO_DS/GA_FAMILY/GA_RES_20_5_FULL_TRAIN/'
    fs.learn_full_train(output_folder_, context_train_, y_train_, context_val_, y_val_, mlp_path, fresh_weights_)
