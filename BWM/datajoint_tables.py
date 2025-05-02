import datajoint as dj
dj.config['database.host'] = 'datajoint-tengel.pni.princeton.edu'
dj.config['database.user'] = 'cl1704'
dj.config['database.password'] = 'wuxty2-mYdxej-kerxaq'
schema  = dj.schema('langdon_ibl_manifold')


@schema
class Responses(dj.Manual):
    definition = """
    eid: varchar(64)
    beryl : varchar(32)
    cosmos: varchar(32)
    n_time: int
    sigma: Decimal(4,2)
    trial_length: Decimal(3,2)
    ---
    n_neurons: int
    responses: longblob
    """

@schema
class ShuffledResponses(dj.Manual):
    definition = """
    eid: varchar(64)
    beryl : varchar(32)
    cosmos: varchar(32)
    n_time: int
    sigma: Decimal(4,2)
    trial_length: Decimal(3,2)
    shuffle: int
    ---
    n_neurons: int
    responses: longblob
    """

'''
Experiment 1: Uniform resampling test
'''
@schema
class Experiment_1(dj.Manual):
    definition = """
    beryl : varchar(32)
    cosmos: varchar(32)
    eid: varchar(64)
    n_time: int
    n_active: int
    sigma: float
    trial_length: Decimal(3,2)
    n_splits: int
    null_param:float
    ---
    variance: longblob
    p: float
    inertia: longblob
    null_inertia:longblob
    k: int
    activity_std: float
    n_neurons: int
    responses: longblob
    n_trials: int
    """

'''
Experiment 2: Rotation test
'''
@schema
class Experiment_2(dj.Manual):
    definition = """
    beryl : varchar(32)
    cosmos: varchar(32)
    eid: varchar(64)
    n_time: int
    n_active: int
    sigma: float
    trial_length: Decimal(3,2)
    ---
    inertia: longblob
    null_inertia: longblob
    activity_std: float
    n_neurons: int
    responses: longblob
    n_trials: int
    """


"""
Experiment 3: k-fold cross validation
"""
@schema
class Experiment_3(dj.Manual):
    definition = """
    beryl : varchar(32)
    cosmos: varchar(32)
    eid: varchar(64)
    n_time: int
    n_active: int
    sigma: float
    trial_length: Decimal(3,2)
    fold: int
    ---
    inertia: longblob
    null_inertia: longblob
    activity_std: float
    n_neurons: int
    responses: longblob
    n_trials: int
    """

"""
Single sessions, just choice, no stimulus side
"""
@schema
class Experiment_4(dj.Manual):
    definition = """
    beryl : varchar(32)
    cosmos: varchar(32)
    eid: varchar(64)
    n_time: int
    n_active: int
    sigma: float
    trial_length: Decimal(3,2)
    threshold: float
    active_threshold: float
    ---
    k: int
    null_k: longblob
    n_neurons: int
    responses: longblob
    n_trials: int
    """



'''
Experiment 5: Aggregating within cosmos regions and eids
'''
@schema
class Experiment_5(dj.Manual):
    definition = """
    cosmos: varchar(32)
    eid: varchar(64)
    n_time: int
    n_active: int
    sigma: float
    trial_length: Decimal(3,2)
    threshold: float
    active_threshold: float
    n_features: int
    ---
    k: int
    null_k: longblob
    n_neurons: int
    responses: longblob
    n_trials: int
    """


'''
Experiment 6: r2 vs r2
'''
@schema
class Experiment_6(dj.Manual):
    definition = """
    beryl : varchar(32)
    cosmos: varchar(32)
    eid: varchar(64)
    n_time: int
    n_active: int
    sigma: float
    trial_length: Decimal(3,2)
    threshold: float
    active_threshold: float
    n_features: int
    ---
    r2s: longblob
    null: longblob
    n_neurons: int
    responses: longblob
    n_trials: int
    """

"""
Single sessions. Clustering in selectivity space. Dimension computed on condition averages.
"""
@schema
class Experiment_7(dj.Manual):
    definition = """
    beryl : varchar(32)
    cosmos: varchar(32)
    eid: varchar(64)
    n_time: int
    n_active: int
    sigma: float
    trial_length: Decimal(3,2)
    active_threshold: float
    ---
    n_neurons: int
    scree: longblob
    inertia : longblob
    k: int
    n: int
    variance: float
    silhouette: float
    responses: longblob
    selectivity: longblob
    p_value: float
    """
@schema
class Dimension(dj.Manual):
    definition = """
    eid: varchar(64)
    beryl : varchar(32)
    cosmos : varchar(32)
    n_time: int
    n_active: int
    sigma: float
    trial_length: Decimal(3,2)
    truncation: int
    active_threshold: float
    ---
    n_neurons: int
    n_responsive: int
    n=NULL: int
    k=NULL: int
    inertia: longblob
    scree: longblob
    """

@schema
class DimensionShuffled(dj.Manual):
    definition = """
    eid: varchar(64)
    beryl : varchar(32)
    cosmos : varchar(32)
    n_time: int
    n_active: int
    sigma: float
    trial_length: Decimal(3,2)
    truncation: int
    active_threshold: float
    shuffle: int
    ---
    n_neurons: int
    n_responsive: int
    n=NULL: int
    k=NULL: int
    inertia: longblob
    scree: longblob
    """