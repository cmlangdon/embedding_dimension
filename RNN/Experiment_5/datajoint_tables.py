import datajoint as dj
import torch
import numpy as np


dj.config['database.host'] = 'datajoint-tengel.pni.princeton.edu'
dj.config['database.user'] = 'cl1704'
dj.config['database.password'] = 'wuxty2-mYdxej-kerxaq'
# dj.config['display.limit'] = 100
# dj.config["enable_python_native_blobs"] = True
# dj.conn(reset=True)


schema = dj.schema('langdon_clustering')

@schema
class Experiment_1(dj.Manual):
    definition = """
       # model table
       model_id: char(8)                      # unique model id
       ---
       threshold: float
       lvar: decimal(6,4)
       dim: int
       mse_z: float
       k: int
       null_k: int
       w_rec: longblob
       w_in: longblob
       w_out: longblob
       bias: longblob
       responses: longblob
    """


@schema
class Experiment_2(dj.Manual):
    definition = """
       # model table
       model_id: char(8)                      # unique model id
       ---
       n_neurons: int                                 # number of neurons
       task: enum('ReLU','Tanh')
       task: enum('2AFC','CDDM','PWM')
       epochs:int
       mse_z: float
       n: int
       k: int
       d: int
       ars: decimal(4,2)
       w_rec: longblob
       w_in: longblob
       w_out: longblob
       bias: longblob
    """


@schema
class Experiment_3(dj.Manual):
    definition = """
       # model table
       model_id: char(8)                      # unique model id
       ---
       n_neurons: int                                 # number of neurons
       task: enum('2AFC','CDDM','PWM')
       epochs:int
       mse_z: float
       n: int
       k: int
       d: int
       ars: decimal(4,2)
       w_rec: longblob
       w_in: longblob
       w_out: longblob
       bias: longblob
    """

@schema
class Experiment_7(dj.Manual):
    definition = """
       # model table
       id: int
       ---
       n_inputs: int   
       truncation: int                              
       mse_z: float
       n: int
       k: int
       scree: longblob
       inertia: longblob
       w_rec: longblob
       w_in: longblob
       w_out: longblob
       bias: longblob
    """


