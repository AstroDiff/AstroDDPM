import numpy as np
import torch.optim as optim

## TODO FACTORIZE THE CODE

class WarmUp:
    '''Adds a warmup to the scheduler provided as argument'''
    def __init__(self, optimizer, scheduler, warmup, maxlr):
        self.scheduler=scheduler
        self.warmup=warmup
        self.maxlr=maxlr
        self.optimizer=optimizer
        self.current_step=0
    def step(self):
        self.current_step+=1
        if self.current_step<=self.warmup:
            for p in self.optimizer.param_groups:
                p['lr']=self.maxlr*self.current_step/self.warmup
        else:
            if self.scheduler is None:
                pass
            else:
                self.scheduler.step()
    def state_dict(self):
        res_dict =  self.__dict__.copy()
        res_dict.pop("optimizer")
        return res_dict
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self.scheduler.load_state_dict(state_dict["scheduler"])    

class InverseSquareRootScheduler:
    '''Inverse Square Root Scheduler'''
    def __init__(self, optimizer, warmup, maxlr, minlr=0, last_step=-1):
        self.warmup=warmup
        self.maxlr=maxlr
        self.optimizer=optimizer
        if last_step==-1:
            self.current_step=0
        else:
            self.current_step=last_step
        self.minlr=minlr
        self.config = {'type' : 'invsqrt', 'warmup' : warmup, 'maxlr' : maxlr, 'minlr' : minlr, 'last_step' : last_step}
    def step(self):
        self.current_step+=1
        if self.current_step<=self.warmup:
            for p in self.optimizer.param_groups:
                p['lr']=self.maxlr*self.current_step/self.warmup
        else:
            for p in self.optimizer.param_groups:
                p['lr']=max(self.maxlr/((self.current_step/(self.warmup+1))**0.5), self.minlr)
    def state_dict(self):
        res_dict =  self.__dict__.copy()
        res_dict.pop("optimizer")
        return res_dict
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

        
class InversePowerLawScheduler:
    '''Inverse Power Law Scheduler'''
    def __init__(self, optimizer, warmup, maxlr, minlr=0, last_step=-1, power=0.25):
        self.warmup=warmup
        self.maxlr=maxlr
        self.optimizer=optimizer
        self.power=power
        self.minlr=minlr
        if last_step==-1:
            self.current_step=0
        else:
            self.current_step=last_step
        self.config = {'type' : 'invpower', 'warmup' : warmup, 'maxlr' : maxlr, 'minlr' : minlr, 'last_step' : last_step, 'power' : power}
    def step(self):
        self.current_step+=1
        if self.current_step<=self.warmup:
            for p in self.optimizer.param_groups:
                p['lr']=self.maxlr*self.current_step/self.warmup
        else:
            for p in self.optimizer.param_groups:
                p['lr']=max(self.maxlr/((self.current_step/(self.warmup+1))**self.power), self.minlr)
    def state_dict(self):
        res_dict =  self.__dict__.copy()
        res_dict.pop("optimizer")
        return res_dict
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


class CosineAnnealingScheduler:
    '''Cosine Annealing Scheduler'''
    def __init__(self, optimizer, warmup, maxlr, maxstep = 1000, minlr=0, last_step=-1):
        self.warmup=warmup
        self.maxlr=maxlr
        self.optimizer=optimizer
        if last_step==-1:
            self.current_step=0
        else:
            self.current_step=last_step
        self.minlr=minlr
        self.maxstep=maxstep
        self.config = {'type' : 'cosine', 'warmup' : warmup, 'maxlr' : maxlr, 'minlr' : minlr, 'last_step' : last_step, 'maxstep' : maxstep}
    def step(self):
        self.current_step+=1
        if self.current_step<=self.warmup:
            for p in self.optimizer.param_groups:
                p['lr']=self.maxlr*self.current_step/self.warmup
        else:
            for p in self.optimizer.param_groups:
                p['lr']=max(self.maxlr*(1+np.cos(np.pi*(self.current_step-self.warmup)/(self.maxstep)))/2, self.minlr)
    def state_dict(self):
        res_dict =  self.__dict__.copy()
        res_dict.pop("optimizer")
        return res_dict
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

class LinearScheduler:
    '''Linear Scheduler'''
    def __init__(self, optimizer, warmup, maxlr, minlr=0, last_step=-1, maxstep=1000):
        self.warmup=warmup
        self.maxlr=maxlr
        self.optimizer=optimizer
        if last_step==-1:
            self.current_step=0
        else:
            self.current_step=last_step
        self.minlr=minlr
        self.maxstep=maxstep
        self.config = {'type' : 'linear', 'warmup' : warmup, 'maxlr' : maxlr, 'minlr' : minlr, 'last_step' : last_step, 'maxstep' : maxstep}
    def step(self):
        self.current_step+=1
        if self.current_step<=self.warmup:
            for p in self.optimizer.param_groups:
                p['lr']=self.maxlr*self.current_step/self.warmup
        else:
            for p in self.optimizer.param_groups:
                p['lr']=max(self.maxlr*(1-(self.current_step-self.warmup)/self.maxstep), self.minlr)
    def state_dict(self):
        res_dict =  self.__dict__.copy()
        res_dict.pop("optimizer")
        return res_dict
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

class StepScheduler:
    '''Step Scheduler'''
    def __init__(self, optimizer, warmup, maxlr, minlr=0,last_step = -1, step_period=100):
        self.warmup=warmup
        self.maxlr=maxlr
        self.optimizer=optimizer
        if last_step==-1:
            self.current_step=0
        else:
            self.current_step=last_step
        self.step_period=step_period
        self.minlr=minlr
        self.config = {'type' : 'step', 'warmup' : warmup, 'maxlr' : maxlr, 'minlr' : minlr, 'last_step' : last_step, 'step_period' : step_period}
    def step(self):
        self.current_step+=1
        if self.current_step<=self.warmup:
            for p in self.optimizer.param_groups:
                p['lr']=self.maxlr*self.current_step/self.warmup
        else:
            for p in self.optimizer.param_groups:
                p['lr']=max(self.maxlr*(0.5**(int((self.current_step-self.warmup)/self.step_period))), self.minlr)
    def state_dict(self):
        res_dict =  self.__dict__.copy()
        res_dict.pop("optimizer")
        return res_dict
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

def get_optimizer_and_scheduler(config, params):
    '''Returns the optimizer and scheduler from the config dict'''
    if "optimizer" not in config.keys():
        config["optimizer"] = {}
    if "scheduler" not in config.keys():
        config["scheduler"] = {}

    optimizer_config = config["optimizer"]
    scheduler_config = config["scheduler"]
    completed_one_optim = False
    completed_message_optim = "Config parameters incomplete for the given optimizer. Using default values for the following parameters: "
    if "type" not in optimizer_config.keys():
        optimizer_config["type"] = "adam"
        print('Optimizer type not specified. Using default optimizer: Adam')

    if optimizer_config["type"].lower() == "adam":
        if "lr" not in optimizer_config.keys():
            optimizer_config["lr"] = 1e-3
        if "weight_decay" not in optimizer_config.keys():
            optimizer_config["weight_decay"] = 0
        optimizer = optim.Adam(params, lr=optimizer_config["lr"], weight_decay=optimizer_config["weight_decay"])
    elif optimizer_config["type"].lower() == "sgd":
        if "lr" not in optimizer_config.keys():
            optimizer_config["lr"] = 1e-3
        if "weight_decay" not in optimizer_config.keys():
            optimizer_config["weight_decay"] = 0
        if "momentum" not in optimizer_config.keys():
            optimizer_config["momentum"] = 0
        optimizer = optim.SGD(params, lr=optimizer_config["lr"], weight_decay=optimizer_config["weight_decay"], momentum=optimizer_config["momentum"])
    elif optimizer_config["type"].lower() == "adamw":
        if "lr" not in optimizer_config.keys():
            optimizer_config["lr"] = 1e-3
        if "weight_decay" not in optimizer_config.keys():
            optimizer_config["weight_decay"] = 0
        optimizer = optim.AdamW(params, lr=optimizer_config["lr"], weight_decay=optimizer_config["weight_decay"])
    else:
        raise ValueError("Optimizer {} not implemented, chose Adam, AdamW or SGD".format(optimizer_config["type"]))

    def get_scheduler_parameters(scheduler_config):
        if "warmup" not in scheduler_config.keys():
            scheduler_config["warmup"] = 0
        if "maxlr" not in scheduler_config.keys():
            scheduler_config["maxlr"] = optimizer_config["lr"]
        if "minlr" not in scheduler_config.keys():
            scheduler_config["minlr"] = 0
        if "last_step" not in scheduler_config.keys():
            scheduler_config["last_step"] = -1
        if "step_period" not in scheduler_config.keys():
            scheduler_config["step_period"] = 100
        if "power" not in scheduler_config.keys():
            scheduler_config["power"] = 0.25
        if "maxstep" not in scheduler_config.keys():
            scheduler_config["maxstep"] = 1000
        return scheduler_config
    stable_config = get_scheduler_parameters(scheduler_config)

    if "type" not in scheduler_config.keys():
        scheduler = None
    elif scheduler_config["type"].lower() == "linear":
        scheduler = LinearScheduler(optimizer, warmup=stable_config["warmup"], maxlr=stable_config["maxlr"], minlr=stable_config["minlr"], last_step=stable_config["last_step"], maxstep=stable_config["maxstep"])
    elif scheduler_config["type"].lower() == "cosine":
        scheduler = CosineAnnealingScheduler(optimizer, warmup=stable_config["warmup"], maxlr=stable_config["maxlr"], minlr=stable_config["minlr"], last_step=stable_config["last_step"], maxstep=stable_config["maxstep"])
    elif scheduler_config["type"].lower() == "invsqrt":
        scheduler = InverseSquareRootScheduler(optimizer, warmup=stable_config["warmup"], maxlr=stable_config["maxlr"], minlr=stable_config["minlr"], last_step=stable_config["last_step"])
    elif scheduler_config["type"].lower() == "invpower":
        scheduler = InversePowerLawScheduler(optimizer, warmup=stable_config["warmup"], maxlr=stable_config["maxlr"], minlr=stable_config["minlr"], power=stable_config["power"], last_step=stable_config["last_step"])
    elif scheduler_config["type"].lower() == "step":
        scheduler = StepScheduler(optimizer, warmup=stable_config["warmup"], maxlr=stable_config["maxlr"], minlr=stable_config["minlr"], step_period=stable_config["step_period"], last_step=stable_config["last_step"])
    else:
        raise ValueError("Scheduler {} not implemented. Chose one from linear, cosine, invsqrt".format(scheduler_config["type"]))
    optimizer.config = optimizer_config
    return optimizer, scheduler