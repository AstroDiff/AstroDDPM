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
    