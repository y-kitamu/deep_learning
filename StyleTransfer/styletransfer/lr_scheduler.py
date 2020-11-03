def get_scheduler(default_lr=1e-4, lr_decay=5e-5):
    def _scheduler(epoch):
        return default_lr / (1.0 + lr_decay * epoch)
    return _scheduler
