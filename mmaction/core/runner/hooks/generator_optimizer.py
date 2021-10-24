

from torch.nn.utils import clip_grad
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class GeneratorOptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            if hasattr(runner.model, 'module'):
                params = runner.model.module.generator.parameters()
            else:
                params = runner.model.generator.parameters()
            grad_norm = self.clip_grads(params)
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        # print(list(runner.model.module.generator.parameters())[0][0])
        #exit()
        runner.optimizer.step()