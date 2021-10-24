
import torch
from torch.nn.utils import clip_grad
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class Alternative_Train_GeneratorOptimizerHook(Hook):

    def __init__(self, grad_clip=None, alternative_train_iters=[20, 20]):
        self.grad_clip = grad_clip
        assert len(alternative_train_iters) == 2
        # self.alternative_train_iters = alternative_train_iters
        self.prefix_sum1, self.prefix_sum2 = alternative_train_iters[0], sum(alternative_train_iters)

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()

        if runner.iter % self.prefix_sum2 < self.prefix_sum1 and runner.epoch > 3:
            if hasattr(runner.model, 'module'):
                zero_grad(runner.model.module.generator.fast_regression.parameters())
            else:
                zero_grad(runner.model.generator.fast_regression.parameters())
            runner.logger.info('train the static perturbation only')
        else:
            if hasattr(runner.model, 'module'):
                zero_grad(runner.model.module.generator.slow_decoder.parameters())
            else:
                zero_grad(runner.model.generator.slow_decoder.parameters())
            runner.logger.info('train the motion vectors only')

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

        runner.optimizer.step()


def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


if __name__ == '__main__':

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(1, 1, bias=False)
            self.fc2 = torch.nn.Linear(1, 1, bias=False)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x
    net = TestModel()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()

    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
    y = x*2*3 + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
    # torch can only train on Variable, so convert them to Variable

    # train the network
    for t in range(5):
        for p in net.fc1.parameters():
            print('fc1', p.data, end=',')
        for p in net.fc2.parameters():
            print('fc2', p.data)
        prediction = net(x)  # input x and predict based on x
        loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        if t % 2 == 0:
            zero_grad(net.fc1.parameters())
        else:
            zero_grad(net.fc2.parameters())
        optimizer.step()  # apply gradients









