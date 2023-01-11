from torch.utils import data
import torch
import logging
import tqdm
from configuration import Configuration
import learn2learn as l2l
from model import Model
from metadata import LWANDataset
LOGGER = logging.getLogger(__name__)

class METALMTC:
    def __init__(self):
        super().__init__()
        self.alpha = Configuration['model']['alpha']
        self.beta = Configuration['model']['beta']
        self.epochs = Configuration['model']['epochs']
        self.meta_batch_size = Configuration['model']['batch_size']
        self.adaptation_steps = Configuration['model']['adaptation_steps']
        self.dataset = LWANDataset(num_task=Configuration['model']['num_task'],
                              p=Configuration['model']['p'],
                              N=Configuration['model']['N'],
                              K=Configuration['model']['K'])
        self.dataloader = data.DataLoader(self.dataset,
                                          batch_size=Configuration['model']['batch_size'],
                                          shuffle=True)

    def fast_adapt(self, support_x, support_y, query_x, query_y, learner, loss, steps):
        for step in range(steps):
            predicted = learner(support_x)
            train_error = loss(predicted, support_y)
            train_error.backward()
            learner.adapt(loss=train_error)

        prediction = learner(query_x)
        query_error = loss(prediction, query_y)
        return query_error

    def train(self):
        model = Model(n_classes=4654, dropout_rate=0.5)
        model = model.cuda()
        maml = l2l.algorithms.MAML(model, lr=self.alpha, first_order = False)
        opt = torch.optim.Adam(maml.parameters(), lr=self.beta)
        loss = torch.nn.functional.binary_cross_entropy
        for epoch in range(Configuration['model']['epochs']):
            number = 0
            error = 0
            with tqdm.tqdm(self.dataloader,unit="batch") as loop:
                for support_x, support_y, query_x, query_y in loop:
                    #one batch
                    batch_error = 0
                    opt.zero_grad()
                    #遍历task 更新同一个clone后的模型
                    for task in range(self.meta_batch_size):
                        learner = maml.clone()
                        tmp_support_x, tmp_support_y, tmp_query_x, tmp_query_y = support_x[task,:,:], support_y[task,:,:], query_x[task,:,:], query_y[task,:,:]
                        tmp_support_x, tmp_support_y, tmp_query_x, tmp_query_y = torch.squeeze(tmp_support_x, dim=0), torch.squeeze(tmp_support_y, dim=0), torch.squeeze(tmp_query_x, dim=0), torch.squeeze(tmp_query_y,dim=0)
                        query_error = self.fast_adapt(tmp_support_x, tmp_support_y, tmp_query_x, tmp_query_y, learner, loss, self.adaptation_steps)
                        query_error.backward()
                        batch_error += query_error
                    batch_error = batch_error / self.meta_batch_size
                    for p in maml.parameters():
                        p.grad.data.mul_(1.0 / self.meta_batch_size)
                    opt.step()
                    error += batch_error
                    number += 1
                loop.set_description(f'Epoch [{epoch + 1}/{20}]')
                loop.set_postfix(loss=batch_error/number)


        #fine-tune
        optimizer = torch.optim.Adam(model.parameters(), lr=Configuration['model']['lr'])
        fine_tune_generator = self.dataset.get_train()
        with tqdm.tqdm(fine_tune_generator, unit="batch") as loop:
            for X, y in loop:
                number += 1
                optimizer.zero_grad()
                y_hat = net(X)
                l = loss(y_hat, y)
                loss_sum += l
                optimizer.zero_grad()
                l.mean().backward()
                optimizer.step()
                loop.set_description(f'Epoch [{1}/{1}]')
                loop.set_postfix(loss=loss_sum / number)

        test_generator,test_targets = self.dataset.get_test()
        self.dataset.calculate_performance(model=model, generator=test_generator, true_targets=test_targets)


if __name__ == '__main__':
    Configuration.configure()
    METALMTC().train()
