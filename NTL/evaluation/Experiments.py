
from config.base import Config
from torch.utils.data import DataLoader
import numpy as np
class runExperiment():

    def __init__(self, model_configuration, exp_path):
        self.model_config = Config.from_dict(model_configuration)
        self.exp_path = exp_path

    def run_test(self, train_data,val_data,test_data, logger,contamination,query_num):

        model_class = self.model_config.model
        loss_class = self.model_config.loss
        optim_class = self.model_config.optimizer
        sched_class = self.model_config.scheduler
        stopper_class = self.model_config.early_stopper
        network = self.model_config.network
        trainer_class = self.model_config.trainer
        shuffle = self.model_config['shuffle'] if 'shuffle' in self.model_config else True


        try:
            x_dim = self.model_config['x_dim']
        except:
            x_dim = train_data.dim_features
        try:
            batch_size = self.model_config['batch_size']
        except:
            batch_size = int(np.ceil(len(train_data)/5))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle,
                                  drop_last=False)

        if len(val_data) == 0:
            val_loader = None
        else:
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                    drop_last=False)

        if len(test_data) == 0:
            test_loader = None
        else:
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                     drop_last=False)

        model = model_class(network(),x_dim, config=self.model_config)
        optimizer = optim_class(model.parameters(),
                                lr=self.model_config['learning_rate'], weight_decay=self.model_config['l2'])

        if sched_class is not None:
            scheduler = sched_class(optimizer)
        else:
            scheduler = None

        trainer = trainer_class(model, loss_function=loss_class(self.model_config['loss_temp']),
                         config=self.model_config)


        val_loss,val_auc,test_auc,test_ap,test_f1,test_score = \
            trainer.train(train_loader=train_loader,
                      contamination=contamination,query_num = query_num,
                      optimizer=optimizer, scheduler=scheduler,
                      validation_loader=val_loader, test_loader=test_loader, early_stopping=stopper_class,
                      logger=logger)

        return val_auc, test_auc, test_ap,test_f1, test_score
