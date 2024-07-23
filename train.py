import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import SpeechRecognition
from data_processing import Data, collate_fn_padd
import lightning as L

#After trying to create my own pytorch based training system,
#I found that it was more unstructured than Pytorch lightning. So the code has been switched to use lightning.
#Splitting settings into functions as lightning does makes a much more readable class than using pytorch.

class SpeechModel(L.LightningModule):
    def __init__ (self, model, args):
        super(SpeechModel, self).__init__()
        self.model = model
        self.criteria = nn.CTCLoss(blank=28, zero_infinity=True)
        self.args = args


    def forward(self, x, hidden):
        return self.model(x, hidden)


    # Network set up with the Adam optimiser, with a scheduler which is set to reduce the learning rate as the loss does not decrease after 6 epochs, by a rate of 0.5
    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.model.parameters(), args.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=6)
        return [self.optimizer], [self.scheduler]
    

    # Data loading is created by putting the data location and variables into an argument then returning the data as DataLoader.
    def training_loader(self):
        d_params = Data.parameters
        d_params.update(self.args.dparams_override)
        train_dataset = Data(json_path = self.args.train_file, **d_params)
        return DataLoader(dataset= train_dataset, batch_size= self.args.batch_size, 
                          num_workers= self.args.data_workers, collate_fn= collate_fn_padd, pin_memory=True)
    
    
    def val_loader(self):
        d_params = Data.parameters
        d_params.update(self.args.dparams_override)
        val_dataset = Data(json_path = self.args.valid_file, **d_params, valid=True)
        return DataLoader(dataset=val_dataset, batch_size=self.args.batch_size, 
                          num_workers= self.args.data_workers, collate_fn= collate_fn_padd, pin_memory=True)
    

    def step(self, batch):
        spectograms, labels, input_lengths, label_lengths = batch
        bs = spectograms.shape[0] #Batch size
        hidden = self.model._init_hidden(bs)
        hn, c0 = hidden[0].to(self.device), hidden[1].to(self.device)
        output, _ = self(spectograms, (hn, c0))
        output = F.log_softmax(output, dim=2)
        loss = self.criteria(output, labels, input_lengths, label_lengths)
        return loss
    

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        logs = {'loss' : loss, 'lr': self.optimizer.param_groups[0]['lr']}
        return {'loss': loss, 'logs': logs}
    

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        return {'val_loss': loss}
    
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean() # Pulls val loss from dict into a 1D tensor, which is then computed for the mean.
        self.scheduler.step(avg_loss) # Alert the scheduler to the loss change
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log':tensorboard_logs}
    
    
    def checkpoint_callback(args):
        return L.ModelCheckpoint(
            filepath= args.save_model_path, save_top_k=True,
            verbose=True, monitor='val_loss', mode='min',
            prefix=''
        )
        

def main(args):
    h_params = SpeechRecognition.hyper_parameters
    h_params.update(args.hparams_override)

    model = SpeechRecognition(**h_params)

    if args.load_model_from:
        speech_module = SpeechModel.load_from_checkpoint(args.load_model_from, model=model, args=args)
    else:
        speech_module = SpeechModel(model, args)

