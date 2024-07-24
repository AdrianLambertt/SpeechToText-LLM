import os
import ast
import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from model import SpeechRecognition
from data_processing import Data, collate_fn_padd

# After trying to create my own pytorch based training system,I found that it was more 
# unstructured than Pytorch lightning. So the code has been switched to use lightning.
# Unfortunately pytorch Lightning is harder to understand, I have followed LearnedVectors "Train.py for this file".

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
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=6)
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
        

def main(args):
    h_params = SpeechRecognition.hyper_parameters
    h_params.update(args.hparams_override)
    model = SpeechRecognition(**h_params)

    if args.load_model:
        speech_module = SpeechModel.load_from_checkpoint(args.root_dir +'checkpoint.ckpt', model=model, args=args)
    else:
        speech_module = SpeechModel(model, args)

    logger = L.TensorBoardLogger(args.logdir, name='Speech_recognition')

    # Variables like Gpus, num_nodes and distributed_backend are removed as I will not need them, running a single GPU machine.
    # Gradient_clip_val is an argument used to limit exploding gradients, clipping them to 1.0
    # Checkpoints automatically enabled into the CWD: will add argument to change this.
    trainer = L.Trainer(
        max_epochs=args.epochs, logger=logger, gradient_clip_val=1.0,
        val_check_interval=args.valid_every, default_root_dir=args.root_dir
    )
    trainer.fit(speech_module)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-w', '--data_workers', default=0, type=int, help='n data loading workers, default 0 = main process only')

    # Training and validation files
    parser.add_argument('--train_file', default=None, required=True, type=str, help='JSON file to load training data')
    parser.add_argument('--valid_file', default=None, required=True, type=str, help='JSON file to load validation data')
    parser.add_argument('--valid_every', default=1000, required=False, type=int, help='Validation checks after N batches')

    # Directory and Path for models and logs
    parser.add_argument('--root_dir', required=True, type=str, help='The models Checkpoint location')
    parser.add_argument('--load_model', default=False, required=False, type=bool, help='Restore model from last checkpoint')
    parser.add_argument('--logdir', default='tb_logs', required=False, type=str, help='Path to save logs')

    # General
    parser.add_argument('--epochs', default=10, type=int, help='Number of total epochs to train for')
    parser.add_argument('--batch_size', default=64, type=int, help='Size of each batch')
    parser.add_argument('--learning rate', default=1e-3, type=float, help='Sets the learning rate')
    parser.add_argument('hparams_override', default='{}', type=str, required=False, help='Override hyper parameters. Create in dict form: {"num classes": 10 }')
    parser.add_argument('dparams_override', default='{}', type=str, required=False, help='Override data parameters. Create in dict form: {"sample rate": 200 }')

    args = parser.parse_args()
    args.hparams_override = ast.literal_eval(args.hparams_override)
    args.dparams_override = ast.literal_eval(args.dparams_overrde)


    if not os.path.isdir(os.path.dirname(args.root_dir)):
        raise Exception("The directory for path {} could not be found".format(args.root_dir))
    
    main(args)
