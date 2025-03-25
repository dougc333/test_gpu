import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


class MyTrainDataset(Dataset):
  def __init__(self, size):
    self.size = size
    self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

  def __len__(self):
    return self.size

  def __getitem__(self, index):
    return self.data[index]

def ddp_setup(rank:int, world_size:int):
  import os
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  torch.cuda.set_device(rank)
  init_process_group(backend="gloo", rank=rank, world_size=world_size)

class Trainer:
  def __init__(self,
      model: torch.nn.Module,
      train_dataset: DataLoader,
      optimizer: torch.optim.Optimizer,
      gpu_id: int,
      save_every: int)-> None:
    self.gpu_id = gpu_id
    #might need f"cuda:{gpu_id}"
    self.model = model.to(gpu_id)
    self.train_data = train_dataset
    self.optimiser = optimizer
    self.save_every = save_every
    self.model = DDP(self.model, device_ids=[self.gpu_id])

  def _run_batch(self, source, targets):
    self.optimizer.zero_grad()
    output = self.model(source)
    loss = torch.nn.CrossEntropyLoss(output, targets)
    loss.backward()
    self.optimizer.step()

  def _run_epoch(self, epoch:int):
    b_sz = len(next(iter(self.train_data))[0])
    print(f"[GPU {self.gpu_id}] Starting epoch {epoch} Batch Size: {b_sz} Steps: {len(self.train_data)}")
    self.train_data.sampler.set_epoch(epoch)
    for source, targets in self.train_data:
      source = source.to(self.gpu_id)
      targets = targets.to(self.gpu_id)
      self._run_batch(source, targets)


  def _save_checkpoint(self, epoch):
    ckp = self.model.state_dict()
    PATH="checkpoint"
    torch.save(ckp, PATH)
    print(f"Saved checkpoint for epoch {epoch} at {PATH}")

  def train(self, epochs:int):
    for epoch in range(epochs):
      self._run_epoch(epoch)
      if self.gpu_id==0 and epoch % self.save_every == 0:
        self._safe_checkpoint(epoch)

def load_train_objs():
  train_set = MyTrainDataset(2048)
  model = torch.nn.Linear(20, 1)
  optimiser = torch.optim.SGD(model.parameters(), lr=1e-3)
  return train_set, model, optimiser


def prepare_dataloader(dataset: Dataset, batch_size:int):
  return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False,sampler = DistributedSampler(dataset) )


def main(rank:int, world_size:int, save_every:int, total_epochs:int, batch_size:int):
  ddp_setup(rank, world_size)
  dataset, model, optimizer = load_train_objs()
  train_data = prepare_dataloader(dataset, batch_size)
  trainer = Trainer(model, train_data, optimizer, rank, save_every)
  trainer.train(total_epochs)
  destroy_process_group()


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser("ddp  from pytorch examples")
  parser.add_argument("total epochs", type=int, help="total epochs to train")
  parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
  parser.add_argument("save_every", type=int, help="save every")
  args = parser.parse_args()
  world_size = torch.cuda.device_count()
  mp.spawn(main, args.save_every, args.total_epochs, args.batch_size, nprocs=world_size)
  