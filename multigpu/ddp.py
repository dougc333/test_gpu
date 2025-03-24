import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.distributed import init_process_group, destroy_process_group 


class MyTrainDataset:
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
      optimiser: torch.optim.Optimizer,
      gpu_id: int,
      save_every: int)-> None:
    self.model = model
    self.train_data = train_dataset
    self.optimiser = optimiser
    self.gpu_id = gpu_id
    self.save_every = save_every

  def _run_batch(self, source, targets):
    self.optimiser.zero_grad()
    output = self.model(source)
    loss = torch.nn.CrossEntropyLoss()(output, targets)
    loss.backward()
    self.optimiser.step()

  def _safe_checkpoint(self, epoch):
    ckp = self.model.state_dict()
    torch.save(ckp, f"checkpoint_epoch_{epoch}.pth")
    print(f"Saved checkpoint for epoch {epoch}")


def load_train_objs():
  train_set = MyTrainDataset()
  model = torch.nn.Linear(1, 1)
  optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
  return train_set, model, optimiser


def prepare_dataloader(dataset: Dataset, batch_size:int):
  return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False,sampler = DistributedSampler(dataset) )


def main(rank:int, world_size:int, save_every:int, total_epochs:int, batch_size:int):
  ddp_setup()
  dataset, model, optimiser = Trainer.load_train_objs()
  train_data = Trainer.prepare_dataloader(dataset, batch_size)
  trainer = Trainer(model, train_data, optimiser, rank, save_every)
  trainer.train(total_epochs)
  destroy_process_group()


if __name__ == "__main__":
  main()