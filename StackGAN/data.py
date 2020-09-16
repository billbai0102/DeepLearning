import torch
from torch.utils.data import Dataset, DataLoader

class DataStream:
    def __init__(self, dl):
        self.dl = iter(dl)
        self.stream = torch.cuda.Stream()
        self.next_input = None
        self.next_target = None
        self.load_next_data()

    def load_next_data(self):
        try:
            self.next_input, next_target = next(self.dl)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        cur_input = self.next_input
        cur_target = self.next_target
        self.load_next_data()

        return cur_input, cur_target
