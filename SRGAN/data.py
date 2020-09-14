import torch


class DataStream:
    def __init__(self, dl):
        self.dl = iter(dl)
        self.stream = torch.cuda.Stream()

        self.next_input = None
        self.next_target = None

        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.dl)
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
        self.preload()

        return cur_input, cur_target
