import torch


def to_cuda(samples, targets, device):
    return samples.to(device, non_blocking=True), targets.to(device, non_blocking=True)


class DataPrefetcher:
    def __init__(self, loader, device, prefetch=True):
        """
        Preloads data from a loader and moves it to the specified device for faster processing.
        """
        self.loader = iter(loader)
        self.device = device
        self.prefetch = prefetch
        if self.prefetch:
            self.stream = torch.cuda.Stream()
            self._preload()
        else:
            self.stream = None
            self.next_samples = None
            self.next_targets = None

    def _preload(self):
        try:
            samples, targets = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(samples, targets, self.device)

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                targets.record_stream(torch.cuda.current_stream())
            self._preload()
        else:
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets