import torch


class _CalcMeterHelper:
    def __init__(self):
        pass

    # https://github.com/qubvel/segmentation_models.pytorch
    @staticmethod
    def take_channels(*xs, ignore_channels=None):
        if ignore_channels is None:
            return xs
        else:
            channels = [
                channel
                for channel in range(xs[0].shape[1])
                if channel not in ignore_channels
            ]
            xs = [
                torch.index_select(
                    x, dim=1, index=torch.tensor(channels).to(x.device)
                )
                for x in xs
            ]
            return xs

    # https://github.com/qubvel/segmentation_models.pytorch
    @staticmethod
    def threshold(x, threshold=None):
        if threshold is not None:
            return (x > threshold).type(x.dtype)
        else:
            return x


class GlobalDice:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.__calc_meter_helper = _CalcMeterHelper()

    # https://github.com/qubvel/segmentation_models.pytorch
    def _f_score(self, pr, gt, threshold=None, ignore_channels=None):
        """Calculate F-score between ground truth and prediction
        Args:
            pr (torch.Tensor): predicted tensor
            gt (torch.Tensor):  ground truth tensor
            threshold: threshold for outputs binarization
        Returns:
            float: F score
        """

        pr = self.__calc_meter_helper.threshold(pr, threshold=threshold)
        pr, gt = self.__calc_meter_helper.take_channels(
            pr, gt, ignore_channels=ignore_channels
        )

        tp = torch.sum(gt * pr)
        fp = torch.sum(pr) - tp
        fn = torch.sum(gt) - tp
        return tp, fp, fn

    def update(self, pr, gt):
        r = self._f_score(pr, gt, threshold=0.5)
        self.tp += r[0].cpu().item()
        self.fp += r[1].cpu().item()
        self.fn += r[2].cpu().item()

    def get_metric(self):
        beta = 1
        eps = 1e-7
        score = ((1 + beta ** 2) * self.tp + eps) / (
            (1 + beta ** 2) * self.tp + beta ** 2 * self.fn + self.fp + eps
        )
        return score

    def reset(self):
        self.tp = 0
        self.fn = 0
        self.fp = 0


class GlobalIoU:
    def __init__(self):
        self.intersection = 0
        self.union = 0
        self.__calc_meter_helper = _CalcMeterHelper()

    # https://github.com/qubvel/segmentation_models.pytorch
    def _iou(self, pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
        """Calculate Intersection over Union between ground truth and prediction
        Args:
            pr (torch.Tensor): predicted tensor
            gt (torch.Tensor):  ground truth tensor
            eps (float): epsilon to avoid zero division
            threshold: threshold for outputs binarization
        Returns:
            float: IoU (Jaccard) score
        """

        pr = self.__calc_meter_helper.threshold(pr, threshold=threshold)
        pr, gt = self.__calc_meter_helper.take_channels(
            pr, gt, ignore_channels=ignore_channels
        )

        intersection = torch.sum(gt * pr)
        union = torch.sum(gt) + torch.sum(pr) - intersection + eps

        return intersection, union

    def update(self, pr, gt):
        r = self._iou(pr, gt, threshold=0.5)
        self.intersection += r[0].cpu().item()
        self.union += r[1].cpu().item()

    def get_metric(self):
        eps = 1e-7
        return (self.intersection + eps) / self.union

    def reset(self):
        self.intersection = 0
        self.union = 0
