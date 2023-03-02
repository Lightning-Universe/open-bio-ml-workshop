import torch 


class DeviceStatsMonitor:
    r"""
    Automatically monitors and logs device stats during training stage. ``DeviceStatsMonitor``
    is a special callback as it requires a ``logger`` to passed as argument to ``Fabric``.
    """
    def __init__(self):
        self.step = 0

    def _get_and_log_device_stats(self, fabric, key: str):
        device = fabric.device
        if device.type == "cpu":
            return

        device_stats = torch.cuda.memory_stats(device)

        for logger in fabric.loggers:
            separator = logger.group_separator
            prefixed_device_stats = _prefix_metric_keys(device_stats, key, separator)
            logger.log_metrics(prefixed_device_stats, step=self.step)

    def on_train_batch_start(self, fabric):
        self._get_and_log_device_stats(fabric, "on_train_batch_start")

    def on_train_batch_end(self, fabric):
        self._get_and_log_device_stats(fabric, "on_train_batch_end")
        self.step += 1


def _prefix_metric_keys(metrics_dict, prefix: str, separator: str):
    return {prefix + separator + k: v for k, v in metrics_dict.items()}
