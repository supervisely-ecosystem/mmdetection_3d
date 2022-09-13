import datetime
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger.text import TextLoggerHook
import supervisely_lib as sly
from sly_train_progress import add_progress_to_request
import sly_globals as g
import classes as cls


@HOOKS.register_module()
class SuperviselyLoggerHook(TextLoggerHook):
    def __init__(self,
                 by_epoch=True,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 interval_exp_name=1000):
        super(SuperviselyLoggerHook, self).__init__(by_epoch, interval, ignore_last, reset_flag, interval_exp_name)
        self.progress_epoch = None
        self.progress_iter = None
        self._lrs = []

    def _log_info(self, log_dict, runner):
        super(SuperviselyLoggerHook, self)._log_info(log_dict, runner)
        
        if log_dict['mode'] == 'train' and 'time' in log_dict.keys():
            self.time_sec_tot += (log_dict['time'] * self.interval)
            time_sec_avg = self.time_sec_tot / (runner.iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (len(runner.data_loader) * runner.max_epochs - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_dict['eta'] = eta_str

        if self.progress_epoch is None:
            self.progress_epoch = sly.Progress("Epochs", runner.max_epochs)
        if self.progress_iter is None:
            self.progress_iter = sly.Progress("Iterations", len(runner.data_loader))

        fields = []
        if log_dict['mode'] == 'train':
            self.progress_epoch.set_current_value(log_dict["epoch"])
            self.progress_iter.set((log_dict['iter'] - 1) % len(runner.data_loader) + 1, len(runner.data_loader))
            fields.append({"field": "data.eta", "payload": log_dict['eta']})

        fields.append({"field": "state.isValidation", "payload": log_dict['mode'] == 'val'})

        add_progress_to_request(fields, "Epoch", self.progress_epoch)
        add_progress_to_request(fields, "Iter", self.progress_iter)
        
        if log_dict['mode'] == 'train':
            epoch_float = float(self.progress_epoch.current - 1) + float(self.progress_iter.current) / float(self.progress_iter.total)
            
            fields.extend([
                {"field": "state.chartLR.series[0].data", "payload": [[epoch_float, round(log_dict["lr"], 6)]], "append": True},
                {"field": "state.chartLoss.series[0].data", "payload": [[epoch_float, round(log_dict["loss"], 6)]], "append": True},
                {"field": "state.chartTime.series[0].data", "payload": [[epoch_float, log_dict["time"]]], "append": True},
                {"field": "state.chartDataTime.series[0].data", "payload": [[epoch_float, log_dict["data_time"]]], "append": True},
                {"field": "state.chartMemory.series[0].data", "payload": [[epoch_float, log_dict["memory"]]], "append": True}
            ])
            
        if log_dict['mode'] == 'val':
            for class_ind, class_name in enumerate(cls.selected_classes):
                fields.extend([
                    {"field": f"state.chartAP_25.series[{class_ind}].data", "payload": [[log_dict["epoch"], log_dict[f"{class_name}_AP_0.25"]]], "append": True},
                    {"field": f"state.chartAR_25.series[{class_ind}].data", "payload": [[log_dict["epoch"], log_dict[f"{class_name}_rec_0.25"]]], "append": True},
                    {"field": f"state.chartAP_5.series[{class_ind}].data", "payload": [[log_dict["epoch"], log_dict[f"{class_name}_AP_0.50"]]], "append": True},
                    {"field": f"state.chartAR_5.series[{class_ind}].data", "payload": [[log_dict["epoch"], log_dict[f"{class_name}_rec_0.50"]]], "append": True},
                ])
            fields.extend([
                {"field": f"state.chartMAP_25.series[0].data", "payload": [[log_dict["epoch"], log_dict["mAP_0.25"]]], "append": True},
                {"field": f"state.chartMAR_25.series[0].data", "payload": [[log_dict["epoch"], log_dict["mAR_0.25"]]], "append": True},
                {"field": f"state.chartMAP_5.series[0].data", "payload": [[log_dict["epoch"], log_dict["mAP_0.50"]]], "append": True},
                {"field": f"state.chartMAR_5.series[0].data", "payload": [[log_dict["epoch"], log_dict["mAR_0.50"]]], "append": True},
            ])
        try:
            g.api.app.set_fields(g.task_id, fields)
        except Exception as e:
            print("Unabled to write metrics to chart!")
            print(e)