import supervisely as sly
import sly_globals as g

log_interval = 1

def init_general(state):
    state["gpusId"] = 0
    state["randomSeed"] = 0
    state["logConfigInterval"] = 5
    state["epochs"] = 12
    state["valInterval"] = 1
    state["batchSizePerGPU"] = 4
    state["workersPerGPU"] = 4

def init_checkpoints(state):
    state["checkpointInterval"] = 1
    state["maxKeepCkptsEnabled"] = True
    state["maxKeepCkpts"] = 3
    state["saveBest"] = True

def init_optimizer(state):
    state["optimizer"] = "Adam"
    state["lr"] = 1e-3
    state["weightDecay"] = 0
    state["gradClipEnabled"] = True
    state["maxNorm"] = 30
    state["beta1"] = 0.9
    state["beta2"] = 0.999
    state["nesterov"] = False
    state["amsgrad"] = False
    state["momentum"] = 0.9
    state["momentumDecay"] = 0.004

def init_lr_scheduler(data, state):
    state["lrPolicy"] = "Step"
    state["availableLrPolicy"] = ["Fixed", "Step", "Exp", "Poly", "Inv", "CosineAnnealing", "FlatCosineAnnealing",
                                 "CosineRestart", "Cyclic", "OneCycle"]
    data["fullPolicyNames"] = ["Constant LR", "Step LR", "Exponential LR", "Polynomial LR Decay",
                               "Inverse Square Root LR", "Cosine Annealing LR", "Flat + Cosine Annealing LR",
                               "Cosine Annealing with Restarts", "Cyclic LR", "OneCycle LR"]
    # warmup
    state["useWarmup"] = False
    state["warmup"] = "constant"
    state["warmupIters"] = 0
    state["warmupRatio"] = 0.1
    state["schedulerByEpochs"] = True
    state["warmupByEpoch"] = False

    state["lr_step"] = 3
    state["minLREnabled"] = False
    state["minLR"] = None
    state["minLRRatio"] = None
    state["power"] = 1
    state["gamma"] = 0.1
    state["startPercent"] = 0.75
    state["periods"] = ""
    state["restartWeights"] = ""
    state["highestLRRatio"] = 10
    state["lowestLRRatio"] = 1e-4
    state["cyclicTimes"] = 10
    state["stepRatioUp"] = 0.4
    state["annealStrategy"] = "cos"
    state["cyclicGamma"] = 1
    state["totalStepsEnabled"] = False
    state["totalSteps"] = None
    state["maxLR"] = ""
    state["pctStart"] = 0.3
    state["divFactor"] = 25
    state["finalDivFactor"] = 1e4
    state["threePhase"] = False

def init(data, state):
    init_general(state)
    init_checkpoints(state)
    init_optimizer(state)
    init_lr_scheduler(data, state)

    state["currentTab"] = "general"
    state["collapsedWarmup"] = True
    state["collapsedParams"] = True
    state["disabledParams"] = True
    data["doneParams"] = False


def restart(data, state):
    data["doneParams"] = False


@g.my_app.callback("use_hyp")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_hyp(api: sly.Api, task_id, context, state, app_logger):
    global log_interval
    log_interval = state["logConfigInterval"]
    fields = [
        {"field": "data.doneParams", "payload": True},
        {"field": "state.collapsedMonitoring", "payload": False},
        {"field": "state.disabledMonitoring", "payload": False},
        {"field": "state.activeStep", "payload": 8},
    ]
    if state["batchSizePerGPU"] > state["finalLenTrain"]:
        fields.append({"field": "state.batchSizePerGPU", "payload": state["finalLenTrain"]})
        g.my_app.show_modal_window(
            f"Specified batch size is more than train split length. Batch size will be equal to length of train split ({state['finalLenTrain']})."
        )
    g.api.app.set_fields(g.task_id, fields)