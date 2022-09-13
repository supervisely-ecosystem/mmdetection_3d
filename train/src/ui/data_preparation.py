import os, os.path as osp
import pickle as pkl
import numpy as np
import open3d as o3d
import supervisely as sly
import sly_globals as g
import splits


train_set_path = osp.join(g.my_app.data_dir, "train.pkl")
val_set_path = osp.join(g.my_app.data_dir, "val.pkl")

def init(data, state):

    state["preparingData"] = False
    data["doneData"] = False
    state["collapsedData"] = True
    state["disabledData"] = True
    state["center_coords"] = [True, True, True] # [x, y, z]
    state["window_size"] = [100.0, 100.0, 20.0]
    state["train_data_mode"] = "full"

    init_progress("ConvertTrain", state)
    init_progress("ConvertVal", state)


def restart(data, state):
    data["doneData"] = False


def init_progress(index, state):
    state[f"progress{index}"] = False
    state[f"progressCurrent{index}"] = 0
    state[f"progressTotal{index}"] = None
    state[f"progressPercent{index}"] = 0

def centerize_ptc(points, centerize):
    centerize_vec = [0, 0, 0]
    for i in range(3):
        if centerize[i]:
            dim_trans = points[:,i].min() + (points[:,i].max() - points[:,i].min()) * 0.5
            points[:,i] -= dim_trans
            centerize_vec[i] = -dim_trans

    return points, centerize_vec


def save_set_to_annotation(state, save_path, items, split_name, slide_boxes):
    fields = [
        {"field": f"state.progressConvert{split_name}", "payload": True},
        {"field": f"state.progressTotalConvert{split_name}", "payload": len(items)},
    ]
    g.api.app.set_fields(g.task_id, fields)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    annotations = []
    log_step = 5
    frames_to_pcds = {}
    pcds_to_frames = {}
    if len(items) < log_step:
        log_step = 1

    for idx, item in enumerate(items):
        if idx % log_step == 0:
            fields = [
                {"field": f"state.progressCurrentConvert{split_name}", "payload": idx},
                {"field": f"state.progressPercentConvert{split_name}", "payload": int(idx / len(items) * 100)}
            ]
            g.api.app.set_fields(g.task_id, fields)
        
        os.makedirs(osp.join(g.project_dir, item.dataset_name, "bin"), exist_ok=True)
        filename = osp.join(item.dataset_name, "pointcloud", item.name)
        pcd = o3d.io.read_point_cloud(osp.join(g.project_dir, filename))
        pcd_np = np.asarray(pcd.points)

        pcd_sboxes = []
        pcdim = state["point_cloud_dim"]
        for sbox in slide_boxes:
            pcd_sboxes.append([
                pcd_np[:,0].min() + (pcd_np[:,0].max() - pcd_np[:,0].min()) * 0.5 - pcdim[0] * 0.5 + sbox[0],
                pcd_np[:,0].min() + (pcd_np[:,0].max() - pcd_np[:,0].min()) * 0.5 - pcdim[0] * 0.5 + sbox[1],
                pcd_np[:,1].min() + (pcd_np[:,1].max() - pcd_np[:,1].min()) * 0.5 - pcdim[1] * 0.5 + sbox[2],
                pcd_np[:,1].min() + (pcd_np[:,1].max() - pcd_np[:,1].min()) * 0.5 - pcdim[1] * 0.5 + sbox[3],
                pcd_np[:,2].min() + (pcd_np[:,2].max() - pcd_np[:,2].min()) * 0.5 - pcdim[2] * 0.5 + sbox[4],
                pcd_np[:,2].min() + (pcd_np[:,2].max() - pcd_np[:,2].min()) * 0.5 - pcdim[2] * 0.5 + sbox[5]
            ])

        if item.dataset_name not in frames_to_pcds.keys():
            frames_to_pcds[item.dataset_name] = sly.json.load_json_file(osp.join(g.project_dir, item.dataset_name, "frame_pointcloud_map.json"))
            pcds_to_frames[item.dataset_name] = {v: k for k, v in frames_to_pcds[item.dataset_name].items()}

        pcd_to_frame = pcds_to_frames[item.dataset_name]
        frame_number = pcd_to_frame[item.name]
        ann_path = osp.join(g.project_dir, item.dataset_name, "annotation.json")
        ann_json = sly.json.load_json_file(ann_path)
        # key_id_map = KeyIdMap().load_json(osp.join(g.project_dir, "key_id_map.json"))
        ann = sly.PointcloudEpisodeAnnotation.from_json(ann_json, g.project_meta)
        for frame in ann.frames:
            if frame.index != int(frame_number):
                continue
            
            for slide_box_idx, sbox in enumerate(pcd_sboxes):
                ptc_info = {
                    'sample_idx': idx,
                    'lidar_points': {},
                    'annos': {
                        'gt_bboxes_3d': [],
                        'gt_names': [],
                        'gt_labels_3d': [],
                        'box_type_3d': 'LiDAR'
                    }
                }
                pcd_eps = 1e-3
                pcd_slide = pcd_np[
                    (pcd_np[:,0] > sbox[0] - pcd_eps) &
                    (pcd_np[:,0] < sbox[1] + pcd_eps) &
                    (pcd_np[:,1] > sbox[2] - pcd_eps) &
                    (pcd_np[:,1] < sbox[3] + pcd_eps) &
                    (pcd_np[:,2] > sbox[4] - pcd_eps) &
                    (pcd_np[:,2] < sbox[5] + pcd_eps)
                ]
                if len(pcd_slide) == 0:
                    continue
                slide_name = f"{item.name}_{slide_box_idx}"
                bin_filename = osp.join(item.dataset_name, "bin", f"{slide_name}.bin")
                trans_vec = [0, 0, 0]
                if any(state["center_coords"]):
                    pcd_slide, trans_vec = centerize_ptc(pcd_slide, state["center_coords"])
                 
                intensity = np.zeros((pcd_slide.shape[0], 1), dtype=np.float32)
                pcd_slide = np.hstack((pcd_slide, intensity))
                pcd_slide.astype(np.float32).tofile(osp.join(g.project_dir, bin_filename))
                ptc_info['lidar_points']['lidar_path'] = bin_filename
                for fig in frame.figures:
                    if fig.video_object.obj_class.name not in state["selectedClasses"]:
                        continue
                    box_info = [] # x, y, z, dx, dy, dz, rot, [vel_x, vel_y]
                    pos = fig.geometry.position

                    if pos.x < sbox[0] or pos.x >= sbox[1] or \
                        pos.y < sbox[2] or pos.y >= sbox[3] or \
                        pos.z < sbox[4] or pos.z >= sbox[5]:
                        continue
                    pos_x = pos.x + trans_vec[0]
                    pos_y = pos.y + trans_vec[1]
                    pos_z = pos.z + trans_vec[2]

                    box_pos = [pos_x, pos_y, pos_z]
                    box_info.extend(box_pos)
                    dim = fig.geometry.dimensions
                    box_info.extend([dim.x, dim.y, dim.z])
                    box_info.extend([fig.geometry.rotation.z])
                    box_info.extend([0, 0]) # TODO: add vel

                    ptc_info['annos']['gt_names'].append(fig.video_object.obj_class.name)
                    ptc_info['annos']['gt_bboxes_3d'].append(box_info)
                    ptc_info['annos']['gt_labels_3d'].append(state["selectedClasses"].index(fig.video_object.obj_class.name))
                ptc_info['annos']['gt_bboxes_3d'] = np.array(ptc_info['annos']['gt_bboxes_3d'], dtype=np.float32)
                ptc_info['annos']['gt_labels_3d'] = np.array(ptc_info['annos']['gt_labels_3d'], dtype=np.int32)
                annotations.append(ptc_info)

    g.api.app.set_field(g.task_id, f"state.progressConvert{split_name}", False)
        
    with open(save_path, 'wb') as f:
        pkl.dump(annotations, f)


def get_slide_boxes(state):
    pcd = state["point_cloud_dim"].copy()
    ws = state["window_size"].copy()
    for i in range(3):
        if pcd[i] < ws[i]:
            ws[i] = pcd[i]

    slides_x, overlap_x = divmod(pcd[0], ws[0])
    slides_y, overlap_y = divmod(pcd[1], ws[1])
    slides_z, overlap_z = divmod(pcd[2], ws[2])
    if overlap_x != 0:
        slides_x += 1
    if overlap_y != 0:
        slides_y += 1
    if overlap_z != 0:
        slides_z += 1

    sboxes = []
    for z in range(int(slides_z)):
        for y in range(int(slides_y)):
            for x in range(int(slides_x)):
                sboxes.append([
                    0 if x == 0 else ws[0] * x - overlap_x,
                    ws[0] if x == 0 else ws[0] * (x + 1) - overlap_x,
                    0 if y == 0 else ws[1] * y - overlap_y,
                    ws[1] if y == 0 else ws[1] * (y + 1) - overlap_y,
                    0 if z == 0 else ws[2] * z - overlap_z,
                    ws[2] if z == 0 else ws[2] * (z + 1) - overlap_z,
                ])
    return sboxes, ws

@g.my_app.callback("prepare_data")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def prepare_data(api: sly.Api, task_id, context, state, app_logger):
    
    if state["train_data_mode"] == 'sliding_window':
        sboxes, state["window_size"] = get_slide_boxes(state)
        pcr = [
            -state["window_size"][0] * 0.5, 
            -state["window_size"][1] * 0.5, 
            -state["window_size"][2] * 0.5, 
            state["window_size"][0] * 0.5,
            state["window_size"][1] * 0.5,
            state["window_size"][2] * 0.5
        ]
    elif state["train_data_mode"] == 'full':
        sboxes = [[
            0, state["point_cloud_dim"][0],
            0, state["point_cloud_dim"][1],
            0, state["point_cloud_dim"][2]
        ]]
        if any(state["center_coords"]):
            pcr = [
                -state["point_cloud_dim"][0] * 0.5 if state["center_coords"][0] else state["point_cloud_range"][0], 
                -state["point_cloud_dim"][1] * 0.5 if state["center_coords"][1] else state["point_cloud_range"][1], 
                -state["point_cloud_dim"][2] * 0.5 if state["center_coords"][2] else state["point_cloud_range"][2], 
                state["point_cloud_dim"][0] * 0.5 if state["center_coords"][0] else state["point_cloud_range"][3],
                state["point_cloud_dim"][1] * 0.5 if state["center_coords"][1] else state["point_cloud_range"][4],
                state["point_cloud_dim"][2] * 0.5 if state["center_coords"][2] else state["point_cloud_range"][5]
            ]
        else:
            pcr = state["point_cloud_range"]
    g.api.app.set_field(g.task_id, f"state.point_cloud_range", pcr)
    if splits.train_set is not None:
        sly.logger.info("Converting train annotations to mmdet3d format...")
        # if not osp.exists(train_set_path): # TODO: for debug
        save_set_to_annotation(state, train_set_path, splits.train_set, "Train", sboxes)
    if splits.val_set is not None:
        sly.logger.info("Converting val annotations to mmdet3d format...")
        #if not osp.exists(val_set_path): # TODO: for debug
        # TODO: eval on the same boxes for debug
        # save_set_to_annotation(state, val_set_path, val_set, "Val")
        save_set_to_annotation(state, val_set_path, splits.val_set, "Val", sboxes)
    fields = [
        {"field": "state.preparingData", "payload": False},
        {"field": "data.doneData", "payload": True},
        {"field": "state.collapsedAugs", "payload": False},
        {"field": "state.disabledAugs", "payload": False},
        {"field": "state.activeStep", "payload": 6},
    ]

    g.api.app.set_fields(g.task_id, fields)

