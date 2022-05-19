import supervisely as sly

def main():
    api = sly.Api.from_env()

    # task id of the deployed model
    task_id = 13227

    # get model info
    response = api.task.send_request(task_id, "get_session_info", data={}, timeout=1)
    print("APP returns data:")
    print(response)

    # get pointcloud episodes annotation by pointcloud ids
    params = {
        "pointcloud_ids": [1120312, 1120313, 1120314],
        "threshold": 0.5,
        "classes": None, # list of str or None (all classes)
        "project_type": 'point_cloud_episodes' # ['point_cloud_episodes', 'point_clouds']
    }
    predictions = api.task.send_request(
        task_id, 
        "inference_pointcloud_ids", 
        data=params, 
        timeout=60)
    print("APP returns data:")
    print(predictions["results"])


if __name__ == "__main__":
    main()