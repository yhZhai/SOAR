from pathlib import Path


hmdb_to_ucf_correspondence = {
    "climb": ["RockClimbingIndoor", "RopeClimbing"],
    "fencing": "Fencing",
    "golf": "GolfSwing",
    "kick_ball": "SoccerPenalty",
    "pullup": "PullUps",
    "punch": ["BoxingPunchingBag", "BoxingSpeedBag", "Punch"],
    "pushup": "PushUps",
    "ride_bike": "Biking",
    "ride_horse": "HorseRiding",
    "shoot_ball": "Basketball",
    "shoot_bow": "Archery",
    "walk": "WalkingWithDog",
}


def gen(label_map: str, src_path: str, tgt_path: str):
    label_mapper = {}
    with open(label_map, 'r') as f:
        for i, line in enumerate(f):
            line = line.replace("\n", "")
            label_mapper[i] = line
            label_mapper[line] = i
    overlapped_classes = list(hmdb_to_ucf_correspondence.keys())
    overlapped_classes_idx = [label_mapper[class_name] for class_name in overlapped_classes]
    with open(src_path, 'r') as src:
        with open(tgt_path, 'w') as tgt:
            for line in src:
                if int(line.replace("\n", "").split(" ")[-1]) in overlapped_classes_idx:
                    continue
                else:
                    tgt.write(line)


def get_ucf_to_hmdb_class_index_mapper(ucf_anno: str, hmdb_anno: str):
    def _get_mapper(anno: str):
        assert Path(anno).exists()
        result = {}
        with open(anno, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                result[line] = int(i)
        return result 

    ucf_mapper = _get_mapper(ucf_anno)
    hmdb_mapper = _get_mapper(hmdb_anno)

    mapper = {}
    for k, v in hmdb_to_ucf_correspondence.items():
        if isinstance(v, list):
            for item in v:
                mapper[ucf_mapper[item]] = hmdb_mapper[k]
        else:
            mapper[ucf_mapper[v]] = hmdb_mapper[k]

    print(mapper)
    return mapper

if __name__ == "__main__":
    # gen("tools/data/hmdb51/label_map.txt",
    #     "data/hmdb51/hmdb51_val_split_1_videos.txt",
    #     "data/hmdb51/hmdb51_val_split_1_videos_clean.txt")
    get_ucf_to_hmdb_class_index_mapper("tools/data/ucf101/label_map.txt",
                                       "tools/data/hmdb51/label_map.txt")
