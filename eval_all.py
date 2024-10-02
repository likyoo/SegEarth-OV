import os
configs_list = [
    # rs semantic segmentation
    './configs/cfg_openearthmap.py',
    './configs/cfg_loveda.py',
    './configs/cfg_iSAID.py',
    './configs/cfg_potsdam.py',
    './configs/cfg_vaihingen.py',
    './configs/cfg_uavid.py',
    './configs/cfg_udd5.py',
    './configs/cfg_vdd.py',
    # rs single-class
    './configs/cfg_whu_aerial.py',
    './configs/cfg_whu_sat_II.py',
    './configs/cfg_inria.py',
    './configs/cfg_xBD.py',
    './configs/cfg_chn6-cug.py',
    './configs/cfg_deepglobe_road.py',
    './configs/cfg_massachusetts_road.py',
    './configs/cfg_spacenet_road.py',
    './configs/cfg_wbs-si.py',
]

for config in configs_list:
    print(f"Running {config}")
    # os.system(f"bash ./dist_test.sh {config}")
    os.system(f'python eval.py --config {config} --work-dir work_dirs/tmp')
