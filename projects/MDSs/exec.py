# -*- coding: utf-8 -*-
from projects.MDSs.HG import exec as exec_hg
from projects.MDSs.MT import exec as exec_mt
from projects.MDSs.MDSs import exec as exec_mdss
from projects.MDSs.Collapsing import exec as exec_collapsing


def exec_home():
    # exec_hg("FLIC_HG_v0923.1_Home",
    #         {"epochs": 500, "trainBS": 8, "inferBS": 256,
    #          "dataSource": "FLIC", "trainCount": 1000, "validCount": 500, "labelRatio": 0.5})

    # exec_mdss("FLIC_MDSs_v0923.1_Home_PALW0.000005Const",
    #           {"PALWeight_max": 0.000005, "epochs": 500, "trainBS": 8, "inferBS": 256,
    #            "dataSource": "FLIC", "trainCount": 1000, "validCount": 500, "labelRatio": 0.5})

    # exec_mdss("Fly_MDSs_v0926.2_Home_PALW0.000005Const",
    #           {"PALWeight_max": 0.000005, "epochs": 500, "trainBS": 8, "inferBS": 256,
    #            "dataSource": "Fly", "trainCount": 100, "validCount": 500, "labelRatio": 0.3})

    exec_hg("Pranav_HG_v0926.2_Home",
            {"epochs": 500, "trainBS": 16, "inferBS": 128,
             "dataSource": "Pranav", "trainCount": 100, "validCount": 500, "labelRatio": 0.3})
    exec_collapsing("Pranav_Cola_v0926.2_Home",
              {"epochs": 500, "trainBS": 16, "inferBS": 128,
               "dataSource": "Pranav", "trainCount": 100, "validCount": 500, "labelRatio": 0.3})
    exec_collapsing("Pranav_MT_v0926.2_Home",
              {"epochs": 500, "trainBS": 16, "inferBS": 128,
               "dataSource": "Pranav", "trainCount": 100, "validCount": 500, "labelRatio": 0.3,
               "rotRange": 30.0})
    exec_mdss("Pranav_MDSs_v0926.2_Home_PALW0.000005Const",
              {"PALWeight_max": 0.000005, "epochs": 500, "trainBS": 2, "inferBS": 128,
               "dataSource": "Pranav", "trainCount": 100, "validCount": 500, "labelRatio": 0.3})


def exec_campus():
    # exec_hg("Fly_HG_v0923.1_Camp",
    #         {"epochs": 500, "trainBS": 16, "inferBS": 128,
    #          "dataSource": "Fly", "trainCount": 100, "validCount": 500, "labelRatio": 0.3})a's    # exec_collapsing("Fly_Cola_v0923.1_Camp",
    #           {"epochs": 500, "trainBS": 16, "inferBS": 128,
    #            "dataSource": "Fly", "trainCount": 100, "validCount": 500, "labelRatio": 0.3})
    # exec_collapsing("Fly_MT_v0923.1_Camp",
    #           {"epochs": 500, "trainBS": 16, "inferBS": 128,
    #            "dataSource": "Fly", "trainCount": 100, "validCount": 500, "labelRatio": 0.3,
    #            "rotRange": 30.0})
    # exec_mdss("Fly_MDSs_v0926.2_Camp_PALW0",
    #           {"PALWeight_max": 0, "epochs": 500, "trainBS": 2, "inferBS": 128,
    #            "dataSource": "Fly", "trainCount": 100, "validCount": 500, "labelRatio": 0.3})
    exec_mdss("Pranav_MDSs_v0926.2_Camp_PALW0",
              {"PALWeight_max": 0, "epochs": 500, "trainBS": 2, "inferBS": 128,
               "dataSource": "Pranav", "trainCount": 100, "validCount": 500, "labelRatio": 0.3})


def exec_laptop():
    exec_collapsing("Mouse_Cola_v0923.1_Test",
              {"epochs": 500, "trainBS": 8, "inferBS": 128,
               "dataSource": "Mouse", "trainCount": 100, "validCount": 500, "labelRatio": 0.3})


if __name__ == "__main__":
    exec_home()
