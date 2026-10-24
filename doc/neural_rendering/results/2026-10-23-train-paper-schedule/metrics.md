# Training at paper schedule

Source head: `80049027025683fffa6a7f421a4d94177a5c08b5`. GPU UUID: `GPU-96322c27-d5ad-69a1-22ff-f9edbbed80ba` (RTX 4070 Ti SUPER).

Synthetic dump: 2,000,000 records, generator seed 0; fixed validation split seed 0.
Each run: 100,000 steps, batch-total 65,536, 2 hidden layers, 64x64x8 latent,
LeakySmeLU beta=1 eps=0.05, power loss n=3, Adam, LR 0.02 cosine to 0.0004.
Culling: 64@1024 -> 16@4096 -> 4@16384 -> 1@65536, 25,000 steps per phase.
Baseline: 1@65536 for all steps. Eight seeds (0-7) per cell; no tuning or ablations.
Tables below use evaluate.py on exported fp16 .nnm weights; train held-out metrics are retained locally.

Variance (8 seeds, final held-out power-loss L1, lower is better):

| method | mean | std | min | max | wall per run (s) |
|---|---|---|---|---|---|
| 2x16 baseline | 0.052845 | 0.013344 | 0.035369 | 0.076387 | 696.75 |
| 2x16 Bitterli culling | 0.034148 | 0.001208 | 0.032004 | 0.036462 | 744.02 |
| 2x32 baseline | 0.024094 | 0.003549 | 0.016440 | 0.028707 | 682.78 |
| 2x32 Bitterli culling | 0.016753 | 0.000469 | 0.016250 | 0.017558 | 745.35 |

Accuracy (mean over 8 seeds, best seed in brackets, held-out set):

| model | method | pow-loss | rel L1 (all) | rel L1 (top 1%) | rel L1 (grazing > 75) | decoder params |
|---|---|---|---|---|---|---|
| 2x16 | baseline | 0.052845 (0.035369) | 0.147011 (0.113293) | 0.207640 (0.157780) | 0.175951 | 659 |
| 2x16 | Bitterli | 0.034148 (0.032004) | 0.102780 (0.091517) | 0.145688 (0.126435) | 0.121801 | 659 |
| 2x32 | baseline | 0.024094 (0.016440) | 0.073035 (0.051050) | 0.104771 (0.068136) | 0.087002 | 1827 |
| 2x32 | Bitterli | 0.016753 (0.016250) | 0.064275 (0.058799) | 0.094869 (0.086118) | 0.076474 | 1827 |

Per-run complete process wall times (training includes validation/export; evaluation separate):

| width | method | seed | training (s) | evaluation (s) |
|---|---|---|---|---|
| 2x16 | baseline | 0 | 555.88 | 2.29 |
| 2x16 | culling | 0 | 674.45 | 2.31 |
| 2x32 | baseline | 0 | 557.13 | 3.24 |
| 2x32 | culling | 0 | 565.25 | 3.15 |
| 2x16 | baseline | 1 | 579.60 | 2.39 |
| 2x16 | culling | 1 | 586.39 | 2.13 |
| 2x32 | baseline | 1 | 579.17 | 3.50 |
| 2x32 | culling | 1 | 1077.34 | 3.22 |
| 2x16 | baseline | 2 | 1048.57 | 2.78 |
| 2x16 | culling | 2 | 991.26 | 2.16 |
| 2x32 | baseline | 2 | 946.08 | 3.53 |
| 2x32 | culling | 2 | 1313.36 | 3.36 |
| 2x16 | baseline | 3 | 1058.90 | 2.50 |
| 2x16 | culling | 3 | 1304.77 | 3.10 |
| 2x32 | baseline | 3 | 913.17 | 3.65 |
| 2x32 | culling | 3 | 686.43 | 3.40 |
| 2x16 | baseline | 4 | 698.65 | 2.12 |
| 2x16 | culling | 4 | 656.93 | 2.19 |
| 2x32 | baseline | 4 | 640.12 | 3.04 |
| 2x32 | culling | 4 | 545.22 | 2.98 |
| 2x16 | baseline | 5 | 506.75 | 2.09 |
| 2x16 | culling | 5 | 533.79 | 2.07 |
| 2x32 | baseline | 5 | 509.94 | 2.96 |
| 2x32 | culling | 5 | 533.97 | 2.96 |
| 2x16 | baseline | 6 | 498.49 | 2.04 |
| 2x16 | culling | 6 | 571.33 | 2.25 |
| 2x32 | baseline | 6 | 576.29 | 3.66 |
| 2x32 | culling | 6 | 619.27 | 3.20 |
| 2x16 | baseline | 7 | 627.20 | 2.37 |
| 2x16 | culling | 7 | 633.24 | 2.29 |
| 2x32 | baseline | 7 | 740.31 | 3.19 |
| 2x32 | culling | 7 | 621.96 | 3.15 |

Sphere PNG: best 2x32 culling seed by held-out power loss (1);
top: ground truth | model; bottom: amplified 4x error. This is synthetic BRDF fitting,
not a Cycles neural-closure furnace/MIS qualification.

Raw per-run metrics:

```json
[
  {
    "width": 16,
    "method": "baseline",
    "seed": 0,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s0",
    "train_wall_s": 555.8830441000173,
    "eval_wall_s": 2.2888090999331325,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.05270292220539604,
      "rel_l1": 0.1632994315026939,
      "rel_l1_top1": 0.23189639752254193,
      "peak_energy_top01": 0.6880562567259908,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.19620147247182496,
      "error_map_res": 32,
      "error_map_max": 0.9600526690483093,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s0\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.05469041514078721,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s0\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 16,
    "method": "culling",
    "seed": 0,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s0",
    "train_wall_s": 674.44957539998,
    "eval_wall_s": 2.308136000065133,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.036462231313091245,
      "rel_l1": 0.11302626752028805,
      "rel_l1_top1": 0.1617679796201528,
      "peak_energy_top01": 0.8254526963239496,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.13375989382704428,
      "error_map_res": 32,
      "error_map_max": 0.8812902569770813,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s0\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.04024055431096566,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s0\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 32,
    "method": "baseline",
    "seed": 0,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s0",
    "train_wall_s": 557.1312004000647,
    "eval_wall_s": 3.2404884999850765,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.028707088486904994,
      "rel_l1": 0.0873968555144572,
      "rel_l1_top1": 0.12549057515405532,
      "peak_energy_top01": 0.8597259298545891,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.10473407670979631,
      "error_map_res": 32,
      "error_map_max": 0.8458272218704224,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s0\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.03149068117944937,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s0\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 32,
    "method": "culling",
    "seed": 0,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s0",
    "train_wall_s": 565.2502830999438,
    "eval_wall_s": 3.150184000027366,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.016345216751206173,
      "rel_l1": 0.05918427613568812,
      "rel_l1_top1": 0.08611830040316358,
      "peak_energy_top01": 0.9439193883046323,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.06978526865813416,
      "error_map_res": 32,
      "error_map_max": 1.0729129314422607,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s0\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.015132081811656099,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s0\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 16,
    "method": "baseline",
    "seed": 1,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s1",
    "train_wall_s": 579.5961650999961,
    "eval_wall_s": 2.388277699938044,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.03536940449063929,
      "rel_l1": 0.12100511306013459,
      "rel_l1_top1": 0.17469411321652706,
      "peak_energy_top01": 0.8575286632790962,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.14482697564894126,
      "error_map_res": 32,
      "error_map_max": 1.006773591041565,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s1\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.03796593504278442,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s1\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 16,
    "method": "culling",
    "seed": 1,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s1",
    "train_wall_s": 586.3879876999417,
    "eval_wall_s": 2.1335960999131203,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.03375108833317302,
      "rel_l1": 0.09904887012853479,
      "rel_l1_top1": 0.1403622507555767,
      "peak_energy_top01": 0.856984758048973,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.1160855942729282,
      "error_map_res": 32,
      "error_map_max": 0.8853024840354919,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s1\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.04433864753263262,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s1\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 32,
    "method": "baseline",
    "seed": 1,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s1",
    "train_wall_s": 579.1707601998933,
    "eval_wall_s": 3.501536300056614,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.024682710155666074,
      "rel_l1": 0.07123102825476109,
      "rel_l1_top1": 0.09988399967982302,
      "peak_energy_top01": 0.8967416816276461,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.08408280958146472,
      "error_map_res": 32,
      "error_map_max": 0.8625314831733704,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s1\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.02673616533409146,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s1\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 32,
    "method": "culling",
    "seed": 1,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s1",
    "train_wall_s": 1077.3387404000387,
    "eval_wall_s": 3.22373329999391,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.016250008246525994,
      "rel_l1": 0.058799315306027625,
      "rel_l1_top1": 0.08629239520422956,
      "peak_energy_top01": 0.9073262343855064,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.06941721389010551,
      "error_map_res": 32,
      "error_map_max": 1.766255259513855,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s1\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.014928559389358238,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s1\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 16,
    "method": "baseline",
    "seed": 2,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s2",
    "train_wall_s": 1048.572993199923,
    "eval_wall_s": 2.7838221999118105,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.06685536954691956,
      "rel_l1": 0.15972334572652047,
      "rel_l1_top1": 0.21958604094716594,
      "peak_energy_top01": 0.7618270286473282,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.19053052711399013,
      "error_map_res": 32,
      "error_map_max": 0.8805187940597534,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s2\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.08945492625988338,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s2\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 16,
    "method": "culling",
    "seed": 2,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s2",
    "train_wall_s": 991.2639076000778,
    "eval_wall_s": 2.159226599964313,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.032003648510232206,
      "rel_l1": 0.09761539704298627,
      "rel_l1_top1": 0.13947021358063513,
      "peak_energy_top01": 0.8478869634164972,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.11637233557023018,
      "error_map_res": 32,
      "error_map_max": 0.8804330229759216,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s2\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.04394849069333106,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s2\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 32,
    "method": "baseline",
    "seed": 2,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s2",
    "train_wall_s": 946.0844404998934,
    "eval_wall_s": 3.5271050999872386,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.026964444835807733,
      "rel_l1": 0.06544735176700259,
      "rel_l1_top1": 0.08975069879053832,
      "peak_energy_top01": 0.9364286327448215,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.07732566950930621,
      "error_map_res": 32,
      "error_map_max": 1.3390839099884033,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s2\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.03098102328654657,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s2\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 32,
    "method": "culling",
    "seed": 2,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s2",
    "train_wall_s": 1313.356971799978,
    "eval_wall_s": 3.3610154999187216,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.017558312618294308,
      "rel_l1": 0.06635653048446741,
      "rel_l1_top1": 0.09808696107438553,
      "peak_energy_top01": 0.8982944332912507,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.07924824873792644,
      "error_map_res": 32,
      "error_map_max": 1.297776222229004,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s2\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.02027177761172574,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s2\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 16,
    "method": "baseline",
    "seed": 3,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s3",
    "train_wall_s": 1058.8972807000391,
    "eval_wall_s": 2.503804499981925,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.05681257066762868,
      "rel_l1": 0.14406283167477366,
      "rel_l1_top1": 0.19854954790818774,
      "peak_energy_top01": 0.7815092973185712,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.17121662321563377,
      "error_map_res": 32,
      "error_map_max": 0.9091894030570984,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s3\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.07235663632873313,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s3\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 16,
    "method": "culling",
    "seed": 3,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s3",
    "train_wall_s": 1304.7658024000702,
    "eval_wall_s": 3.1012491000583395,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.0349849871157941,
      "rel_l1": 0.10934078744694525,
      "rel_l1_top1": 0.15692882540588787,
      "peak_energy_top01": 0.8809132983252789,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.12945329218768795,
      "error_map_res": 32,
      "error_map_max": 1.6920214891433716,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s3\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.0401978897112543,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s3\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 32,
    "method": "baseline",
    "seed": 3,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s3",
    "train_wall_s": 913.1740875999676,
    "eval_wall_s": 3.6519733999157324,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.02229562821321599,
      "rel_l1": 0.05104992899380781,
      "rel_l1_top1": 0.06813616906424234,
      "peak_energy_top01": 0.9343256007329518,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.05866305567299316,
      "error_map_res": 32,
      "error_map_max": 0.8449866771697998,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s3\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.0239690205817833,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s3\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 32,
    "method": "culling",
    "seed": 3,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s3",
    "train_wall_s": 686.4308669999009,
    "eval_wall_s": 3.396062899962999,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.016396663267201437,
      "rel_l1": 0.0678287271613161,
      "rel_l1_top1": 0.10138851987862499,
      "peak_energy_top01": 0.8977936749721908,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.08128536136976844,
      "error_map_res": 32,
      "error_map_max": 0.8882644176483154,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s3\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.016963337957165835,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s3\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 16,
    "method": "baseline",
    "seed": 4,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s4",
    "train_wall_s": 698.6460714000277,
    "eval_wall_s": 2.1173137000296265,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.039372611095323536,
      "rel_l1": 0.11329298142341263,
      "rel_l1_top1": 0.15777980680803522,
      "peak_energy_top01": 0.8868757279052891,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.13448336767861616,
      "error_map_res": 32,
      "error_map_max": 0.8561903238296509,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s4\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.04077046182032904,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s4\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 16,
    "method": "culling",
    "seed": 4,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s4",
    "train_wall_s": 656.9250158000505,
    "eval_wall_s": 2.191017900011502,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.034665647865436494,
      "rel_l1": 0.09482971252599794,
      "rel_l1_top1": 0.1323730044919294,
      "peak_energy_top01": 0.8778369696283346,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.11183331039926818,
      "error_map_res": 32,
      "error_map_max": 0.8869756460189819,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s4\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.039290738767506925,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s4\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 32,
    "method": "baseline",
    "seed": 4,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s4",
    "train_wall_s": 640.1207119999453,
    "eval_wall_s": 3.0393716000253335,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.025610297051033456,
      "rel_l1": 0.08234038514358476,
      "rel_l1_top1": 0.12079584430440249,
      "peak_energy_top01": 0.886892901993404,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.09936775170815887,
      "error_map_res": 32,
      "error_map_max": 0.8468897938728333,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s4\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.026629928019809682,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s4\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 32,
    "method": "culling",
    "seed": 4,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s4",
    "train_wall_s": 545.2168269000249,
    "eval_wall_s": 2.978933300008066,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.017387052986943655,
      "rel_l1": 0.06788111608432315,
      "rel_l1_top1": 0.10054248227193531,
      "peak_energy_top01": 0.8714208015211958,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.08146507129493807,
      "error_map_res": 32,
      "error_map_max": 2.369746685028076,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s4\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.019906920542307985,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s4\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 16,
    "method": "baseline",
    "seed": 5,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s5",
    "train_wall_s": 506.7519043999491,
    "eval_wall_s": 2.0932078999467194,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.05527126400177026,
      "rel_l1": 0.15044925843902004,
      "rel_l1_top1": 0.2113823195521638,
      "peak_energy_top01": 0.7609709332063671,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.1801103283828367,
      "error_map_res": 32,
      "error_map_max": 0.9391997456550598,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s5\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.06226094873068367,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s5\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 16,
    "method": "culling",
    "seed": 5,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s5",
    "train_wall_s": 533.7896854000865,
    "eval_wall_s": 2.0664421999827027,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.03368472069161214,
      "rel_l1": 0.11826620054020581,
      "rel_l1_top1": 0.1693325211606484,
      "peak_energy_top01": 0.8405154539747697,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.14148384652774929,
      "error_map_res": 32,
      "error_map_max": 0.9112062454223633,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s5\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.03845341804354011,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s5\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 32,
    "method": "baseline",
    "seed": 5,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s5",
    "train_wall_s": 509.93510559992865,
    "eval_wall_s": 2.960720199975185,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.02218908045071121,
      "rel_l1": 0.07685972079303245,
      "rel_l1_top1": 0.11387815693674791,
      "peak_energy_top01": 0.890210583528206,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.09241222158182366,
      "error_map_res": 32,
      "error_map_max": 0.9323890209197998,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s5\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.025337127444693422,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s5\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 32,
    "method": "culling",
    "seed": 5,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s5",
    "train_wall_s": 533.967031700071,
    "eval_wall_s": 2.9643781000049785,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.016454288195540803,
      "rel_l1": 0.06009769718325309,
      "rel_l1_top1": 0.08796638308379946,
      "peak_energy_top01": 0.9277646304840204,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.07087934907781653,
      "error_map_res": 32,
      "error_map_max": 0.9996340870857239,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s5\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.018274057934447245,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s5\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 16,
    "method": "baseline",
    "seed": 6,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s6",
    "train_wall_s": 498.49332939996384,
    "eval_wall_s": 2.0429285999853164,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.07638716973797616,
      "rel_l1": 0.198929442368559,
      "rel_l1_top1": 0.2846709340383238,
      "peak_energy_top01": 0.6106052419516997,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.23972845393459438,
      "error_map_res": 32,
      "error_map_max": 1.0466485023498535,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s6\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.11273592907023294,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s6\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 16,
    "method": "culling",
    "seed": 6,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s6",
    "train_wall_s": 571.3322371999966,
    "eval_wall_s": 2.2465976999374107,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.03353934780476073,
      "rel_l1": 0.09151689326962534,
      "rel_l1_top1": 0.12643523988848787,
      "peak_energy_top01": 0.920575132499338,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.10728078631311398,
      "error_map_res": 32,
      "error_map_max": 0.9039175510406494,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s6\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.039777472508653426,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s6\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 32,
    "method": "baseline",
    "seed": 6,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s6",
    "train_wall_s": 576.289350099978,
    "eval_wall_s": 3.66229879995808,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.02586494166678901,
      "rel_l1": 0.07997930096407004,
      "rel_l1_top1": 0.11541003952991677,
      "peak_energy_top01": 0.8884048801579456,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.09530449635516183,
      "error_map_res": 32,
      "error_map_max": 0.8403023481369019,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s6\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.02803269215771267,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s6\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 32,
    "method": "culling",
    "seed": 6,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s6",
    "train_wall_s": 619.2745857000118,
    "eval_wall_s": 3.198631399893202,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.016628217307559937,
      "rel_l1": 0.06471385435077293,
      "rel_l1_top1": 0.09567782170452742,
      "peak_energy_top01": 0.8817167453845215,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.0769620967795682,
      "error_map_res": 32,
      "error_map_max": 1.1834125518798828,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s6\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.016998269620117745,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s6\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 16,
    "method": "baseline",
    "seed": 7,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s7",
    "train_wall_s": 627.1969092999352,
    "eval_wall_s": 2.374708399991505,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.039986542475160876,
      "rel_l1": 0.12532769847406064,
      "rel_l1_top1": 0.18256335470206664,
      "peak_energy_top01": 0.8036472723987678,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.15051396225623692,
      "error_map_res": 32,
      "error_map_max": 0.945777952671051,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s7\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.041026941108099506,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_baseline_s7\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 16,
    "method": "culling",
    "seed": 7,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s7",
    "train_wall_s": 633.2352593000978,
    "eval_wall_s": 2.2928382999962196,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.03409600405687125,
      "rel_l1": 0.09859575505274355,
      "rel_l1_top1": 0.13883479646268057,
      "peak_energy_top01": 0.9840510494779373,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.1181406374175354,
      "error_map_res": 32,
      "error_map_max": 1.770355224609375,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s7\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.045312679101661826,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w16_culling_s7\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 32,
    "method": "baseline",
    "seed": 7,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s7",
    "train_wall_s": 740.308384199976,
    "eval_wall_s": 3.193326500011608,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.01643977597735391,
      "rel_l1": 0.06997391141357144,
      "rel_l1_top1": 0.10481984961902713,
      "peak_energy_top01": 0.9136661557612087,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.08412260883161793,
      "error_map_res": 32,
      "error_map_max": 3.124183177947998,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s7\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.015322193204481608,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_baseline_s7\\evaluation\\sphere_compare.png"
    }
  },
  {
    "width": 32,
    "method": "culling",
    "seed": 7,
    "run": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s7",
    "train_wall_s": 621.9618297999259,
    "eval_wall_s": 3.1502037999453023,
    "evaluation": {
      "n": 98983,
      "pow_loss": 0.017001261991735628,
      "rel_l1": 0.06934064321387581,
      "rel_l1_top1": 0.10288159028541503,
      "peak_energy_top01": 0.8660808166881251,
      "grazing_frac": 0.36433528989826536,
      "rel_l1_grazing": 0.08274680211929976,
      "error_map_res": 32,
      "error_map_max": 0.8439993858337402,
      "nnm": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s7\\model.nnm",
      "split": "val",
      "sphere_rel_l1": 0.016720061602912686,
      "sphere_png": "D:\\Github\\Blender-RND\\train-paper-schedule\\w32_culling_s7\\evaluation\\sphere_compare.png"
    }
  }
]
```
