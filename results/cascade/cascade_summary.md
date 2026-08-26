# Cascade consistency across scales

|   scale | source                                     |   gt_acc |   pred_acc |     gt_auc |   pred_auc |   agreement |   n_disagree |   mgct_recall_drop |   acc_drop |    auc_drop |
|--------:|:-------------------------------------------|---------:|-----------:|-----------:|-----------:|------------:|-------------:|-------------------:|-----------:|------------:|
|      40 | feature_dca SVM (paper record)                  | 0.9      |   0.825    | nan        | nan        |    0.875    |            7 |         nan        |  0.075     | nan         |
|      40 | calibration figure pipeline distance-CNN                        | 0.875    |   0.8      |   0.974502 |   0.870257 |    0.875    |            5 |           0.444444 |  0.075     |   0.104246  |
|     113 | cascade_eval fold0-2 fair SVM                    | 0.884956 |   0.831858 |   0.971709 |   0.94959  |    0.893805 |           12 |           0.35     |  0.0530973 |   0.0221191 |
|     193 | full (authoritative cascade pipeline_results_interim_cascade.json) | 0.884956 |   0.831858 |   0.971709 |   0.94959  |    0.893805 |           12 |         nan        |  0.0530973 |   0.0221191 |
