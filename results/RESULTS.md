# 'Bring me that one': results

Answers: 864 (3 models x 3 conditions x 96 scenes). Invalid: 14. Chance: 0.33 with 3 objects, 0.17 with 6.

## Accuracy by model and condition (95% bootstrap CI)

| model | image | image_ray | text |
|---|---|---|---|
| gemma4:12b | 0.44 [0.33, 0.53] | 0.82 [0.74, 0.90] | 0.47 [0.38, 0.57] |
| qwen3.5:9b | 0.21 [0.14, 0.29] | 0.67 [0.57, 0.76] | 0.33 [0.24, 0.44] |
| gemma3:12b | 0.33 [0.24, 0.43] | 0.70 [0.60, 0.78] | 0.31 [0.22, 0.42] |

## Accuracy by angle to the closest rival (deg)

```
sep                     8     15    30
model      cond                       
gemma3:12b image      0.25  0.44  0.31
           image_ray  0.53  0.72  0.84
           text       0.31  0.38  0.25
gemma4:12b image      0.38  0.34  0.59
           image_ray  0.69  0.84  0.94
           text       0.38  0.53  0.50
qwen3.5:9b image      0.19  0.25  0.19
           image_ray  0.41  0.78  0.81
           text       0.28  0.41  0.31
```

## Accuracy by target distance

```
dist                   far  near
model      cond                 
gemma3:12b image      0.29  0.38
           image_ray  0.79  0.60
           text       0.29  0.33
gemma4:12b image      0.29  0.58
           image_ray  0.92  0.73
           text       0.46  0.48
qwen3.5:9b image      0.10  0.31
           image_ray  0.81  0.52
           text       0.27  0.40
```

## Accuracy by number of objects

```
n                        3     6
model      cond                 
gemma3:12b image      0.40  0.27
           image_ray  0.81  0.58
           text       0.29  0.33
gemma4:12b image      0.48  0.40
           image_ray  0.83  0.81
           text       0.67  0.27
qwen3.5:9b image      0.21  0.21
           image_ray  0.65  0.69
           text       0.35  0.31
```

## Direction or proximity? Accuracy split by whether the target is also the object nearest the hand
(shortcut 'pick the object nearest the hand' alone scores 0.35)

```
                      target NOT nearest the hand  target IS nearest the hand
model      cond                                                              
gemma3:12b image                             0.26                        0.47
           image_ray                         0.74                        0.62
           text                              0.35                        0.24
gemma4:12b image                             0.23                        0.82
           image_ray                         0.90                        0.68
           text                              0.52                        0.38
qwen3.5:9b image                             0.18                        0.26
           image_ray                         0.73                        0.56
           text                              0.21                        0.56
```

## When wrong, what did the model pick? (share of wrong answers)
`rival` = the object at the next-closest angle; `nearest_hand` / `nearest_body` = the object closest to the hand / body (shortcuts that ignore direction).

```
                      wrong  rival  nearest_hand  nearest_body
model      cond                                               
gemma3:12b image         64   0.55          0.50          0.47
           image_ray     29   0.72          0.14          0.07
           text          66   0.52          0.15          0.18
gemma4:12b image         54   0.57          0.65          0.63
           image_ray     17   0.76          0.12          0.06
           text          51   0.59          0.12          0.10
qwen3.5:9b image         62   0.55          0.29          0.26
           image_ray     32   0.72          0.22          0.22
           text          64   0.55          0.59          0.56
```

## Does drawing the ray help? (paired permutation test, Holm-corrected)

- gemma4:12b: change +0.39, p = 0.0001, Holm p = 0.0003
- qwen3.5:9b: change +0.46, p = 0.0001, Holm p = 0.0003
- gemma3:12b: change +0.36, p = 0.0001, Holm p = 0.0003
