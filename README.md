# ese5190-2022-final-project-keyboard-part

## 1.1 Goal and Results
The original magic_wand_ble model can only classify between 0-9. We built our own network to classify between both 0-9 and A-Z, which means our model have 36 classes now.

Colab version of the training part: https://colab.research.google.com/drive/1JqMVoSU88o-7ZZmzwoQUtnsRRwZ0dx7c?usp=sharing

The accuracy of our model after 50 epochs training:
<img src="./keyboard_model_acc_epochs_50.png." alt="Accuracy" width="600"/>

## Notes
Add this at the beginning of model.cc:

```
/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Automatically created from a TensorFlow Lite flatbuffer using the command:
// xxd -i magic_wand_model.tflite > magic_wand_model_data.cc
// See the README for a full description of the creation process.

#include "magic_wand_model_data.h"

const unsigned char g_magic_wand_model_data[]
```
