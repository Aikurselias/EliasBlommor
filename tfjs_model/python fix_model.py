"""
Run this script in the same folder as your model.json:
    python fix_model.py

It removes the RandomFlip/RandomRotation/RandomZoom/RandomContrast layers
that crash TensorFlow.js in the browser, and saves a fixed model.json.
"""

import json
import shutil

INPUT  = 'model.json'
OUTPUT = 'model.json'  # overwrites in-place (backup created first)

with open(INPUT, 'r', encoding='utf-8') as f:
    model = json.load(f)

# Backup original
shutil.copy(INPUT, 'model_backup.json')
print("Backup saved as model_backup.json")

layers = model['modelTopology']['model_config']['config']['layers']

# 1. Remove the entire 'sequential' augmentation block
before = len(layers)
layers_fixed = [l for l in layers if l['name'] != 'sequential']
print(f"Removed {before - len(layers_fixed)} layer(s) (sequential augmentation block)")

# 2. Rewire MobileNetV2 so its input comes from input_2 directly
for layer in layers_fixed:
    if layer['name'] == 'mobilenetv2_1.00_224':
        layer['inbound_nodes'] = [[["input_2", 0, 0, {"training": False}]]]
        print("Rewired mobilenetv2_1.00_224 → input_2")
        break

model['modelTopology']['model_config']['config']['layers'] = layers_fixed

with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(model, f, separators=(',', ':'))

print(f"Fixed model.json saved ({len(layers_fixed)} layers).")
print("Done! Upload the new model.json to GitHub (tfjs_model folder).")
