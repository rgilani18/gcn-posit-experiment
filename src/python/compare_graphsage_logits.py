
import numpy as np

pytorch = np.fromfile(
    "amazon_photo_test_logits_fp32.bin", dtype=np.float32
).reshape(1530, 8)

cpp = np.fromfile(
    "amazon_photo_test_logits_cpp_fp32.bin", dtype=np.float32
).reshape(1530, 8)

labels = np.fromfile(
    "amazon_photo_test_labels_int64.bin", dtype=np.int64
)

abs_diff = np.abs(pytorch - cpp)

pred_pt = pytorch.argmax(axis=1)
pred_cpp = cpp.argmax(axis=1)

print("PyTorch accuracy:", np.mean(pred_pt == labels))
print("C++ accuracy:    ", np.mean(pred_cpp == labels))
print("Prediction agreement:", np.mean(pred_pt == pred_cpp))
print("Maximum absolute logit difference:", abs_diff.max())
print("Mean absolute logit difference:   ", abs_diff.mean())
print("RMSE:", np.sqrt(np.mean((pytorch - cpp) ** 2)))
