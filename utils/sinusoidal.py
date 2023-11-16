import numpy as np


frequencies = np.exp(
    np.linspace(
        np.log(1),
        np.log(1000),
        32 // 2,
    )
)

embedding_list = []
for y in np.arange(0, 1, 0.01):
    x = np.array([[[[y]]]])
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = np.concatenate(
        [np.sin(angular_speeds * x), np.cos(angular_speeds * x)], axis=3
    )
    embedding_list.append(embeddings[0][0][0])
embedding_array = np.transpose(embedding_list)
