import matplotlib.pyplot as plt
import pandas as pd

# Results retrieval

main_method_12k_min = [1466, 1510, 1098, 1076, 1042, 1006, 1048, 986, 986]

main_method_12k_mean = [2604, 2393, 2300, 2110, 2031, 2114, 2164, 1973, 1990]

main_method_15k_min = [1830, 1632, 1334, 1158, 1158, 1158, 1158, 1158, 1138]

main_method_15k_mean = [2472, 2439, 2413, 2259, 2311, 2258, 2243, 2174, 2025]


wo_ssl_method_12k_min = [
    1968,
    1880,
    1902,
    1800,
    1794,
    1956,
    1922,
]

wo_ssl_method_12k_mean = [
    2652,
    2652,
    2654,
    2656,
    2654,
    2654,
    2653,
]


wo_ssl_method_15k_min = [
    1916,
    1968,
    1782,
    1936,
    1946,
    1970,
    1880,
    1632,
    1756,
]

wo_ssl_method_15k_mean = [
    2653,
    2651,
    2656,
    2653,
    2656,
    2654,
    2655,
    2519,
    2460,
]


wo_vae_method_12k_min = [
    1782,
    1654,
    1346,
    1312,
    992,
    1562,
    1054,
    992,
]

wo_vae_method_12k_mean = [
    2496,
    2456,
    2297,
    2347,
    2105,
    2483,
    2236,
    2350,
]

wo_vae_method_15k_min = [
    1936,
    1722,
    1706,
    1262,
    1320,
    1024,
    1078,
    1008,
    990,
]

wo_vae_method_15k_mean = [
    2593,
    2425,
    2415,
    2270,
    2240,
    2190,
    2176,
    2245,
    2006,
]

min_dict = {
    "main_method_12k_min": main_method_12k_min,
    "main_method_15k_min": main_method_15k_min,
    "wo_ssl_method_12k_min": wo_ssl_method_12k_min,
    "wo_ssl_method_15k_min": wo_ssl_method_15k_min,
    "wo_vae_method_12k_min": wo_vae_method_12k_min,
    "wo_vae_method_15k_min": wo_vae_method_15k_min,
}

mean_dict = {
    "main_method_12k_mean": main_method_12k_mean,
    "main_method_15k_mean": main_method_15k_mean,
    "wo_ssl_method_12k_mean": wo_ssl_method_12k_mean,
    "wo_ssl_method_15k_mean": wo_ssl_method_15k_mean,
    "wo_vae_method_12k_mean": wo_vae_method_12k_mean,
    "wo_vae_method_15k_mean": wo_vae_method_15k_mean,
}

# Equalize list lengths
max_len = max(len(l_) for l_ in min_dict.values())
for k in min_dict:
    min_dict[k] += [None] * (max_len - len(min_dict[k]))

max_len = max(len(l_) for l_ in mean_dict.values())
for k in mean_dict:
    mean_dict[k] += [None] * (max_len - len(mean_dict[k]))

# Split 12k and 15k
min_12k_dict = {k: v for k, v in min_dict.items() if "12k" in k}
min_15k_dict = {k: v for k, v in min_dict.items() if "15k" in k}
mean_12k_dict = {k: v for k, v in mean_dict.items() if "12k" in k}
mean_15k_dict = {k: v for k, v in mean_dict.items() if "15k" in k}

# Create the dfs for plotting
min_12k_df = pd.DataFrame(min_12k_dict)
min_15k_df = pd.DataFrame(min_15k_dict)
mean_12k_df = pd.DataFrame(mean_12k_dict)
mean_15k_df = pd.DataFrame(mean_15k_dict)


# Plot
plt.figure(figsize=(10, 6))

for column in min_12k_df.columns:
    plt.plot(min_12k_df.index, min_12k_df[column], label=column)

plt.title("Min. trans. counts in prototypes - 12k initial dataset")
plt.xlabel("Prototype generation round")
plt.ylabel("Minimum trans. count in prototypes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot
plt.figure(figsize=(10, 6))

for column in min_15k_df.columns:
    plt.plot(min_15k_df.index, min_15k_df[column], label=column)

plt.title("Min. trans. counts in prototypes - 15k initial dataset")
plt.xlabel("Prototype generation round")
plt.ylabel("Minimum trans. count in prototypes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot
plt.figure(figsize=(10, 6))

for column in mean_12k_df.columns:
    plt.plot(mean_12k_df.index, mean_12k_df[column], label=column)

plt.title("Mean trans. counts in prototypes - 12k initial dataset")
plt.xlabel("Prototype generation round")
plt.ylabel("Mean trans. count in prototypes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot
plt.figure(figsize=(10, 6))

for column in mean_15k_df.columns:
    plt.plot(mean_15k_df.index, mean_15k_df[column], label=column)

plt.title("Mean trans. counts in prototypes - 15k initial dataset")
plt.xlabel("Prototype generation round")
plt.ylabel("Mean trans. count in prototypes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
