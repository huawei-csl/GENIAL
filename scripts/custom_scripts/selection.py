
import pandas as pd


df = pd.read_csv('/home/ramaudruz/data_dir/misc/starting_encs/swact_data_with_encoding-selected.csv')


df['hist_value'] = pd.cut(
    df['min_val'],
    bins=10
)

df['hist_value_str'] = df['hist_value'].astype(str)

df2 = df.sample(frac=1).reset_index(drop=True)

selection = pd.concat([
    df2[df2['hist_value_str'] == '(256.356, 366.518]'].iloc[:5],
    df2[df2['hist_value_str'] == '(366.518, 475.588]'].iloc[:1],
    df2[df2['hist_value_str'] == '(475.588, 584.658]'].iloc[:1],
    df2[df2['hist_value_str'] == '(584.658, 693.729]'].iloc[:1],
    df2[df2['hist_value_str'] == '(693.729, 802.799]'].iloc[:1],
    df2[df2['hist_value_str'] == '(802.799, 911.869]'].iloc[:1],
    df2[df2['hist_value_str'] == '(1020.94, 1130.01]'].iloc[:1],
    df2[df2['hist_value_str'] == '(1239.08, 1348.151]'].iloc[:1],

])

selection.to_csv('/home/ramaudruz/data_dir/misc/starting_encs/swact_data_with_encoding-selected-SMALL.csv', index=False)



