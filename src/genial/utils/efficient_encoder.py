from time import time
from typing import List
import pandas as pd


class EfficientEncodingHelper:
    def string_dic_to_bool_cols(self, encoding_series: pd.Series, prefix: str) -> pd.DataFrame:
        """
        This method splits a pandas series of encodings into a dataframe only containing boolean columns.
        """
        # Split the column storing dictionaries as strings
        splitted_series = encoding_series.map(self.string_split)

        # Get the int to index
        index_to_int_map = self.get_index_to_int_map(splitted_series.iloc[0])

        # Create a boolean column for each integers
        data_dic = {}
        bool_count = None
        for i, int_ in index_to_int_map.items():
            temp_series = splitted_series.map(lambda x: x[i])
            if bool_count is None:
                bool_count = len(temp_series.iloc[0])
            for j in range(bool_count):
                data_dic[f"{prefix}_enc_{int_}_index_{j}"] = temp_series.map(lambda x: x[j]).astype(int).astype(bool)
        return pd.DataFrame(data_dic)

    @staticmethod
    def bool_cols_to_string_dic(encoded_df: pd.DataFrame, prefix: str) -> pd.Series:
        int_to_loop = list(set([int(c.split("_")[2]) for c in encoded_df.columns if c.startswith(prefix)]))
        int_to_loop.sort()
        bool_count = len(set([int(c.split("_")[4]) for c in encoded_df.columns if c.startswith(prefix)]))
        data_dic = {}
        for i in int_to_loop:
            cols = [f"{prefix}_enc_{i}_index_{j}" for j in range(bool_count)]
            data_dic[i] = encoded_df[cols].astype(int).astype(str).sum(1)
        dict_list = pd.DataFrame(data_dic).to_dict(orient="records")
        return pd.Series(dict_list).astype(str)

    @staticmethod
    def string_split(string: str) -> List[str]:
        """
        Convenience function for splitting a string on the single quote character.
        """
        return string.split("'")

    @staticmethod
    def get_index_to_int_map(string):
        index_to_int_map = {}
        for i, string in enumerate(string):
            if i % 2 == 0:
                int_str = string.split("{")[-1].split(",")[-1].split(":")[0].strip()
                if int_str == "}":
                    break
                value = int(int_str)
            else:
                index_to_int_map[i] = value
        return index_to_int_map

    def get_encoded_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method takes in a df that has both an "in_enc_dict" and "out_enc_dict" columns and creates a lightweight df
        with the encodings. It is the reverse of the get_decoded_df method.
        """
        in_enc_df = self.string_dic_to_bool_cols(df["in_enc_dict"], prefix="in")
        out_enc_df = self.string_dic_to_bool_cols(df["out_enc_dict"], prefix="out")
        return pd.concat([in_enc_df, out_enc_df], axis=1)

    def get_decoded_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method takes in a lightweight df containing encodings and creates a df that has both an "in_enc_dict" and
        "out_enc_dict" columns. It is the reverse of the get_encoded_df method.
        """
        in_enc_dict_series = self.bool_cols_to_string_dic(df, prefix="in")
        out_enc_dict_series = self.bool_cols_to_string_dic(df, prefix="out")
        return pd.DataFrame({"in_enc_dict": in_enc_dict_series, "out_enc_dict": out_enc_dict_series})

    def get_in_encoded_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method only encodes the in encodings columns, creating a lightweight df with the encodings.
        """
        return self.string_dic_to_bool_cols(df["in_enc_dict"], prefix="in")

    def get_in_decoded_df(self, df: pd.DataFrame) -> pd.Series:
        """
        This method decodes the in encodings columns, recovering the encoding dictionary string.
        """
        return self.bool_cols_to_string_dic(df, prefix="in")


if __name__ == "__main__":
    test_dir = None

    # Instantiate encoding helper
    eeh = EfficientEncodingHelper()

    # Loading heavy
    start = time()
    encoding_dicts_df = pd.read_parquet(test_dir + "full_dataset_heavy.parquet")
    print(f"Loading heavy took {time() - start}\n")

    ################ PART ONE - ONLY IN ENCODINGS ############################
    print("\nStarting in only...")

    # Here we first check how much an eval would take. This is the time we could potentially save
    start = time()
    _ = encoding_dicts_df["in_enc_dict"].map(eval)
    print(f"eval of in encodings takes {time() - start}")

    # Then, we check how long it would take to encode and decode

    # Encoding heavy to lightweight
    start = time()
    in_encoded_df = eeh.get_in_encoded_df(encoding_dicts_df)
    print(f"Encoding heavy to light (in) took {time() - start}")

    # Decoding lightweight to heavy
    start = time()
    in_decoded_df = eeh.get_in_decoded_df(in_encoded_df)
    print(f"Decoding light to heavy (in) took {time() - start}")

    # Equality test
    if pd.DataFrame.equals(encoding_dicts_df["in_enc_dict"], in_decoded_df):
        print("Equality test (in) succeeded!")
    else:
        print("Equality test (in) failed!")

    # ################ PART TWO - BOTH IN & OUT ENCODINGS ############################
    # print('\nStarting in and out...')
    #
    # # Reset
    # del (encoding_dicts_df, in_encoded_df, in_decoded_df)
    # encoding_dicts_df = pd.read_parquet(test_dir + 'full_dataset_heavy.parquet')
    #
    # # Here we first check how much an eval would take. This is the time we could potentially save
    # start = time()
    # _ = encoding_dicts_df['in_enc_dict'].map(eval)
    # _ = encoding_dicts_df['out_enc_dict'].map(eval)
    # print(f'eval of in and out encodings takes {time() - start}')
    #
    # # Encoding heavy to lightweight
    # start = time()
    # encoded_df = eeh.get_encoded_df(encoding_dicts_df)
    # print(f'Encoding heavy to light (in and out) took {time() - start}')
    #
    # # Decoding lightweight to heavy
    # start = time()
    # decoded_df = eeh.get_decoded_df(encoded_df)
    # print(f'Decoding light to heavy (in and out) took {time() - start}')
    #
    # # Storing heavy
    # start = time()
    # decoded_df.to_parquet(test_dir + 'full_dataset_heavy_test.parquet', index=False)
    # print(f'Storing heavy (in and out) took {time() - start}')
    # # os.remove(test_dir + 'full_dataset_heavy_test.parquet')
    #
    # # Storing light
    # start = time()
    # encoded_df.to_parquet(test_dir + 'full_dataset_light_test.parquet', index=False)
    # print(f'Storing light (in and out) took {time() - start}')
    # # os.remove(test_dir + 'full_dataset_light_test.parquet')
    #
    # # Equality test
    # if pd.DataFrame.equals(encoding_dicts_df, decoded_df):
    #     print('Equality test (in and out) succeeded!')
    # else:
    #     print('Equality test (in and out) failed!')
