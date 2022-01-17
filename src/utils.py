import re
import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def drop_columns(df, columns):
    return df.drop(columns, axis=1)


def preprocess_categorical_columns(df, columns):
    df[columns] = df.copy()[columns].apply(lambda x: x.str.strip().str.replace(" ", "_").str.replace("-", ""))
    return df


def preprocess_emp_length(df):
    """
    Anything value that cannot cast to a floating point value is
    set to -1.
    Data is not assumed to be missing at random.

    This function replaces 'na' and any string value in 'emp_length' with -1.0.
    It drops all columns specified by columns argument.

    :param df :pandas.DataFrame
    :return: pandas.DataFrame
    """
    df.loc[df.emp_length.apply(lambda x: re.findall(r"\D", x)).apply(len) > 0, "emp_length"] = -1.0
    df["emp_length"] = df["emp_length"].replace("na", -1.0).astype(float)
    return df


def dataset_specific_cleanup(df, columns):
    df["emp_length_filled_na"] = df.copy()["emp_length"].replace("na", -1.0).astype(int)
    columns_to_drop = columns + ["emp_length"]
    df = df.drop(columns=columns_to_drop)
    return df.rename({"emp_length_filled_na": "emp_length"}, axis="columns")


def fillnans(df, columns):
    """Fill NaNs in columns with -1.0.
    Making the assumption that input validators is not missing at random.

    :param df :pandas.DataFrame
    :param columns :str
    :return: pandas.DataFrame
    """
    inf = numpy.Inf
    df[columns + "_filled_nans"] = df.copy()[columns].fillna(-1.0).replace(inf, -1.0).replace(-inf, -1.0)
    df = df.drop(columns=columns)
    return df.rename({columns + "_filled_nans": columns}, axis="columns")


def preprocess_purpose_cat(df):
    """
    A Preprocessing function that maps '{category name} small business' -> {category name}'
    for the purpose_cat column in the dataset.

    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    purpose_cat = df["purpose_cat"].copy()
    endswith_smb_dict = {
        f"{cat}": " ".join(cat.split(" ")[:-2]) for cat in purpose_cat.unique() if " small business" in cat
    }
    remaining_name_dict = {cat: cat for cat in purpose_cat.unique() if " small business" not in cat}
    purpose_cat_mapping_dict = {**endswith_smb_dict, **remaining_name_dict}
    df["purpose_cat"] = df["purpose_cat"].map(purpose_cat_mapping_dict)
    return df


def make_categorical_columns(df, categories, columns):
    """
    Makes categorical columns with names corresponding to one-hot encodings.

    :param df: pandas.DataFrame
    :param categories: List
    :param columns: List
    :return:
    """
    oh_columns = []
    for col, cat in zip(columns, categories):
        for cat_value in cat:
            oh_columns.append(f"{col}_{cat_value}")

    df.columns = oh_columns
    return df


def train_minmax(df, columns):
    """
    Function that does MinMax scaling all numerical columns specified by columns argument.

    :param df: pandas.DataFrame
    :param columns: list
    :return: Tuple(pandas.DataFrame, MinMaxScaler)
    """
    x = df[columns]
    minmax = MinMaxScaler()
    df[columns] = minmax.fit_transform(x)
    return df, minmax


def train_oh_encode(df, columns):
    """
    Function that does OneHot encoding of categorical columns specified by columns argument.
    :param df: pandas.DataFrame
    :param columns: list
    :return: Tuple(pandas.DataFrame, OneHotEncoder)
    """
    x = df[columns]
    oh = OneHotEncoder()
    x = oh.fit_transform(x).todense()
    x = pandas.DataFrame(x)
    x = make_categorical_columns(x, oh.categories_, columns)
    df = df.drop(columns=columns)
    df = pandas.concat([df, x], axis=1)
    return df, oh