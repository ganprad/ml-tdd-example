from mvalidators.data_schema import InputDataFrameSchema, PreprocessedDataSchema


def test_dataframe_schema(test_df, test_model):
    temp = test_model.preprocess(InputDataFrameSchema(test_df))
    temp = test_model.train_encode(temp)
    PreprocessedDataSchema(temp)
