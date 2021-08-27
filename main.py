import neiro_lib
import preprocessing
import argparse
import pandas as pd


def get_filename():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='path to input file')
    parser.add_argument('output', help='path to output file')
    args = parser.parse_args()
    return str(args.input), str(args.output)


def main():
    input_file, output_file = get_filename()
    df = preprocessing.read_file(input_file)

    all_data, train_data, target_data = preprocessing.preprocess_data(df)

    X_train, Y_train = preprocessing.get_time_feature(train_data)
    X, Y = preprocessing.get_time_feature(target_data)

    model = neiro_lib.init_model()
    model =neiro_lib.load_model(__file__[0:-8] + "\model", model)

    neiro_lib.train_model(X_train,Y_train,model)

    Y = []
    print("Производится вычисление")
    for i in range(int(X.size/7)):
        y = neiro_lib.predict_model(X[i,:], model)
        Y.append(y)
        if target_data.index[i]+1 in target_data.index:
            X[i+1,6] = y
    target_data['A+'] = Y

    result = all_data.append(target_data)
    result = result.sort_index()
    result = result.drop("A_prev", axis=1)
    result.to_excel(excel_writer=output_file)


if __name__ == '__main__':
    main()


