clc;clearvars;close all;

[X, Y] = load_data();

% Predict and calculate accuracy
C = digit_classify_f(X)
Y'
accuracy = sum(C == Y') / length(Y);
fprintf("Accuracy: %f\n", accuracy);



function [X, Y] = load_data()
    % This function loads the data from the DATA_ROOT directory
    % It returns X, a cell array of size N x 1, where each cell contains a 3 x L matrix representing a sequence of coordinates
    % It also returns Y, a vector of length N, where Y(i) is the class (0-9) of the ith sequence

    paths = get_paths();
    X = {};
    Y = [];
    for i = 1:length(paths)
        path = paths{i};
        data = load(path);
        X{end+1} = data.pos;
        label = get_label(path);
        Y(end+1) = label;
    end
end

function label = get_label(path)
    % label is: path.split("/")[-1].split("_")[0]
    path_split = split(path, "/");
    filename = path_split{end};
    filename_split = split(filename, "_")
    label = str2num(filename_split{2});
end


function paths = get_paths()
    DATA_ROOT = "./digits_3d_training_data/digits_3d/training_data";
    paths = {};
    d = dir(fullfile(DATA_ROOT, '**', '*.mat'));
    for i = 1:length(d)
        paths{end+1} = fullfile(d(i).folder, d(i).name);
    end
end


function C = digit_classify_f(testdata)
    % This function takes in testdata, which contains N 3 x L sequences representing coordinates.
    % It returns a vector C of length N, where C(i) is the predicted class (0-9) of the ith sequence.
    % Since L is not the same for all sequences the input is a cell array of size N x 1, where each cell contains a 3 x L matrix.

    % Secondly, we pad each sequence to 222
    % Or truncate if it is longer than 222
    % We want our full input matrix to be N x 3 x 222
    D = zeros(length(testdata), 222, 3);
    fprintf("Size of D: %d x %d x %d\n", size(D, 1), size(D, 2), size(D, 3));

    for i = 1:length(testdata)
        seq = testdata{i};
        size(seq)
        seq_len = size(seq, 1);
        fprintf("Sequence %d has length %d\n", i, seq_len);
        if seq_len > 222
            fprintf("Sequence %d is too long, truncating\n", i);
            seq = seq(1:222, :);
            seq_len = 222;
        end
        D(i, 1:seq_len, :) = seq;
    end

    fprintf("Size of D: %d x %d x %d\n", size(D, 1), size(D, 2), size(D, 3));

    % Now D is N x 222 x 3, we need to permute it to be N x 3 x 222
    %D = permute(D, [1 3 2]);

    fprintf("Size of D: %d x %d x %d\n", size(D, 1), size(D, 2), size(D, 3));

    % load the keras model
    model = importKerasNetwork("best_model_new.h5")

    % Compute the model input and output sizes
    %inputSize = model.Layers(1).InputSize
    %outputSize = model.Layers(end).OutputSize
    C = zeros(size(D, 1), 1);
    for i = 1:size(D, 1)
        % Predict one sample
        x_sample = D(i, :, :);
        % Remove the batch dimension
        x_sample = reshape(x_sample, [size(x_sample, 2), size(x_sample, 3)]);
        x_sample = x_sample';
        %size(x_sample)
        output1 = predict(model, x_sample);
        [max_val, max_idx] = max(output1);
        C(i) = max_idx - 1;
    end
end
    

