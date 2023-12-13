clc;clearvars;close all;
model = load_model()

[X, Y] = load_data();

% Predict and calculate accuracy
inds = randperm(length(X));
X = X(inds);
Y = Y(inds);
X = X(1:10);
Y = Y(1:10);

C = [];

for ii = [1:length(X)]
    C(end+1) = digit_classify(X{ii},model);

end

accuracy = sum(C == Y) / length(Y);
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
    filename_split = split(filename, "_");
    label = str2num(filename_split{end-1});
end


function paths = get_paths()
    DATA_ROOT = "./digits_3d_training_data/digits_3d/training_data";
    paths = {};
    d = dir(fullfile(DATA_ROOT, '**', '*.mat'));
    for i = 1:length(d)
        paths{end+1} = fullfile(d(i).folder, d(i).name);
    end
end
