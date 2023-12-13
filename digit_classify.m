function label = digit_classify(X,varargin)
    %   DIGIT_CLASSIFY Summary of this function goes here
    %   Uses pretrained CNN to classify 3d digits
    %   
    %   Parameters
    %   X: N x 3 input matrix
    %   model (optional): input preloaded model to speed up function. 
    %   Use function load_model() to load model

    if ~isempty(varargin)
        model = varargin{1};
    else
        model = importNetworkFromTensorFlow('model_pb');
    end
    D = zeros(222, 3);


    
    N = size(X, 1);
    % Truncated input
    if N > 222
        X = X(1:222, :);
        N = 222;
    end
    % Zero padding
        D(1:N, :) = X;

    output1 = predict(model, D);
    [max_val, max_idx] = max(output1);
    label = max_idx - 1;
end

