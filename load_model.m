function model = load_model()
% Used to load pretrained tensorflow model to matlab
    model = importNetworkFromTensorFlow('model_pb')
end

