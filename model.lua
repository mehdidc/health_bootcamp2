require 'dp'
require 'nn'

function model(X_train, y_train, X_test)
    local opt = {
        hiddenSize = {200, 200},
        activation = 'ReLU',
        dropout = {0, 0.5},
        sparse_init = false,
        momentum = 0.8,
        learningRate = 0.1,
--      maxOutNorm = 1.,
        batchSize = 100,
        weight_decay_factor = 0.,
        cuda = false,
        maxEpoch = 20
    }
    
    if torch.min(y_train) == 0 then
        y_train = y_train + 1
    end

    inputSize = X_train:size(2)
    classes = {1, 2}

    outputSize = #classes

    -- Construct the data object
    x_train_v = construct_x_view(X_train)
    y_train_v = construct_y_view(y_train, classes)
    datasource = dp.DataSource{
        train_set = dp.DataSet{
            which_set='train',
            inputs=x_train_v,
            targets=y_train_v,
        },
        input_preprocess={dp.Standardize()}
    }
    datasource._classes = classes
    datasource._name = 'data'
    datasource._feature_size = inputSize
    

    -- Setup the model
    mlp = dp.Sequential()
    for i , hiddenSize in ipairs(opt.hiddenSize) do
       local dense = dp.Neural{
          input_size = inputSize, 
          output_size = hiddenSize,
          transfer = nn[opt.activation](),
          dropout =  (opt.dropout and (opt.dropoutProb[i] or 0) > 0 and 
                      nn.Dropout(opt.dropoutProb[depth])),
          sparse_init = opt.sparse_init
       }
       mlp:add(dense)
       inputSize = hiddenSize
    end

    mlp:add(
       dp.Neural{
          input_size = inputSize, 
          output_size = #(datasource:classes()),
          transfer = nn.LogSoftMax(),
          sparse_init = opt.sparse_init
       }
    )
    


    --[[Propagators]]--
    train = dp.Optimizer{
       loss = dp.NLL(),
       visitor = { -- the ordering here is important:
          dp.Momentum{momentum_factor = opt.momentum},
          dp.Learn{
                learning_rate = opt.learningRate,
                observer = {dp.AdaptiveLearningRate{}}
          },
          dp.MaxNorm{max_out_norm = opt.maxOutNorm},
          dp.WeightDecay{wd_factor = opt.weight_decay_factor}
       },
       feedback = dp.Confusion(),
       sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
       progress = false
    }
    valid = dp.Evaluator{
       loss = dp.NLL(),
       feedback = dp.Confusion(),  
       sampler = dp.Sampler{}
    }
    test = dp.Evaluator{
       loss = dp.NLL(),
       feedback = dp.Confusion(),
       sampler = dp.Sampler{}
    }    

    --[[Experiment]]--
    xp = dp.Experiment{
       model = mlp,
       optimizer = train,
--       validator = valid,
--       tester = test,
       observer = {
           dp.FileLogger()
       },
       random_seed = os.time(),
       max_epoch = opt.maxEpoch
    }

    --[[GPU or CPU]]--
    if opt.cuda then
       require 'cutorch'
       require 'cunn'
       cutorch.setDevice(opt.useDevice)
       xp:cuda()
    end
        
    xp:run(datasource)
    -- Predict
    x_test = construct_x_view(X_test)

    test_set = dp.DataSet{
        which_set='test',
        inputs=x_test,
    }
    
    test_set:carry():putObj('nSample', X_test:size(1))
    y_test, carry = mlp:evaluate(x_test, test_set:carry())

    proba = y_test._input
    targets = argmax_2D(proba) - 1
    return targets, proba
end

function construct_x_view(x)
    local x_v
    x_v = dp.DataView()
    x_v:forward('bf', x)
    return x_v
end
function construct_y_view(y, classes)
    local y_v
    y_v = dp.ClassView()
    y_v:forward('b', y)
    y_v:setClasses(classes)
    return y_v
end


