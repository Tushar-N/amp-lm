require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.misc'
require 'util.OneHot'

local BatchLoader = require 'util/BatchLoader'
local model_utils = require 'util/model_utils'
local LSTM = require 'model/LSTM'

------------------------------ parameters --------------------------------
cmd = torch.CmdLine()
cmd:option('-data_dir','data/','data directory with residue_lstm_input.txt')
cmd:option('-rnn_size', 300, 'size of LSTM internal state')
cmd:option('-char_size', 50, 'size of char dict')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout', 0.8,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',30,'number of timesteps to unroll for')
cmd:option('-batch_size',20,'number of sequences to train on in parallel')
cmd:option('-max_epochs',100,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.80,'fraction of data that goes into train set')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',100,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',300,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:text()

------------------------------ argument parsing --------------------------------
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

if opt.gpuid>=0 then
	require 'cutorch'
	require 'cunn'
	cutorch.setDevice(opt.gpuid)
end

local split_sizes = {opt.train_frac, 1-opt.train_frac, 0} 

------------------------------ data loading --------------------------------
local loader = BatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
local vocab_size = loader.vocab_size
local vocab = loader.vocab_mapping
print('vocab size: ' .. vocab_size)
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end


------------------------------ model definition --------------------------------
protos={}
protos.rnn = LSTM.lstm(vocab_size, opt.char_size, opt.rnn_size, opt.num_layers, opt.dropout)
protos.criterion = nn.ClassNLLCriterion()


-- the initial state of the cell/hidden states
local h_init = torch.zeros(opt.batch_size, opt.rnn_size)

if opt.gpuid>=0 then
	h_init=h_init:cuda()
	protos.rnn:cuda()
	protos.criterion:cuda()
end

init_state = {}
for L=1,opt.num_layers do
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end



-- flatten parameters into a single tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)
params:uniform(-0.08, 0.08) -- initialize with small random numbers
print('number of parameters in the model: ' .. params:nElement())

-- clone the LSTM for seq_length time steps so each has it's own set of gradients
clones = {}
for name,proto in pairs(protos) do
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-------------------------------------------------------------------------------

-- evaluate the performance on the held out data
function eval_split(split_index, max_batches)

    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end
    loader:reset_batch_pointer(split_index)

    local loss = 0
    local rnn_state = {[0] = init_state} 
    
	-- go over all validation set batches
    for i = 1, n do

        local x, y = loader:next_batch(split_index)
		if opt.gpuid>=0 then
			x, y= x:float():cuda(), y:float():cuda()
		end

		local loss_i
        for t=1, opt.seq_length do
            clones.rnn[t]:evaluate()
            local lst = clones.rnn[t]:forward({x[{{},t}], unpack(rnn_state[t-1])})
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end

            prediction = lst[#lst] 
            loss_i = clones.criterion[t]:forward(prediction, y[{{},t}])
			loss = loss + loss_i
        end

        rnn_state[0] = rnn_state[#rnn_state]
        print(i .. '/' .. n .. '...' .. loss_i)
    end

    loss = loss / opt.seq_length / n
    return loss
end


local init_state_global = clone_list(init_state)
function feval(x)

    if x ~= params then params:copy(x) end
    grad_params:zero() -- initialize all gradients to zero

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
	if opt.gpuid>=0 then
		x, y= x:float():cuda(), y:float():cuda() 
	end

	------------------- book-keeping variables -------------------
	local rnn_state = {[0] = init_state_global} -- init the state to whwere we left off
	local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} 
	local predictions = {}          
    local loss = 0

    ------------------- forward pass -------------------
    for t=1, opt.seq_length do
        clones.rnn[t]:training() -- make sure dropout works properly
        local lst = clones.rnn[t]:forward({x[{{},t}], unpack(rnn_state[t-1])})
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end 
        predictions[t] = lst[#lst]
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{},t}])
    end
    loss = loss / opt.seq_length

    ------------------ backward pass -------------------
    for t=opt.seq_length,1,-1 do
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{},t}])
        table.insert(drnn_state[t], doutput_t)
        local dlst = clones.rnn[t]:backward({x[{{},t}], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
		for k=2, #dlst do table.insert(drnn_state[t-1], dlst[k]) end
    end

	----------- BPTT and gradient clipping --------------
    init_state_global = rnn_state[#rnn_state] 
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

------------------------------ optimization --------------------------------
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local best_val=9999999

-- repeatedly perform optimization on data
for i = 1, iterations do
   
    local epoch = i / loader.ntrain
    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state) -- rmsprop is good for RNNs
    local time = timer:time().real
    local train_loss = loss[1]
   
	-- evaluate performance every few iterations, or at the end
    if i % opt.eval_val_every == 0 or i == iterations then

        local val_loss = eval_split(2) -- 2 = validation
		if val_loss<best_val then
			protos.rnn:clearState()
			best_val=val_loss
			os.execute(string.format('rm %s/*.t7', opt.checkpoint_dir))
			print ('best_loss: ', val_loss)
		    local savefile = string.format('%s/lm_epoch%.2f_%.4f.t7', opt.checkpoint_dir, epoch, val_loss)
		    local checkpoint = {protos=protos, opt=opt, val_loss=val_loss, vocab=loader.vocab_mapping, epoch=epoch, i=i}
		    torch.save(savefile, checkpoint)
		end
    end

	-- print progress every few iterations
    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end
   
	-- clear memory
    if i % 200 == 0 then collectgarbage() end 
end
