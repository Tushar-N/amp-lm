require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util/misc'
require 'util.OneHot'

local AMP_BatchLoader = require 'util/AMP_BatchLoader'
local model_utils = require 'util/model_utils'
local LSTM = require 'model/LSTM'

cmd = torch.CmdLine()
cmd:option('-data_dir','data/','data directory with input.seqs')
cmd:option('-load','nil','model to load')
cmd:option('-rnn_size', 300, 'size of LSTM internal state')
cmd:option('-char_size', 50, 'size of char dict')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0.8,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-batch_size',12,'number of sequences to train on in parallel')
cmd:option('-max_epochs',1000,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.80,'fraction of data that goes into train set')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',100,'every how many iterations should we evaluate on validation data?')
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
local loader = AMP_BatchLoader.create(opt.data_dir, opt.batch_size, split_sizes)
local vocab, vocab_size = loader.vocab, loader.vocab_size
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

------------------------------ model definition --------------------------------
protos = {}
protos.fwd = LSTM.lstm(vocab_size, opt.char_size, opt.rnn_size, opt.num_layers, opt.dropout)
protos.bwd = LSTM.lstm(vocab_size, opt.char_size, opt.rnn_size, opt.num_layers, opt.dropout)
protos.mic_estimator= nn.Sequential() --2 layer regression
					:add(nn.JoinTable(2))
					:add(nn.Linear(2*opt.rnn_size, loader.output_size))
					:add(nn.Sigmoid())
					
protos.mic_criterion=nn.MSECriterion()

-- the initial state of the cell/hidden states
init_state = {}
local h_init = torch.zeros(opt.batch_size, opt.rnn_size)

if opt.gpuid>=0 then
	h_init=h_init:cuda()
	protos.fwd:cuda()
	protos.bwd:cuda()
	protos.mic_estimator:cuda()
	protos.mic_criterion:cuda()
end

for L=1,opt.num_layers do
	table.insert(init_state, h_init:clone())
	table.insert(init_state, h_init:clone())
end


params, grad_params = model_utils.combine_all_parameters(protos.fwd, protos.bwd, protos.mic_estimator)
params:uniform(-0.08, 0.08) -- small numbers uniform

print('number of parameters in the model: ' .. params:nElement())

-- load an old model???
if opt.load~='nil' then
	checkpoint = torch.load(opt.load)
	protos = checkpoint.protos
end


-- clone both the forward and backward LSTMs
clones = {}
for name,proto in pairs(protos) do
	if name~='mic_estimator' and name~='mic_criterion' then
		print('cloning ' .. name)
    	clones[name] = model_utils.clone_many_times(proto, loader.max_seq_length, not proto.parameters)
	end
end

-------------------------------------------------------------------------------

-- evaluate the performance on the held out data
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    
    for i = 1,n do -- iterate over batches in the split

        local x, y = loader:next_batch(1)
		if opt.gpuid>=0 then
			x, y = x:float():cuda(), y:float():cuda()
		end
	
		local seq_length=x:size(2)
		local fwd_state, bwd_state = {[0] = init_state}, {[seq_length+1] = init_state}

		local fwdh_out, bwdh_out
        for t=1, seq_length do
            clones.fwd[t]:evaluate()
			clones.bwd[t]:evaluate()
			local _t=seq_length-t+1

		    local fwdlst = clones.fwd[t]:forward{x[{{}, t}], unpack(fwd_state[t-1])}
			local bwdlst = clones.bwd[_t]:forward{x[{{}, _t}], unpack(bwd_state[_t+1])}

			fwdh_out, bwdh_out=fwdlst[#fwdlst]:clone(), bwdlst[#bwdlst]:clone()
		    fwd_state[t], bwd_state[_t] = {}, {}
		    for i=1,#init_state do
				table.insert(fwd_state[t], fwdlst[i]) 
				table.insert(bwd_state[_t], bwdlst[i])
			end
		end

		protos.mic_estimator:evaluate()
		local prediction=protos.mic_estimator:forward({fwdh_out, bwdh_out})
		local loss_i=protos.mic_criterion:forward(prediction, y)
		loss=loss+loss_i

        -- carry over lstm state
        --fwd_state[0] = fwd_state[#fwd_state]
		--bwd_state[seq_length+1] = bwd_state[1]
        print(i .. '/' .. n .. '...' .. loss_i)
    end

    loss = loss / n
    return loss
end


local fwdstate_glob, bwdstate_glob = clone_list(init_state), clone_list(init_state)
function feval(x)
    if x ~= params then params:copy(x) end
    grad_params:zero() -- initialize all gradients to zero

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
	if opt.gpuid>=0 then
    	x, y = x:float():cuda(), y:float():cuda()
	end

	------------------- book-keeping variables -------------------
	local seq_length=x:size(2)
	local fwdh_out, bwdh_out
	protos.mic_estimator:training()
	local fwd_state, bwd_state = {[0] = fwdstate_glob}, {[seq_length+1] = bwdstate_glob}

	local dfwd_state = {[seq_length] = clone_list(init_state, true)}
	local dbwd_state = {[1] = clone_list(init_state, true)}

    ------------------- forward pass -------------------
   
    for t=1, seq_length do
        clones.fwd[t]:training()
		clones.bwd[t]:training()
		local _t=seq_length-t+1
        local fwdlst = clones.fwd[t]:forward{x[{{}, t}], unpack(fwd_state[t-1])}
		local bwdlst = clones.bwd[_t]:forward{x[{{}, _t}], unpack(bwd_state[_t+1])}
		fwdh_out, bwdh_out=fwdlst[#fwdlst]:clone(), bwdlst[#bwdlst]:clone()
        fwd_state[t], bwd_state[_t] = {}, {}
        for i=1,#init_state do
			table.insert(fwd_state[t], fwdlst[i]) 
			table.insert(bwd_state[_t], bwdlst[i])
		end
    end
	
	------------------- predict MIC values -------------------
	local prediction=protos.mic_estimator:forward({fwdh_out, bwdh_out})
	local loss=protos.mic_criterion:forward(prediction, y)
	local dpred=protos.mic_criterion:backward(prediction, y)
	local dlstm=protos.mic_estimator:backward({fwdh_out, bwdh_out}, dpred)

    ------------------ backward pass -------------------
  
    for t=seq_length,1,-1 do
		
		local _t=seq_length-t+1
		if t==seq_length then --propogate mic_estimate errors
			dfwd_state[t][#dfwd_state[t]]:copy(dlstm[1])
			dbwd_state[_t][#dbwd_state[_t]]:copy(dlstm[2])
		end
        local fwd_dlst = clones.fwd[t]:backward({x[{{}, t}], unpack(fwd_state[t-1])}, dfwd_state[t])
        local bwd_dlst = clones.bwd[_t]:backward({x[{{}, _t}], unpack(bwd_state[_t+1])}, dbwd_state[_t])

		dfwd_state[t-1], dbwd_state[_t+1] = {}, {}
		for k=2, #fwd_dlst do table.insert(dfwd_state[t-1], fwd_dlst[k]) end
		for k=2, #bwd_dlst do table.insert(dbwd_state[_t+1], bwd_dlst[k]) end

	end

    ----------- BPTT and gradient clipping --------------
 --    fwdstate_glob = fwd_state[#fwd_state]
	-- bwdstate_glob = bwd_state[1]
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


    if epoch>1 and epoch%1000==0 then
    	optim_state['learningRate'] = optim_state['learningRate']/10.0
    	print ('lr set to '.. optim_state['learningRate'])
    end

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] 
   
    -- evaluate performance every few iterations, or at the end
    if i % opt.eval_val_every == 0 or i == iterations then
        local val_loss = eval_split(2) -- 2 = validation
		if val_loss<best_val then
			best_val=val_loss
			os.execute(string.format('rm %s/*.t7', opt.checkpoint_dir))
			print ('best_loss: ', val_loss)
        	local savefile = string.format('%s/lm_epoch%.2f_%.7f.t7', opt.checkpoint_dir, epoch, val_loss)
            local checkpoint = {protos=protos, opt=opt, val_loss=val_loss, vocab=loader.vocab, epoch=epoch, i=i}
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


